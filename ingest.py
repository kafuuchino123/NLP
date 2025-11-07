import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm

from PyPDF2 import PdfReader
import nltk
from sentence_transformers import SentenceTransformer
import faiss

# 下载必要的 NLTK 数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class IndexMetadata:
    """索引元数据"""
    file_path: str
    file_name: str
    created_at: str
    chunk_size: int
    overlap: int
    total_chunks: int
    embedding_model: str
    file_hash: str

def compute_file_hash(file_path: str) -> str:
    """计算文件的 SHA-256 哈希值"""
    import hashlib
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]

def clean_text(text: str) -> str:
    """清理文本，移除特殊字符和多余空白"""
    # 替换常见的特殊字符为空格
    text = re.sub(r'[_*#\[\]()~>`]', ' ', text)
    # 规范化空白字符
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_document(file_path: str) -> str:
    """读取 PDF 或 Markdown 文档，返回清理后的纯文本"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")
        
    try:
        if file_path.suffix.lower() == '.pdf':
            reader = PdfReader(file_path)
            text = []
            for page in tqdm(reader.pages, desc="读取 PDF"):
                content = page.extract_text() or ""
                text.append(clean_text(content))
            return "\n".join(text)
            
        elif file_path.suffix.lower() in ['.md', '.markdown']:
            with open(file_path, "r", encoding="utf-8") as f:
                return clean_text(f.read())
        else:
            raise ValueError("只支持 PDF 和 Markdown 文件")
            
    except Exception as e:
        raise RuntimeError(f"读取文件 {file_path.name} 失败: {str(e)}") from e


# 模型缓存，避免多次加载
_sbert_model = None

def get_sbert_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer(model_name)
    return _sbert_model

def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """智能切分文档为句子块
    
    使用 NLTK 进行句子分割，然后将句子组合成适当大小的块，
    确保不会在句子中间截断。
    """
    # 分句
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        sentence_length = len(sentence)
        
        # 如果单个句子超过块大小，按词切分
        if sentence_length > chunk_size:
            words = sentence.split()
            temp_chunk = []
            temp_length = 0
            
            for word in words:
                word_length = len(word) + 1  # +1 for space
                if temp_length + word_length > chunk_size:
                    if temp_chunk:
                        chunks.append(" ".join(temp_chunk))
                    temp_chunk = [word]
                    temp_length = word_length
                else:
                    temp_chunk.append(word)
                    temp_length += word_length
                    
            if temp_chunk:
                chunks.append(" ".join(temp_chunk))
            continue
            
        # 正常句子处理
        if current_length + sentence_length + 1 <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length + 1
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
            
    # 添加最后一个块
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    # 处理重叠
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i in range(len(chunks)):
            if i > 0:
                # 从前一个块的末尾获取重叠部分
                prev_words = chunks[i-1].split()[-overlap//10:]  # 重叠词数而不是字符
                current_chunk = " ".join(prev_words) + " " + chunks[i]
                overlapped_chunks.append(current_chunk)
            else:
                overlapped_chunks.append(chunks[i])
        chunks = overlapped_chunks
        
    return chunks

def build_index(
    file_path: str,
    index_dir: str = "index",
    chunk_size: int = 500,
    overlap: int = 50,
    batch_size: int = 32
) -> IndexMetadata:
    """构建 FAISS 索引
    
    Args:
        file_path: 文档路径
        index_dir: 索引存储目录
        chunk_size: 文本块大小
        overlap: 块间重叠长度
        batch_size: 向量化批次大小
    
    Returns:
        IndexMetadata: 索引元数据
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. 加载和处理文档
        print("📚 正在读取文档...")
        text = load_document(file_path)
        
        # 2. 文本分块
        print("✂️ 正在智能分块...")
        chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            raise ValueError("文档处理后没有有效内容")
        
        # 3. 向量化（批处理以节省内存）
        print("🔢 正在生成向量表示...")
        model = get_sbert_model()
        embeddings_list = []
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="向量化进度"):
            batch = chunks[i:i + batch_size]
            batch_embeddings = model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings_list.append(batch_embeddings)
            
        embeddings = np.vstack(embeddings_list)
        
        # 4. 构建 FAISS 索引
        print("🔍 正在构建检索索引...")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        
        # 5. 保存索引和块数据
        print("💾 正在保存索引文件...")
        
        # 生成元数据
        metadata = IndexMetadata(
            file_path=str(file_path),
            file_name=Path(file_path).name,
            created_at=datetime.now().isoformat(),
            chunk_size=chunk_size,
            overlap=overlap,
            total_chunks=len(chunks),
            embedding_model=model._model_name if hasattr(model, '_model_name') else "sentence-transformers/all-MiniLM-L6-v2",
            file_hash=compute_file_hash(file_path)
        )
        
        # 保存所有文件
        faiss.write_index(index, str(index_dir / "faiss.index"))
        
        with open(index_dir / "chunks.txt", "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk.replace("\n", " ") + "\n")
                
        with open(index_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, ensure_ascii=False, indent=2)
            
        print(f"✅ 索引构建完成！共处理 {len(chunks)} 个文本块")
        return metadata
        
    except Exception as e:
        print(f"❌ 索引构建失败：{str(e)}")
        # 清理可能的部分文件
        for file in ["faiss.index", "chunks.txt", "metadata.json"]:
            try:
                (index_dir / file).unlink(missing_ok=True)
            except Exception:
                pass
        raise
