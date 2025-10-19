import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def load_document(file_path: str) -> str:
    """读取 PDF 或 Markdown 文档，返回纯文本"""
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
    elif file_path.endswith(".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError("只支持 PDF 和 Markdown")
    return text

def split_text(text: str, chunk_size=500, overlap=50):
    """切分文档为小块"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def build_index(file_path: str, index_dir="index"):
    """构建 FAISS 索引"""
    text = load_document(file_path)
    chunks = split_text(text)

    # 向量化模型（轻量，8GB 显存可跑）
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = model.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]

    # FAISS 索引
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))

    with open(os.path.join(index_dir, "chunks.txt"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.replace("\n", " ") + "\n")

    print("✅ 文档索引构建完成！")
