import os
import faiss
import numpy as np
import torch
from typing import Optional, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from config import MODE, API_PROVIDER, MOONSHOT_API_KEY  # 引入配置

# ===== 配置读取 =====
# 根据全局模式自动切换本地/API模式
USE_LOCAL_MODEL = (MODE == "local")
LOCAL_MODEL = "Qwen/Qwen1.5-7B-Chat"  # 本地模型 fallback

# 缓存，避免重复加载
_model_cache = {}
_sbert_model = None

def get_sbert_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer(model_name)
    return _sbert_model

def load_index(index_dir="index"):
    index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
    with open(os.path.join(index_dir, "chunks.txt"), "r", encoding="utf-8") as f:
        chunks = f.readlines()
    return index, chunks

def search_index(query, top_k=3, index_dir="index"):
    model = get_sbert_model("sentence-transformers/all-MiniLM-L6-v2")
    q_emb = model.encode([query], convert_to_numpy=True)
    index, chunks = load_index(index_dir)
    D, I = index.search(q_emb, top_k)
    results = [chunks[i].strip() for i in I[0]]
    return results

# -------------------------
# 本地模型加载（保持不变）
# -------------------------
def _try_load_local_model(model_name: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """加载本地模型，返回 tokenizer 和 model"""
    global _model_cache
    if model_name in _model_cache:
        return _model_cache[model_name]

    try:
        kwargs = {}
        try:
            import bitsandbytes as bnb  # noqa: F401
            load_4bit = True
        except Exception:
            load_4bit = False

        if load_4bit:
            from transformers import BitsAndBytesConfig
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=quantization_config,
                dtype=torch.float16,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        _model_cache[model_name] = (tokenizer, model)
        return tokenizer, model

    except ValueError as e:
        msg = str(e).lower()
        if "requires `accelerate`" in msg or "device_map" in msg:
            raise RuntimeError(
                "加载模型需要 `accelerate`。请先运行：\n\npip install --upgrade accelerate\n\n"
                "如果你想用 4-bit 量化以降低显存需求，请同时安装 bitsandbytes：\n\npip install bitsandbytes\n\n"
                "或者把 MODE 设置为 'api' 使用远端/API 模型。"
            ) from e
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            _model_cache[model_name] = (tokenizer, model)
            return tokenizer, model
        except Exception as e2:
            raise RuntimeError(f"尝试加载模型时多次失败：{e2}") from e2

# -------------------------
# API 调用实现
# -------------------------
def _call_moonshot_api(query, context):
    """调用 Moonshot (Kimi) API 生成答案"""
    if not MOONSHOT_API_KEY:
        return "❌ 请在 config.py 中配置 MOONSHOT_API_KEY"
    
    try:
        import requests
        headers = {
            "Authorization": f"Bearer {MOONSHOT_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "moonshot-v1-8k",  # 或 'moonshot-v1-32k' 支持更长上下文
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的文档问答助手。请基于给定的文档内容回答用户问题。"
                },
                {
                    "role": "user",
                    "content": f"基于以下文档内容：\n\n{context}\n\n回答问题：{query}"
                }
            ],
            "temperature": 0.7,
            "stream": False
        }
        
        response = requests.post(
            "https://api.moonshot.cn/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ API 调用失败：{str(e)}"
    try:
        import requests  # 确保已安装：pip install requests
    except ImportError:
        return "❌ 请先安装 requests：pip install requests"

    # 构建提示词（遵循 RAG 逻辑：用文档片段限定回答范围）
    prompt = f"""请根据以下文档内容回答问题，不要编造信息：

文档内容：
{context}

问题：{query}
"""

    # 调用 Kimi API（参考官方文档：https://platform.moonshot.cn/docs/api）
    url = "https://api.moonshot.cn/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MOONSHOT_API_KEY}"
    }
    data = {
        "model": "moonshot-v1-8k",  # 可替换为其他模型如 moonshot-v1-32k
        "messages": [
            {"role": "system", "content": "你是一个基于文档内容的问答助手，仅根据提供的文档片段回答问题。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,  # 控制回答随机性，0.3 较严谨
        "max_tokens": 1024
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # 抛出 HTTP 错误
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ API 调用失败：{str(e)}"

# -------------------------
# 主生成函数（整合本地/API）
# -------------------------
def generate_answer(query, context):
    """根据配置自动选择本地模型或 API 生成答案"""
    if not USE_LOCAL_MODEL:
        if API_PROVIDER == "moonshot":
            return _call_moonshot_api(query, context)
        else:
            return f"❌ 不支持的 API 提供商：{API_PROVIDER}。目前仅支持 'moonshot'"
    
    # 本地模型逻辑（保持不变）
    prompt = f"根据以下文档内容回答问题：\n\n{context}\n\n问题：{query}\n回答："

    try:
        # transformers已在文件开头导入
        tokenizer, model = _try_load_local_model(LOCAL_MODEL)
    except RuntimeError as e:
        return (
            "⚠️ 本地模型加载失败：\n\n"
            f"{e}\n\n"
            "建议：\n"
            "1) 先运行 `pip install --upgrade accelerate`，然后重试。\n"
            "2) 如果要使用 4-bit，运行 `pip install bitsandbytes`（Windows 上可能需要匹配 CUDA 的 wheel）。\n"
            "3) 在 config.py 中设置 MODE='api' 并配置 API 密钥使用 API。\n"
        )

    device = 0 if torch.cuda.is_available() else -1
    gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    out = gen_pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
    return out[0]["generated_text"]