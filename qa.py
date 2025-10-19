import os
import faiss
import numpy as np
import torch

from sentence_transformers import SentenceTransformer

# LLM 相关
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ===== 可选配置 =====
USE_LOCAL_MODEL = True   # True 本地模型，False 使用 API（请自己实现 API 调用）
LOCAL_MODEL = "Qwen/Qwen1.5-7B-Chat"   # 若内存不足可换更小模型
# LOCAL_MODEL = "gpt2"  # 测试时可临时用更小模型在 CPU 上跑

# 缓存，避免每次请求都重新 load
_model_cache = {}

def load_index(index_dir="index"):
    index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
    with open(os.path.join(index_dir, "chunks.txt"), "r", encoding="utf-8") as f:
        chunks = f.readlines()
    return index, chunks

def search_index(query, top_k=3, index_dir="index"):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q_emb = model.encode([query], convert_to_numpy=True)
    index, chunks = load_index(index_dir)
    D, I = index.search(q_emb, top_k)
    results = [chunks[i].strip() for i in I[0]]
    return results

# -------------------------
# LLM 加载与生成（稳健）
# -------------------------
def _try_load_local_model(model_name):
    """尝试多种方式加载本地模型，返回 (tokenizer, model)"""
    global _model_cache
    if model_name in _model_cache:
        return _model_cache[model_name]

    # 优先尝试：device_map="auto" + 4-bit（如果 bitsandbytes 可用）
    try:
        kwargs = {}
        # 如果 bitsandbytes 可用，优先使用 4-bit 量化以节省显存
        try:
            import bitsandbytes as bnb  # noqa: F401
            load_4bit = True
        except Exception:
            load_4bit = False

        if load_4bit:
            # 需要 bitsandbytes + transformers 支持
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_4bit=True,
                torch_dtype=torch.float16,
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
        # 常见报错：没有 install accelerate -> 给用户明确提示
        msg = str(e).lower()
        if "requires `accelerate`" in msg or "device_map" in msg:
            raise RuntimeError(
                "加载模型需要 `accelerate`。请先运行：\n\npip install --upgrade accelerate\n\n"
                "如果你想用 4-bit 量化以降低显存需求，请同时安装 bitsandbytes：\n\npip install bitsandbytes\n\n"
                "或者把 USE_LOCAL_MODEL 设置为 False 使用远端/API 模型。"
            ) from e
        # 其他 ValueError：回退到不使用 device_map 的加载（可能内存不够）
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            _model_cache[model_name] = (tokenizer, model)
            return tokenizer, model
        except Exception as e2:
            raise RuntimeError(f"尝试加载模型时多次失败：{e2}") from e2

def generate_answer(query, context):
    """使用 LLM 生成答案。若本地模型不可用会返回友好提示"""
    if not USE_LOCAL_MODEL:
        # 你可以在这里实现对接国内 API 的代码并返回 API 回复
        return f"(提示) 当前设置为使用 API，但还未实现 API 调用。问题：{query}\n上下文片段：{context[:200]}"

    # 拼接 prompt（可按需改进）
    prompt = f"根据以下文档内容回答问题：\n\n{context}\n\n问题：{query}\n回答："

    try:
        tokenizer, model = _try_load_local_model(LOCAL_MODEL)
    except RuntimeError as e:
        # 不能加载本地模型，给出明确指引
        return (
            "⚠️ 本地模型加载失败：\n\n"
            f"{e}\n\n"
            "建议：\n"
            "1) 先运行 `pip install --upgrade accelerate`，然后重试。\n"
            "2) 如果要使用 4-bit，运行 `pip install bitsandbytes`（Windows 上可能需要匹配 CUDA 的 wheel）。\n"
            "3) 临时设置 USE_LOCAL_MODEL=False 使用 API。\n"
        )

    # 使用 pipeline 做生成（device 参数交给 pipeline 自动处理）
    device = 0 if torch.cuda.is_available() else -1
    gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    out = gen_pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
    return out[0]["generated_text"]
