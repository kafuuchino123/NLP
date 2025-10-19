# 📘 文档问答助手 (RAG)

一个基于 **RAG 技术** 的文档问答系统，支持 PDF / Markdown 文件。

## 功能
- 上传文档 → 自动切分 → 构建向量索引
- 输入问题 → 检索相关片段 → LLM 生成答案
- 支持 **本地模型 (8GB 显存可运行)** 或 **国内 API**

## 技术栈
- [Streamlit] 前端交互
- [FAISS] 向量数据库
- [Sentence-Transformers] 文本嵌入
- [Transformers] 本地大模型推理
- [PyPDF2 / markdown] 文档解析

## 运行方式
```bash
pip install -r requirements.txt
streamlit run app.py
