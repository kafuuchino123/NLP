import streamlit as st
import os
from ingest import build_index
from qa import search_index, generate_answer

st.set_page_config(page_title="文档问答助手", layout="wide")
st.title("📘 文档问答助手 (RAG)")

if not os.path.exists("uploads"):
    os.makedirs("uploads")

uploaded_file = st.file_uploader("上传 PDF 或 Markdown 文件", type=["pdf", "md"])
if uploaded_file is not None:
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("正在构建文档索引...")
    build_index(file_path)
    st.success("✅ 索引构建完成！")

query = st.text_input("请输入你的问题：")
if query:
    results = search_index(query)
    context = "\n".join(results)
    answer = generate_answer(query, context)
    st.write("💡 回答：")
    st.write(answer)
