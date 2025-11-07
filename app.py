import streamlit as st
import os
import hashlib
from pathlib import Path
from utils import is_valid_file, init_logger
from ingest import build_index
from qa import search_index, generate_answer

# 初始化日志
logger = init_logger()

# 设置页面
st.set_page_config(page_title="文档问答助手", layout="wide")
st.title("📘 文档问答助手 (RAG)")

# 安全地创建上传目录
upload_dir = Path("uploads")
upload_dir.mkdir(exist_ok=True)

# 文件上传区域
uploaded_file = st.file_uploader("上传 PDF 或 Markdown 文件", type=["pdf", "md"])
if uploaded_file is not None:
    # 文件验证
    is_valid, message = is_valid_file(uploaded_file)
    if not is_valid:
        st.error(message)
        st.stop()
    
    try:
        # 使用文件哈希作为文件名，避免路径注入
        content = uploaded_file.getbuffer()
        file_hash = hashlib.sha256(content).hexdigest()[:16]
        file_ext = Path(uploaded_file.name).suffix
        safe_filename = f"{file_hash}{file_ext}"
        file_path = upload_dir / safe_filename
        
        # 安全写入文件
        file_path.write_bytes(content)
        logger.info(f"成功保存文件：{safe_filename}")

        # 构建索引（带进度提示）
        with st.spinner("正在构建文档索引..."):
            try:
                build_index(str(file_path))
                st.success("✅ 索引构建完成！")
            except Exception as e:
                logger.error(f"索引构建失败：{e}")
                st.error(f"索引构建失败：{str(e)}")
                st.stop()
    except Exception as e:
        logger.error(f"文件处理失败：{e}")
        st.error(f"文件处理失败：{str(e)}")
        st.stop()

# 问答区域
query = st.text_input("请输入你的问题：", key="qa_input")  # 添加唯一的key
if query and query.strip():  # 确保输入非空
    try:
        with st.spinner("正在思考..."):
            results = search_index(query)
            if not results:  # 检查是否有检索结果
                st.warning("⚠️ 未找到相关内容，请尝试换个问法")
                st.stop()
            
            context = "\n".join(results)
            answer = generate_answer(query, context)
            
            # 展示结果
            st.write("💡 回答：")
            st.write(answer)
            
            # 可选：展示相关文档片段
            with st.expander("📚 参考文档片段"):
                for i, chunk in enumerate(results, 1):
                    st.markdown(f"**片段 {i}**：\n{chunk}\n---")
    except Exception as e:
        logger.error(f"问答过程出错：{e}")
        st.error(f"处理失败：{str(e)}")
        # 提供更多上下文帮助用户理解错误
        if "connection" in str(e).lower():
            st.error("API 连接失败，请检查网络连接和 API 密钥配置")
        elif "api key" in str(e).lower():
            st.error("API 密钥无效或未配置，请检查配置文件")
        st.stop()
