import os
import logging
from dotenv import load_dotenv  # 加载.env文件

# 加载环境变量
def load_environment():
    load_dotenv()  # 加载.env中的变量（如OPENAI_API_KEY）
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("请在.env文件中配置OPENAI_API_KEY")
    return openai_api_key

# 初始化日志
def init_logger():
    logger = logging.getLogger("doc-qa-bot")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# 检查文件是否合法（大小、格式）
def is_valid_file(file, max_size_mb=50):
    # 检查大小（最大50MB）
    max_size = max_size_mb * 1024 * 1024  # 转换为字节
    if file.size > max_size:
        return False, f"文件大小超过{max_size_mb}MB，请压缩后上传"
    # 检查格式
    if file.name.endswith((".pdf", ".md")):
        return True, "文件格式合法"
    else:
        return False, "不支持的文件格式，仅支持PDF/Markdown"