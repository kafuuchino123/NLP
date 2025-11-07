import os
import logging
from dotenv import load_dotenv  # 加载.env文件

# 初始化全局logger
logger = logging.getLogger("doc-qa-bot")

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

# 检查文件是否合法（大小、格式、安全性）
def is_valid_file(file, max_size_mb=50):
    if not file:
        return False, "未选择文件"
        
    # 检查文件名安全性
    try:
        filename = file.name
        if not filename or ".." in filename or "/" in filename or "\\" in filename:
            return False, "文件名不合法"
    except Exception:
        return False, "无法读取文件名"
        
    # 检查大小
    try:
        size = file.size
        max_size = max_size_mb * 1024 * 1024
        if size > max_size:
            return False, f"文件大小超过{max_size_mb}MB，请压缩后上传"
        if size == 0:
            return False, "文件为空"
    except Exception:
        return False, "无法获取文件大小"
        
    # 检查格式（区分大小写）
    allowed_extensions = {".pdf", ".PDF", ".md", ".MD", ".markdown", ".MARKDOWN"}
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        return False, "不支持的文件格式，仅支持PDF/Markdown"
        
    return True, "文件格式合法"

# 清理过期的上传文件
def cleanup_old_files(directory, max_age_hours=24):
    """清理指定时间之前的上传文件"""
    try:
        from pathlib import Path
        import time
        
        now = time.time()
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return
            
        for file_path in dir_path.glob("*"):
            if not file_path.is_file():
                continue
                
            age_hours = (now - file_path.stat().st_mtime) / 3600
            if age_hours > max_age_hours:
                try:
                    file_path.unlink()
                    logger.info(f"已清理过期文件：{file_path.name}")
                except Exception as e:
                    logger.warning(f"清理文件失败 {file_path.name}: {e}")
    except Exception as e:
        logger.error(f"清理过期文件时出错：{e}")