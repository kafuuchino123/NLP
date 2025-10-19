# 模型/接口配置

# 选择模式: "local" (本地大模型) 或 "api" (国内API)
MODE = "local"

# 如果选择 API 模式，配置国内可用的API
API_PROVIDER = "zhipu"  # 可选: "zhipu", "moonshot"

# 智谱AI API Key (https://open.bigmodel.cn/)
ZHIPU_API_KEY = "你的智谱AI_API_KEY"

# Moonshot (Kimi) API Key (https://platform.moonshot.cn/)
MOONSHOT_API_KEY = "你的Moonshot_API_KEY"

# 本地模型路径 (huggingface 下载后放本地)
LOCAL_MODEL_NAME = "Qwen/Qwen2-7B-Instruct-GGUF"
LOCAL_MODEL_FILE = "qwen2-7b-instruct-q4_k_m.gguf"  # 量化模型
