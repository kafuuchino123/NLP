# 📘 文档问答助手 (RAG)

一个基于 **RAG (检索增强生成) 技术** 的本地文档问答系统，支持 PDF 和 Markdown 文件的智能问答。

## ✨ 特性

- 📑 **智能文档处理**
  - 支持 PDF 和 Markdown 格式
  - 基于句子的智能分块
  - 自动清理和标准化文本
  - 保持文本语义完整性

- 🔍 **高效检索**
  - 使用 FAISS 进行相似度检索
  - 文本块重叠以保持上下文
  - 批处理向量化节省内存
  - 异常处理和自动恢复

- 🤖 **灵活的模型选择**
  - 支持本地模型（Qwen/ChatGLM 等）
  - 支持 API 模式（智谱/Moonshot）
  - 4-bit 量化降低显存占用
  - 自动模型加载和缓存

- 📊 **用户友好界面**
  - 进度反馈和状态提示
  - 错误提示和处理建议
  - 参考文档片段展示
  - 索引元数据管理

## 🛠️ 技术栈

- **前端框架**：
  - Streamlit：交互式界面

- **文本处理**：
  - NLTK：智能文本分块
  - PyPDF2：PDF 文件解析
  - Markdown：Markdown 解析

- **向量化和检索**：
  - Sentence-Transformers：文本向量化
  - FAISS：高性能向量检索
  - NumPy：数值计算

- **模型推理**：
  - Transformers：模型加载和推理
  - BitsAndBytes：模型量化
  - Torch：深度学习支持

## 🚀 快速开始

1. **环境准备**
```bash
# 创建环境
conda create -n nlp python=3.10
conda activate nlp

# 安装依赖
pip install -r requirements.txt

# 下载必要的 NLTK 数据
python -c "import nltk; nltk.download('punkt')"
```

2. **配置**
```python
# config.py 中设置运行模式
MODE = "local"  # 使用本地模型
# 或
MODE = "api"    # 使用API（需配置 API Key）
```

3. **启动应用**
```bash
conda activate nlp
streamlit run app.py
```

## 📝 使用说明

1. 打开应用后，上传 PDF 或 Markdown 文档
2. 等待索引构建完成（会显示进度）
3. 在输入框中输入问题
4. 查看 AI 回答和相关文档片段

## 🔧 进阶配置

### 本地模型模式
- 支持 4-bit 量化，8GB 显存即可运行
- 需要安装 `bitsandbytes` 和 `accelerate`
- 默认使用 Qwen-1.5-7B-Chat，可在配置中更改

### API 模式
- 支持智谱 AI 和 Moonshot (Kimi)
- 需在 config.py 或 .env 中配置 API Key
- 无需大显存，适合普通设备

## 📦 项目结构

```
├── app.py          # Streamlit 主程序
├── ingest.py       # 文档处理和索引构建
├── qa.py           # 检索和问答核心逻辑
├── config.py       # 配置文件
├── utils.py        # 工具函数
├── requirements.txt # 依赖清单
└── index/          # 索引文件目录
    ├── faiss.index    # FAISS 索引
    ├── chunks.txt     # 文本块
    └── metadata.json  # 索引元数据
```
