# 📚 PDF 智能问答系统

![demo](assert/demo.gif)

## 一句话介绍
上传 PDF → 用自然语言提问 → AI 基于原文回答。基于 RAG（检索增强生成）技术。

## 功能
- 📤 上传任意 PDF 文档，自动解析并构建知识库
- 💬 自然语言提问，AI 精准定位原文回答
- 🚫 找不到相关信息时明确告知，不编造（防幻觉）
- 🔄 支持多轮对话，上下文连续

## 技术栈
| 层级 | 技术 |
|------|------|
| LLM | DeepSeek Chat API |
| 应用框架 | LangChain（LCEL） |
| 向量数据库 | Chroma |
| Embedding | GTE Sentence Embedding (中文) |
| 前端 | Streamlit |
| 文档解析 | PyPDF + RecursiveCharacterTextSplitter |

## 快速启动

```bash
# 1. 克隆
git clone https://github.com/ScriptSlingerr/pdf-qa-assistant.git
cd pdf-qa-assistant

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY

# 4. 运行
streamlit run app.py
```

## RAG 流程图

```
[用户提问] → [Embedding] → [Chroma 向量库检索 Top 3]
                                     ↓
                              [拼接 Prompt]
                                     ↓
                                 [DeepSeek]
                                     ↓
                                 [回答]
```

## 项目结构

```
pdf-qa-assistant/
├── app.py              # Streamlit 主程序
├── requirements.txt    # 依赖列表
├── .env.example        # 环境变量模板
├── .gitignore
├── README.md
└── assert/             # 演示素材
```

## 踩坑记录

- DeepSeek 无 Embedding API → 换用 ModelScope GTE 中文模型本地运行
- HuggingFace 被墙 → 设置镜像或 ModelScope 下载
- Streamlit file_uploader 返回对象非路径 → 先写临时文件再传 PyPDFLoader

## License

MIT
