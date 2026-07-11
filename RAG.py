import os
from modelscope import snapshot_download

# 从 ModelScope 下载中文 Embedding 模型（国内可访问）
model_dir = snapshot_download("iic/nlp_gte_sentence-embedding_chinese-small")

import chromadb
from chromadb.utils import embedding_functions

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=model_dir
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.create_collection(
    name="test_docs",
    embedding_function=embedding_fn
)

docs = [
    "LangChain 是一个 LLM 应用开发框架",
    "RAG 是检索增强生成的缩写",
    "Embedding 将文本转化为向量",
    "Chroma 是一个轻量级向量数据库",
    "机器人操作系统 ROS2 用于机械臂控制",
]
collection.add(
    documents=docs,
    ids=[f"doc_{i}" for i in range(len(docs))]
)

results = collection.query(query_texts=["什么是 RAG？"], n_results=2)
print(results["documents"])
