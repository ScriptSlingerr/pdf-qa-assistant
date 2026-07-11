from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from chromadb.utils import embedding_functions
from modelscope import snapshot_download
import os, chromadb
from dotenv import load_dotenv

load_dotenv()

# ---- Chroma 初始化 ----
model_dir = snapshot_download("iic/nlp_gte_sentence-embedding_chinese-small")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=model_dir
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="test_docs",
    embedding_function=embedding_fn
)

# 1. 检索相关文档
results = collection.query(query_texts=["什么是RAG？"], n_results=2)
retrieved_text = "\n".join(results["documents"][0])
print("检索结果:", results["documents"])

# 2. LCEL Chain：拼 prompt + 调 LLM
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

rag_prompt = ChatPromptTemplate.from_template(
    "根据以下参考资料回答问题：\n"
    "参考资料：\n{context}\n\n"
    "问题：{question}\n"
    "回答："
)

rag_chain = rag_prompt | llm
response = rag_chain.invoke({
    "context": retrieved_text,
    "question": "什么是RAG？"
})
print(response.content)
