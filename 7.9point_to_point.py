from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from modelscope import snapshot_download
load_dotenv()

# 注意：这里假设你已经在之前创建了 collection（比如名为 pdf_knowledge 的 Chroma 集合）
# 并且 collection 已经绑定了正确的 embedding 函数（与插入时一致）
model_dir = snapshot_download("iic/nlp_gte_sentence-embedding_chinese-small")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=model_dir
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="test_docs",
    embedding_function=embedding_fn
)


llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",   
)


prompt = ChatPromptTemplate.from_template(
    "你是一个知识库助手。根据以下参考资料回答问题。"
    "如果参考资料中没有相关信息，就说'未找到相关信息'，不要编造。\n\n"
    "参考资料：\n{context}\n\n"
    "问题：{question}\n"
    "回答："
)


chain = prompt | llm

def ask_pdf(question):
    results = collection.query(query_texts=[question], n_results=3)
    context = "\n\n".join(results["documents"][0])
    response = chain.invoke({"context": context, "question": question})
    return response.content

# 测试
if __name__ == "__main__":
    print(ask_pdf("这篇PDF主要讲了什么？"))