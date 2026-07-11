import hashlib
import re

import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import chromadb
from chromadb.utils import embedding_functions
from modelscope import snapshot_download

load_dotenv()  # 读取 DEEPSEEK_API_KEY

st.set_page_config(page_title="PDF 智能问答", page_icon="📚")
st.title("📚 PDF 智能问答系统")
st.caption("上传 PDF 文档，用自然语言提问，AI 基于原文回答")

# ---------- 缓存：Embedding 模型只加载一次 ----------
@st.cache_resource
def init_embedding():
    model_dir = snapshot_download("iic/nlp_gte_sentence-embedding_chinese-small")
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_dir)

embedding_fn = init_embedding()

# ---------- 缓存：Chroma 客户端 ----------
@st.cache_resource
def init_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")

chroma_client = init_chroma_client()

# ---------- 处理上传的 PDF（去重 + 清旧数据） ----------
def process_pdf(file_bytes, file_name):
    # 用文件名作为集合名，避免不同文件冲突
    safe_name = re.sub(r'[^a-zA-Z0-9_一-鿿]','_',file_name)
    collection_name = f"pdf_{safe_name}"

    # 如果集合已存在，先删除（实现“覆盖上传”）
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass  # 不存在则忽略

    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )

    # 保存上传文件到临时文件（PyPDFLoader 需要路径）
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "，", " "]
        )
        chunks = text_splitter.split_documents(pages)
        doc_texts = [c.page_content for c in chunks]
        ids = [f"chunk_{i}" for i in range(len(doc_texts))]

        # 批量插入
        collection.add(documents=doc_texts, ids=ids)
        return collection, len(doc_texts)
    finally:
        os.unlink(tmp_path)  # 清理临时文件

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
# ---------- 问答函数 ----------
def ask_pdf(question, collection):
    results = collection.query(query_texts=[question], n_results=3)
    context = "\n\n".join(results["documents"][0])
    if not context.strip():
        return "未找到相关信息。"

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content

# ---------- 界面逻辑 ----------
uploaded_file = st.file_uploader("上传 PDF 文件", type=["pdf"])

# 当上传了新文件（或换了新文件），处理并保存集合到 session_state
if uploaded_file is not None:
    # 用文件名+文件大小做简单标识，避免重复处理同一文件
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("file_key") != file_key:
        with st.spinner("正在处理 PDF，请稍候..."):
            collection, chunk_count = process_pdf(
                uploaded_file.getvalue(),  # 注意：只能用一次，所以提前读取
                uploaded_file.name
            )
            st.session_state.collection = collection
            st.session_state.file_key = file_key
            st.session_state.chunk_count = chunk_count
        st.success(f"✅ 已处理 {chunk_count} 个文本块")
    else:
        collection = st.session_state.collection
else:
    # 没有上传时，保留已有数据不清空
    collection = st.session_state.get("collection")

# ---------- 显示历史消息 ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------- 用户输入 ----------
if question := st.chat_input("请输入你的问题"):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # 检查是否已上传 PDF
    if st.session_state.collection is None:
        with st.chat_message("assistant"):
            st.write("⚠️ 请先上传一份 PDF 文档。")
        st.session_state.messages.append({"role": "assistant", "content": "请先上传一份 PDF 文档。"})
    else:
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                answer = ask_pdf(question, st.session_state.collection)
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})