import chromadb
from chromadb.utils import embedding_functions
from modelscope import snapshot_download
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

# 加载 PDF
loader = PyPDFLoader(r"D:\BHBF\小五轮3号操作说明.pdf")
pages = loader.load()
print(f"共 {len(pages)} 页")

# 切分文本：每 500 字一块，块与块之间重叠 50 字
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " "]  # 优先在段落/句子边界切断
)
chunks = text_splitter.split_documents(pages)
print(f"切成了 {len(chunks)} 个文本块")
# 把切好的文本块逐条插入
for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk.page_content],
        ids=[f"chunk_{i}"]
    )

print(f"已存入 {collection.count()} 条文本块")