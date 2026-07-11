from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

# 看一下第一个块的内容
print(chunks[5].page_content[:200])