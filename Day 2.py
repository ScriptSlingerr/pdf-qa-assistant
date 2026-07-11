import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model_name="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

prompt = ChatPromptTemplate.from_template("把以下中文翻译成英文:\n{chinese_text}")
chain = prompt | llm
result = chain.invoke({"chinese_text": "机器学习是人工智能领域的一个子领域"})
print(result.content)