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

# Chain 1：生成大纲
outline_prompt = ChatPromptTemplate.from_template(
    "为关于'{topic}'的博客文章生成一个3点大纲，用中文回复"
)
outline_chain = outline_prompt | llm

# Chain 2：根据大纲写正文
write_prompt = ChatPromptTemplate.from_template(
    "根据以下大纲，写一篇300字的博客文章：\n{outline}"
)
write_chain = write_prompt | llm

# 串联：先跑 Chain 1，结果手动传给 Chain 2
def generate_blog(topic):
    outline = outline_chain.invoke({"topic": topic})
    article = write_chain.invoke({"outline": outline.content})
    return article.content

result = generate_blog("RAG检索增强生成")
print(result)
