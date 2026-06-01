from langchain_openai import ChatOpenAI
import time

llm = ChatOpenAI(
    base_url="http://192.168.10.59:11434/v1",
    api_key="Empty",
    model="gemma4:e4b",
    temperature=0
)
start = time.time()
response = llm.invoke("你好，请介绍一下你自己")
end = time.time()

print(response.content)
print(f"\n耗时: {end - start:.2f} 秒")