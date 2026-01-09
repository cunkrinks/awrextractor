from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:1235/v1",  # sesuaikan dengan LM Studio
    api_key="lm-studio",
    model="meta-llama-3.1-8b-instruct",   # sesuaikan dengan nama model di LM Studio
    temperature=0.1,
)

resp = llm.invoke("Tuliskan 1 kalimat: ini test dari Irvan.")
print(resp)