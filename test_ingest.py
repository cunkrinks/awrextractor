from awr_rag import create_embeddings, create_redis_vectorstore
from langchain_core.documents import Document

emb = create_embeddings()
vs = create_redis_vectorstore("redis://localhost:6379", "test_index", emb)

docs = [
    Document(page_content="doc satu", metadata={"id": 1}),
    Document(page_content="doc dua", metadata={"id": 2}),
    Document(page_content="doc tiga", metadata={"id": 3}),
]

print("Adding 3 docs...")
vs.add_documents(docs)
print("Done.")

