from awr_rag import create_embeddings
emb = create_embeddings()
print(emb.embed_query("hello world"))