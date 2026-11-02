from pinecone import Pinecone, ServerlessSpec

from config import PINECONE_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "rag-chatbot-v2"

if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        vector_type="dense",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print("Index created successfully.")
else:
    print("Index already exists.")