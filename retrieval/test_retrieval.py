from pinecone import Pinecone

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    EMBEDDING_MODEL,
)


def create_direct_retriever(top_k=3):
    """Retrieve relevant nodes from Pinecone without generating an answer"""

    # Connect to our existing Pinecone index
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    # Connect LlamaIndex to the same namespace used during indexing
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=PINECONE_NAMESPACE,
    )

    # Use the same embedding model we used during indexing
    embed_model = OpenAIEmbedding(
        model=EMBEDDING_MODEL
    )

    # Reconnect to the existing vectors
    # This does not re embed or reupload our documents
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    # Create a retriever that returns the top_k matches
    retriever = index.as_retriever(
        similarity_top_k=top_k
    )

    return retriever

"""
if __name__ == "__main__":
    question = "What is OOP?"

    create_direct_retriever(
        top_k=3,
    )
"""