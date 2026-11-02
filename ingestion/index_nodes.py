import json
from pathlib import Path

from pinecone import Pinecone

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    EMBEDDING_MODEL,
    MERGED_DATA_PATH,
)

from ingestion.build_nodes import build_nodes


def index_documents():

    # Load the merged documents
    path = Path(MERGED_DATA_PATH)

    if not path.exists():
        raise FileNotFoundError(
            f"Merged data not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        merged_documents = json.load(file)

    # Convert merged documents into LlamaIndex nodes
    nodes = build_nodes(merged_documents)

    if not nodes:
        raise ValueError("No nodes were created.")

    print(f"Created {len(nodes)} nodes.")

    # Connect to Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    # Configure Pinecone as the vector store
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=PINECONE_NAMESPACE,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # Configure OpenAI embeddings
    embed_model = OpenAIEmbedding(
        model=EMBEDDING_MODEL
    )

    # Generate embeddings and store vectors
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    print("Indexing completed.")

    return index


if __name__ == "__main__":
    index_documents()