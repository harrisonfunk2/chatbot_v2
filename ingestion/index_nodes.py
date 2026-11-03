import json
from pathlib import Path

from pinecone import Pinecone

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.schema import TextNode


from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_SUMMARY_NAMESPACE,
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




def build_summary_nodes(documents):
    summary_nodes = []

    for document in documents:
        doc_id = document["document_id"]

        summary_text = (
            f"Document: {document['document_title']}\n"
            f"Summary: {document['document_summary']}\n"
            f"Topics: {', '.join(document['main_topics'])}"
        )

        node = TextNode(
            id_=f"{doc_id}:summary",
            text=summary_text,
            metadata={
                "source_document_id": doc_id,
                "source_file": document["source_file"],
                "node_type": "document_summary",
            },
        )

        summary_nodes.append(node)

    return summary_nodes


def index_summaries():
    # Load the processed documents
    with Path(MERGED_DATA_PATH).open(
        "r", encoding="utf-8"
    ) as file:
        documents = json.load(file)

    # Build one summary node per document
    summary_nodes = build_summary_nodes(documents)

    if not summary_nodes:
        raise ValueError("No summary nodes were created.")

    # Connect to the existing Pinecone index
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    # Use our new summary namespace
    summary_vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=PINECONE_SUMMARY_NAMESPACE,
    )

    summary_storage = StorageContext.from_defaults(
        vector_store=summary_vector_store
    )

    # Generate summary embeddings
    embed_model = OpenAIEmbedding(
        model=EMBEDDING_MODEL
    )

    # Upload summary nodes to Pinecone
    VectorStoreIndex(
        nodes=summary_nodes,
        storage_context=summary_storage,
        embed_model=embed_model,
    )

    print(f"Indexed {len(summary_nodes)} document summaries.")
    print(pinecone_index.describe_index_stats())


def main():
    index_documents()
    index_summaries()


if __name__ == "__main__":
    main()