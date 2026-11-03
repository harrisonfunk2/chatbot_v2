import json
from pathlib import Path

from pinecone import Pinecone

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    EMBEDDING_MODEL,
    MERGED_DATA_PATH,
)

"""
Step 1: Reconnect to your existing Pinecone index
Step 2: Create document-specific page retrievers
Step 3: Create summary nodes pointing to those retrievers
Step 4: Register everything with RecursiveRetriever
"""


def create_recursive_retriever(summary_top_k=2, page_top_k=3):
    """Create a retriever that follows document summaries to relevant pages"""

    # Load our previously processed documents
    with Path(MERGED_DATA_PATH).open("r", encoding="utf-8") as file:
        documents = json.load(file)

    if not documents:
        raise ValueError("No processed documents found.")

    #  Connect to our existing Pinecone vectors
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=PINECONE_NAMESPACE,
    )

    embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)

    page_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    # Create summary reference nodes and document retrievers
    summary_nodes = []
    retriever_dict = {}

    for document in documents:
        doc_id = document["document_id"]
        retriever_id = f"retriever:{doc_id}"

        summary_text = (
            f"Document: {document['document_title']}\n"
            f"Summary: {document['document_summary']}\n"
            f"Topics: {', '.join(document['main_topics'])}"
        )

        # An IndexNode points to another retriever via index_id
        summary_node = IndexNode(
            text=summary_text,
            index_id=retriever_id,
        )

        summary_node.metadata = {
            "document_id": doc_id,
            "source_file": document["source_file"],
        }

        summary_nodes.append(summary_node)

        # Only search page content nodes that belong to this specific document
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="parent_id",
                    value=f"{doc_id}:summary",
                ),
                MetadataFilter(
                    key="node_type",
                    value="page_content",
                ),
            ]
        )

        document_retriever = page_index.as_retriever(
            similarity_top_k=page_top_k,
            filters=filters,
        )

        retriever_dict[retriever_id] = document_retriever

    # Build an in memory vector index of summary references
    summary_index = VectorStoreIndex(
        summary_nodes,
        embed_model=embed_model,
    )

    summary_retriever = summary_index.as_retriever(
        similarity_top_k=summary_top_k,
    )

    retriever_dict["root"] = summary_retriever

    # Connect everything using LlamaIndex's recursive retriever
    recursive_retriever = RecursiveRetriever(
        root_id="root",
        retriever_dict=retriever_dict,
        verbose=False,
    )

    return recursive_retriever


if __name__ == "__main__":
    retriever = create_recursive_retriever()

    question = "What is ETL?"

    results = retriever.retrieve(question)

    for i, result in enumerate(results, start=1):
        metadata = result.node.metadata

        print(f"\n--- Result {i} ---")
        print("Document:", metadata.get("source_file"))
        print("Page:", metadata.get("page_number"))
        print("Score:", result.score)
        print("Text:", result.node.get_content()[:800])
