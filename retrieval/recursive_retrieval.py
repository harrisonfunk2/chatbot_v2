from pinecone import Pinecone

from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_SUMMARY_NAMESPACE,
    EMBEDDING_MODEL,
)

"""
Step 1: Reconnect to existing Pinecone index
Step 2: Create document specific page retrievers
Step 3: Create summary nodes pointing to those retrievers
Step 4: Register everything with RecursiveRetriever
"""


class TwoStageRetriever:

    def __init__(self, summary_index, page_index,
                 summary_top_k=2, page_top_k=3):

        self.summary_retriever = summary_index.as_retriever(
            similarity_top_k=summary_top_k
        )

        self.page_index = page_index
        self.page_top_k = page_top_k

    def retrieve(self, question):
        # First: retrieve relevant document summaries
        summary_results = self.summary_retriever.retrieve(question)

        page_results = []
        seen_nodes = set()

        # Second: retrieve pages belonging to those documents
        for summary in summary_results:
            metadata = summary.node.metadata
            doc_id = metadata.get("source_document_id")

            if not doc_id:
                continue

            filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="parent_id",
                        value=f"{doc_id}:summary"
                    ),
                    MetadataFilter(
                        key="node_type",
                        value="page_content"
                    )
                ]
            )

            page_retriever = self.page_index.as_retriever(
                similarity_top_k=self.page_top_k,
                filters=filters
            )

            results = page_retriever.retrieve(question)

            for result in results:
                node_id = result.node.node_id

                if node_id not in seen_nodes:
                    page_results.append(result)
                    seen_nodes.add(node_id)

        return page_results


def create_recursive_retriever(summary_top_k=2, page_top_k=3):

    # Connect to Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    # Connect to the existing pages namespace
    page_vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=PINECONE_NAMESPACE
    )

    # Connect to the new summaries namespace
    summary_vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=PINECONE_SUMMARY_NAMESPACE
    )

    embed_model = OpenAIEmbedding(
        model=EMBEDDING_MODEL
    )

    # Reconnect to the stored vectors
    page_index = VectorStoreIndex.from_vector_store(
        vector_store=page_vector_store,
        embed_model=embed_model
    )

    summary_index = VectorStoreIndex.from_vector_store(
        vector_store=summary_vector_store,
        embed_model=embed_model
    )

    return TwoStageRetriever(
        summary_index=summary_index,
        page_index=page_index,
        summary_top_k=summary_top_k,
        page_top_k=page_top_k
    )


if __name__ == "__main__":

    retriever = create_recursive_retriever(
        summary_top_k=2,
        page_top_k=3
    )

    results = retriever.retrieve("What is ETL?")

    for result in results:
        print("Document:", result.node.metadata.get("source_file"))
        print("Page:", result.node.metadata.get("page_number"))
        print("Text:", result.node.get_content()[:300])
        print()