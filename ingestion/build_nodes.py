from llama_index.core.schema import (
    TextNode,
    NodeRelationship,
    RelatedNodeInfo,
)


def build_nodes(merged_documents):
    """Convert merged documents into LlamaIndex nodes."""

    nodes = []

    for document in merged_documents:
        doc_id = document["document_id"]

        # Information shared by every node from this PDF
        metadata = {
            "document_id": doc_id,
            "source_file": document["source_file"],
            "relative_path": document["relative_path"],
            "category": document["category"],
            "subject": document["subject"],
        }

        # Create the document summary node

        summary_id = f"{doc_id}:summary"

        summary_text = (
            f"Document: {document['document_title']}\n"
            f"Summary: {document['document_summary']}\n"
            f"Main topics: {', '.join(document['main_topics'])}"
        )

        summary_node = TextNode(
            id_=summary_id,
            text=summary_text,
            metadata={
                **metadata,
                "node_type": "document_summary",
            },
        )

        nodes.append(summary_node)

        # Create one node for each PDF page

        for page in document["pages"]:
            page_number = page["page_number"]

            # Convert structured code examples into text.
            code_text = "\n\n".join(
                f"Language: {example['language']}\n"
                f"Code:\n{example['code']}\n"
                f"Explanation: {example['explanation']}"
                for example in page["code_examples"]
            )

            # Combine relevant page content into searchable text.
            page_text = (
                f"Page title: {page['title']}\n"
                f"Main concept: {page['main_concept']}\n\n"
                f"Original PDF text:\n{page['raw_text']}\n\n"
                f"Screenshot text:\n{page['extracted_text']}\n\n"
                f"Code examples:\n{code_text}\n\n"
                f"Visual descriptions:\n{page['visual_description']}\n\n"
                f"Definitions and formulas:\n"
                f"{'; '.join(page['definitions_and_formulas'])}\n\n"
                f"Key takeaways:\n"
                f"{'; '.join(page['key_takeaways'])}"
            )

            page_node = TextNode(
                id_=f"{doc_id}:page:{page_number}",
                text=page_text,
                metadata={
                    **metadata,
                    "node_type": "page_content",
                    "page_number": page_number,
                    "parent_id": summary_id,
                },
                relationships={
                    NodeRelationship.PARENT: RelatedNodeInfo(
                        node_id=summary_id
                    )
                },
            )

            nodes.append(page_node)

    return nodes