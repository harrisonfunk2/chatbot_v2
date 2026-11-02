from retrieval.recursive_retrieval import create_recursive_retriever
from generation.citation import verify_citations
from openai import OpenAI

import mlflow
from mlflow.openai import autolog
from mlflow.entities import Document

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("rag-chatbot-generation")

autolog()

client = OpenAI()


def format_context(results):
    context_parts = []
    citation_map = {}

    for i, result in enumerate(results, start=1):
        metadata = result.node.metadata

        source = metadata.get("source_file")
        page = metadata.get("page_number")
        text = result.node.get_content()

        context_parts.append(
            f"[{i}] Source: {source}, Page: {page}\n"
            f"Content:\n{text}"
        )

        citation_map[i] = f"Source: {source}, Page: {page}"

    return "\n\n".join(context_parts), citation_map


@mlflow.trace
def generate_answer(question, context):

    instructions = """
    You are a study assistant answering questions about
    the user's uploaded notes.

    Rules:
    - Base factual claims on the supplied source material.
    - Explain concepts clearly and accurately.
    - Preserve code examples from the notes.
    - Cite supporting sources using [1], [2], etc.
    - Do not invent citations or missing code.
    - If the sources do not contain enough information,
      clearly state that limitation.
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=instructions,
        input=(
            f"Question:\n{question}\n\n"
            f"Retrieved sources:\n{context}"
        ),
    )

    return response.output_text



@mlflow.trace(name="retrieve_context", span_type="RETRIEVER")
def retrieve_context(question, retriever):

    results = retriever.retrieve(question)

    documents = [
        Document(
            page_content=result.node.get_content(),
            metadata={
                "doc_uri": result.node.metadata.get("relative_path", ""),
                "page_number": result.node.metadata.get("page_number")
            },
            id=result.node.node_id
        ).to_dict()
        for result in results
    ]

    span = mlflow.get_current_active_span()
    if span is not None:
        span.set_outputs(documents)

    return results



@mlflow.trace(name="rag_pipeline")
def answer_question(question, retriever):
    results = retrieve_context(question, retriever)

    if not results:
        return {
            "answer": "No relevant sources found",
            "retrieved_contexts": [],
            "citation_map": {}
        }

    context, citation_map = format_context(results)

    answer = generate_answer(question, context)

    valid, invalid = verify_citations(answer, citation_map)

    print("Valid citations:", valid)
    print("Invalid citations:", invalid)

    return {
        "answer": answer,
        "retrieved_contexts": [
            result.node.get_content()
            for result in results
        ],
        "citation_map": citation_map,
    }


def main():
    question = "What is ETL?"

    retriever = create_recursive_retriever(
        summary_top_k=2,
        page_top_k=3
    )

    answer = answer_question(question, retriever)

    print("\nAnswer:\n")
    print(answer["answer"])


if __name__ == "__main__":
    main()