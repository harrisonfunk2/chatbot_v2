import json
from pathlib import Path
from retrieval.recursive_retrieval import create_recursive_retriever
from retrieval.test_retrieval import create_direct_retriever

from config import RETRIEVAL_QUESTIONS_PATH

import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("rag-retrieval-evaluation")

def extract_retrieved_pages(results):
    retrieved_pages = []

    for result in results:
        metadata = result.node.metadata

        if metadata.get("node_type") != "page_content":
            continue

        relative_path = metadata.get("relative_path")
        page_number = metadata.get("page_number")

        if relative_path is not None and page_number is not None:
            retrieved_pages.append((relative_path, page_number,))

    return retrieved_pages


def calculate_recall_at_k(retrieved_pages, relevant_pages, k):
    if not relevant_pages:
        return None
    
    # Get the top k retrieved pages
    top_k_retrieved = set(retrieved_pages[:k])

    relevant_set = set(relevant_pages)

    # Count how many of the relevant pages are in the top k retrieved
    relevant_found = sum(
        1 for page in top_k_retrieved if page in relevant_set
    )

    # Calculate recall at k
    recall_at_k = relevant_found / len(relevant_set)

    return recall_at_k


def log_retrieval_run(results, method, k_values):

    with mlflow.start_run(run_name=f"{method}-test_1"):

        mlflow.log_params({
            "retrieval_method": method,
            "embedding_model": "text-embedding-3-small",
            "max_k": max(k_values),
            "num_questions": len(set(
                row["question_id"] for row in results
            )),
        })

        for k in k_values:
            scores = []
            for row in results:
                if row["k"] == k:
                    scores.append(row[f"{method}_recall"])

            if scores:
                avg_recall = sum(scores) / len(scores)

                mlflow.log_metric(f"recall_at_{k}", avg_recall)
           

        mlflow.log_dict(results, f"{method}_evaluation_results.json")


def main():
    with Path(RETRIEVAL_QUESTIONS_PATH).open("r", encoding="utf-8") as file:
            questions = json.load(file)

    k_values = [1, 3, 5, 9]
    max_k = max(k_values)

    direct_retriever = create_direct_retriever(top_k=max_k)
    recursive_retriever = create_recursive_retriever(summary_top_k=2, page_top_k=max_k)

    evaluation_results = []

    for question_data in questions:

        if not question_data["relevant_pages"]:
            continue

        question = question_data["question"]
        question_id = question_data["id"]

        relevant_pages = [
            (page["relative_path"], page["page_number"])
            for page in question_data["relevant_pages"]]

        direct_results = direct_retriever.retrieve(question)
        recursive_results = recursive_retriever.retrieve(question)

        direct_pages = extract_retrieved_pages(direct_results)
        recursive_pages = extract_retrieved_pages(recursive_results)

        for k in k_values:
            direct_recall = calculate_recall_at_k(
                direct_pages, relevant_pages, k)

            recursive_recall = calculate_recall_at_k(
                recursive_pages, relevant_pages, k)

            if direct_recall is None or recursive_recall is None:
                continue

            evaluation_results.append({
                "question_id": question_id,
                "question": question,
                "k": k,
                "direct_recall": direct_recall,
                "recursive_recall": recursive_recall,
                "expected_pages": relevant_pages,
                "direct_retrieved_pages": direct_pages,
                "recursive_retrieved_pages": recursive_pages,
            })

    log_retrieval_run(
        evaluation_results,
        "direct",
        k_values)

    log_retrieval_run(
        evaluation_results,
        "recursive",
        k_values)


if __name__ == "__main__":
    main()