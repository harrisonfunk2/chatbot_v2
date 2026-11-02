
import json
from pathlib import Path

import mlflow
import mlflow.genai

from mlflow.genai.scorers import (
    RetrievalGroundedness,
    RelevanceToQuery,
)

from retrieval.recursive_retrieval import create_recursive_retriever
from generation.generation import answer_question


def load_evaluation_data():
    with Path("evaluation/answer_questions.json").open(
        "r", encoding="utf-8"
    ) as file:
        questions = json.load(file)

    evaluation_data = []

    for question_data in questions:
        evaluation_data.append({
            "inputs": {
                "question": question_data["question"]
            },
            "expectations": {
                "expected_response": question_data["expected_answer"]
            }
        })

    return evaluation_data


retriever = create_recursive_retriever(
    summary_top_k=2,
    page_top_k=3
)


def predict_fn(question):
    output = answer_question(question, retriever)
    return output["answer"]


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("rag-answer-evaluation")

    evaluation_data = load_evaluation_data()

    results = mlflow.genai.evaluate(
        data=evaluation_data,
        predict_fn=predict_fn,
        scorers=[
            RetrievalGroundedness(
                model="openai:/gpt-4.1-mini"
            ),
            RelevanceToQuery(
                model="openai:/gpt-4.1-mini"
            )
        ]
    )

    print("Evaluation metrics:")
    print(results.metrics)


if __name__ == "__main__":
    main()
