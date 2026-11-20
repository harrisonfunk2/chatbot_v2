"""
I updated the test_retrieval.py file to make it similar to the 
recursive_retrieval.py file by making the main functions similar.
This file needs to be updated before use
THIS FILE WONT WORK UNTIL UPDATED
"""


# Existing direct retrieval
from retrieval.test_retrieval import create_direct_retriever

# Existing recursive retrieval
from retrieval.recursive_retrieval import create_recursive_retriever


def print_results(results, method):

    for i, result in enumerate(results, start=1):
        metadata = result.node.metadata

        print(f"\n--- {method} Result {i} ---")
        print("Similarity score:", result.score)
        print("Node ID:", result.node.node_id)
        print("Node type:", metadata.get("node_type"))
        print("Document:", metadata.get("source_file"))
        print("Page:", metadata.get("page_number", "Document summary"))
        print("Text:", result.node.get_content()[:800])



def main():
    question = "What is OOP?"

    # Retrieve using original method
    direct_results = create_direct_retriever(top_k=3)


    # Retrieve using recursive method
    recursive_retriever = create_recursive_retriever(summary_top_k=2, page_top_k=3)
    recursive_results = recursive_retriever.retrieve(question)

    # Print results
    print_results(direct_results, method="Direct")
    print_results(recursive_results, method="Recursive")


    '''print("Direct results:", len(direct_results))
    print("Recursive results:", len(recursive_results))'''


if __name__ == "__main__":
    main()
