import json
from pathlib import Path

from ingestion.pdf_parse import extract_raw_text
from ingestion.vision import generate_vision_descriptions


def merge_document_content(raw_pages, vision_docs):
    """
    Combine raw PDF text and structured vision results
    into document-level records containing merged pages.
    """

    # Group raw pages by document path and page number
    raw_lookup = {}

    for page in raw_pages:
        key = (page["relative_path"], page["page_number"])
        if key in raw_lookup:
            raise ValueError(f"Duplicate raw page: {key}")

        raw_lookup[key] = page["raw_text"]

    merged_documents = []
    matched_pages = set()

    for document in vision_docs:
        source_path = document["relative_path"]
        merged_pages = []

        for page in document["pages"]:
            page_number = page["page_number"]
            key = (source_path, page_number)

            if key not in raw_lookup:
                raise ValueError(
                    f"No raw text record found for {key}"
                )

            matched_pages.add(key)

            merged_page = {
                **page,
                "raw_text": raw_lookup[key],
            }

            merged_pages.append(merged_page)

        # Keep document metadata and summary
        merged_documents.append({
            **document,
            "pages": merged_pages,
        })

    # Make sure no raw pages were silently skipped
    missing = set(raw_lookup) - matched_pages

    if missing:
        raise ValueError(
            f"Raw pages missing vision records: {sorted(missing)}"
        )

    return merged_documents


def main():
    path_to_data = "data"

    raw_pages = extract_raw_text(path_to_data)

    vision_docs = generate_vision_descriptions(path_to_data)

    merged_documents = merge_document_content(
        raw_pages,
        vision_docs
    )

    output_path = Path("data/processed/merged_documents.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(merged_documents, file, indent=2)

    print(f"Saved {len(merged_documents)} merged documents.")


if __name__ == "__main__":
    main()
