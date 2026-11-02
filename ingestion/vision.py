"""Vision-enriched PDF page descriptions and document-level summaries."""
import random
import time
from pydantic import BaseModel
from pypdf import PdfReader
from openai import OpenAI, RateLimitError

from ingestion.pdf_parse import list_pdf_files, document_metadata


class CodeExample(BaseModel):
    language: str
    code: str
    explanation: str


class PageDescription(BaseModel):
    page_number: int
    title: str
    main_concept: str

    extracted_text: str

    code_examples: list[CodeExample]

    definitions_and_formulas: list[str]
    visual_description: str
    key_takeaways: list[str]
    ambiguities: list[str]


class FileDescription(BaseModel):
    document_title: str
    document_summary: str
    main_topics: list[str]
    pages: list[PageDescription]


# Sends a PDF to the OpenAI API for structured page and document analysis, retrying if rate limits are encountered
def parse_with_retry(client, model, file_id, page_count, attempts=6):
    instructions = f"""
Analyze this PDF as study and reference material.

The document contains {page_count} pages.

Return one structured record for every page.

For each page:

1. Extract readable text from screenshots,
   including explanations and notes.

2. Identify code examples in the document.

3. For each code example:
   - Identify the programming language.
   - Transcribe the code as accurately as possible.
   - Preserve formatting, indentation, and syntax.
   - Explain what the code does.
   - Do not correct or invent code.

4. Extract important definitions and formulas.

5. Describe meaningful visuals such as diagrams,
   plots, tables, and charts.

6. Identify the main concepts and takeaways.

7. Record anything unclear or unreadable.

For the entire document:

- Provide a descriptive title.
- Summarize the main subjects covered.
- List the main topics.

Only use information supported by the PDF.
"""
    for attempt in range(attempts):
        try:
            return client.responses.parse(
                model=model,
                input=[{"role": "user", "content": [
                    {"type": "input_file", "file_id": file_id},
                    {"type": "input_text", "text": instructions},
                ]}],
                text_format=FileDescription,
            )
        except RateLimitError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(60, 5 * (2 ** attempt)) + random.uniform(0, 1))
    raise RuntimeError("Vision request did not complete")


# Finds all PDFs, uploads each to OpenAI, calls parse_with_retry(), and organizes the results with document metadata
def generate_vision_descriptions(path_to_data, model="gpt-4o-mini", client=None):
    """Analyze all PDFs under a directory; return document summaries and page descriptions.

    This first version sends one PDF per request. For very long PDFs, implement page
    batching and hierarchical summarization to avoid output truncation/context limits.
    """
    if client is None:
        client = OpenAI()
    root, pdf_files = list_pdf_files(path_to_data)
    results = []

    for pdf in pdf_files:
        metadata = document_metadata(pdf, root)
        page_count = len(PdfReader(str(pdf)).pages)
        print(f"Analyzing {metadata['relative_path']} ({page_count} pages)")
        with pdf.open("rb") as stream:
            uploaded = client.files.create(file=stream, purpose="user_data")

        try:
            response = parse_with_retry(client, model, uploaded.id, page_count)
            description = response.output_parsed
            if description is None:
                raise RuntimeError(f"No structured response for {pdf}")
            pages = description.pages
            numbers = [p.page_number for p in pages]
            if len(pages) != page_count or sorted(numbers) != list(range(1, page_count + 1)):
                raise ValueError(f"Page mismatch in {pdf}: expected 1..{page_count}, got {numbers}")

            results.append({
                **metadata,
                "page_count": page_count,
                "document_title": description.document_title,
                "document_summary": description.document_summary,
                "main_topics": description.main_topics,
                "pages": [page.model_dump() for page in sorted(pages, key=lambda x: x.page_number)],
            })
        finally:
            # Best effort: uploaded source PDFs need not remain in the API's file storage.
            try:
                client.files.delete(uploaded.id)
            except Exception as error:
                print(f"Warning: could not delete temporary upload {uploaded.id}: {error}")
    return results


if __name__ == "__main__":
    import json
    from pathlib import Path
    result = generate_vision_descriptions("data")
    Path("vision_output.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
