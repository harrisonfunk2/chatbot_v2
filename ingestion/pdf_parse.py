"""Extract page text and location metadata from PDFs in a nested data directory."""
from pathlib import Path
from hashlib import sha256
from pypdf import PdfReader


def list_pdf_files(path_to_data):
    """Accept either a PDF or a directory containing PDFs recursively."""
    root = Path(path_to_data).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Data path not found: {root}")
    if root.is_file():
        if root.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF: {root}")
        return root, [root]
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory: {root}")
    return root, sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")


def document_metadata(pdf_path, root):
    """Create location fields without assuming course names or directory depth."""
    pdf_path = Path(pdf_path)
    root = Path(root)
    relative = pdf_path.relative_to(root).as_posix() if root.is_dir() else pdf_path.name
    parts = Path(relative).parts[:-1]   # Exclude the filename from the folder list
    return {
        "document_id": sha256(relative.encode("utf-8")).hexdigest()[:16],
        "source_file": pdf_path.name,
        "relative_path": relative,
        "category": parts[0] if parts else None,    # Top level folder as category (might remove later)
        "folders": list(parts),
        "subject": parts[-1] if parts else None,    # Last folder as subject (might remove later)
    }


def extract_raw_text(path_to_data):
    """Return a flat list of pages with unique document identities and original text."""
    root, pdf_files = list_pdf_files(path_to_data)
    extracted_pages = []
    for pdf in pdf_files:
        metadata = document_metadata(pdf, root)
        reader = PdfReader(str(pdf))
        for number, page in enumerate(reader.pages, start=1):
            extracted_pages.append({
                **metadata,
                "page_number": number,
                "page_count": len(reader.pages),
                "raw_text": (page.extract_text() or "").strip(),
            })
    return extracted_pages


if __name__ == "__main__":
    for page in extract_raw_text("data"):
        print(page["relative_path"], "page", page["page_number"], "characters", len(page["raw_text"]))
