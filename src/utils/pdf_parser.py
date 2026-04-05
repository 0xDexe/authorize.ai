"""
AuthorizeAI — PDF / Text Parser
=================================
Extracts text from clinical notes and policy documents.
Supports PDF (via PyMuPDF), plain text, and FHIR CDA stubs.
"""

from pathlib import Path


def extract_text(file_path: str | Path) -> str:
    """
    Extract text from a file. Auto-detects format from extension.
    Supported: .pdf, .txt, .text, .json, .xml
    """
    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        return _extract_pdf(path)
    elif path.suffix.lower() in (".txt", ".text", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    elif path.suffix.lower() == ".json":
        import json
        data = json.loads(path.read_text())
        # Handle FHIR-style bundles
        if isinstance(data, dict) and "text" in data:
            return data["text"]
        return json.dumps(data, indent=2)
    else:
        return path.read_text(encoding="utf-8", errors="replace")


def _extract_pdf(path: Path) -> str:
    """Extract text from PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF parsing: pip install PyMuPDF"
        )

    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def extract_text_from_string(raw: str, source_hint: str = "") -> str:
    """
    Clean and normalize raw clinical text input.
    Handles common EHR export artifacts.
    """
    import re

    # Normalize line endings
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse excessive whitespace but preserve paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
