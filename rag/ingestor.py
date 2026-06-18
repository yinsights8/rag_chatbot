# Ingestor.py 

########### data ingest file ##############
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF


### 1. document loader

def parse_and_chunk(pdf_path: str, chunk_size: int = 700, chunk_overlap: int = 150) -> list[dict]:
    """Parse PDF and return list of chunk dicts with text + metadata."""
    doc = fitz.open(pdf_path)
    full_text_pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            full_text_pages.append({"text": text, "page": page_num})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = []
    for page_data in full_text_pages:
        splits = splitter.split_text(page_data["text"])
        for split in splits:
            chunks.append({
                "text": split,
                "page": page_data["page"],
                "chunk_index": len(chunks),
            })
    return chunks