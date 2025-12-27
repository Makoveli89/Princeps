import os
import re
import hashlib
from typing import List, Dict, Any, Iterable, Tuple
from pypdf import PdfReader

# Optional token-aware chunking
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def count_tokens(s: str) -> int: return len(_ENC.encode(s))
    def split_tokens(text: str, max_tokens: int, overlap: int) -> List[str]:
        ids = _ENC.encode(text)
        chunks = []
        start = 0
        while start < len(ids):
            end = min(start + max_tokens, len(ids))
            chunk_ids = ids[start:end]
            chunks.append(_ENC.decode(chunk_ids))
            start = max(0, end - overlap)
        return chunks
except Exception:
    _ENC = None
    def count_tokens(s: str) -> int: return max(1, len(s) // 4)
    def split_tokens(text: str, max_tokens: int, overlap: int) -> List[str]:
        size = max_tokens * 4  # ~4 chars/token roughness
        ov   = overlap * 4
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            chunks.append(text[start:end])
            start = max(0, end - ov)
        return chunks

def _clean(s: str) -> str:
    s = s.replace("\x00", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def extract_pdf_text(path: str) -> Tuple[str, int]:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        texts.append(txt)
    all_text = _clean("\n\n".join(texts))
    return all_text, len(reader.pages)

def chunk_text(text: str, chunk_tokens: int = 1200, overlap_tokens: int = 150) -> List[str]:
    if not text:
        return []
    return split_tokens(text, max_tokens=chunk_tokens, overlap=overlap_tokens)

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def iter_pdf_files(root: str, pattern_suffix: str = ".pdf") -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(pattern_suffix):
                yield os.path.join(dirpath, fn)

def make_learning_payload(pdf_path: str, chunk_text_str: str, chunk_idx: int, total_pages: int) -> Dict[str, Any]:
    content_hash = sha256(pdf_path + f"::chunk{chunk_idx}::" + sha256(chunk_text_str))
    task = f"PDF Learning from {os.path.basename(pdf_path)} [chunk {chunk_idx}]"
    solution = {
        "id": content_hash,
        "source": "pdf",
        "file_path": pdf_path,
        "chunk_index": chunk_idx,
        "total_pages": total_pages,
        "text": chunk_text_str,
    }
    feedback = "Extracted from PDF; ready for knowledge routing and downstream pattern analysis."
    metadata = {
        "source": "pdf",
        "filename": os.path.basename(pdf_path),
        "pages": total_pages,
        "content_hash": content_hash,
        "approx_tokens": count_tokens(chunk_text_str),
    }
    # heuristic "quality" for ingestion — you can post-hoc rescore with your validator
    score = 80.0 if len(chunk_text_str) > 200 else 50.0
    return dict(task=task, solution=solution, score=score, feedback=feedback, success=True, metadata=metadata)

def ingest_pdf_dir(
    directory: str,
    umi_client,
    chunk_tokens: int = 1200,
    overlap_tokens: int = 150,
    max_files: int = 0,
) -> Dict[str, Any]:
    """
    Ingests all PDFs under `directory`, chunks, and stores each chunk via UMI.
    """
    processed = 0
    stored = 0
    results = []
    for path in iter_pdf_files(directory):
        processed += 1
        if max_files and processed > max_files:
            break
        text, pages = extract_pdf_text(path)
        if not text:
            continue
        chunks = chunk_text(text, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
        for i, ch in enumerate(chunks):
            payload = make_learning_payload(path, ch, i, pages)
            res = umi_client.store_learning(**payload)
            stored += 1
            results.append(res)
    return {"processed_files": processed, "stored_chunks": stored, "results": results}
