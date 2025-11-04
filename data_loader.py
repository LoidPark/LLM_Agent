# data_loader.py
import os, glob
from typing import List, Dict
from config import DOC_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    import pdfplumber

    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            # 기본 추출 (스캔PDF는 빈문자열일 수 있음)
            t = page.extract_text() or ""
            texts.append(t)
    return "\n".join(texts)


def load_texts_from_folder(folder: str = DOC_DIR) -> List[Dict]:
    paths = sorted(
        glob.glob(os.path.join(folder, "*.txt"))
        + glob.glob(os.path.join(folder, "*.pdf"))
    )
    if not paths:
        raise ValueError(
            f"[load_texts_from_folder] {folder}에 .txt/.pdf 파일이 없습니다."
        )

    docs = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext == ".txt":
            txt = _read_txt(p)
        elif ext == ".pdf":
            txt = _read_pdf(p)
        else:
            continue

        if not txt or not txt.strip():
            # 스캔 PDF이거나 빈 파일일 수 있음
            # 필요 시 여기서 raise로 바꾸고 OCR 안내 메시지로 전환하세요.
            continue

        docs.append({"id": os.path.basename(p), "text": txt})

    if not docs:
        raise ValueError(
            "[load_texts_from_folder] 로드된 문서가 모두 비어 있습니다. "
            "스캔 PDF라면 OCR이 필요합니다(아래 참고)."
        )
    return docs


def split_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    # 아주 단순한 슬라이딩 윈도우 청킹
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks


def chunk_docs(raw_docs: List[Dict]) -> List[Dict]:
    out = []
    for d in raw_docs:
        chunks = split_text(d["text"])
        for i, c in enumerate(chunks):
            out.append({"doc_id": d["id"], "chunk_id": i, "chunk": c})
    if not out:
        raise ValueError(
            "[chunk_docs] 생성된 청크가 0개입니다. CHUNK_SIZE/OVERLAP 또는 문서 내용을 확인하세요."
        )
    return out
