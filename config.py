# config.py

import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_BGE_PATH = "./models/bge-m3"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

DOC_DIR = "./docs"  # .txt 위주로 예시
INDEX_PATH = "./faiss.index"
STORE_PATH = "./store.pkl"  # 문서 텍스트/메타 저장(피클)
