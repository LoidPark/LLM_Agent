import numpy as np
from typing import List, Dict
from indexing import SimpleVectorStore, embed_texts
from config import TOP_K


def retrieve(question: str, top_k: int = TOP_K) -> List[Dict]:
    # BGE-M3로 질의 임베딩 (embed_texts 재사용)
    q_vec = embed_texts([question])[0]  # (d,)
    store = SimpleVectorStore()
    results = store.search(q_vec, top_k=top_k)
    return results
