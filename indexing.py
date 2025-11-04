# indexing.py
import pickle
import faiss
import numpy as np
from typing import List, Dict
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer
from config import LOCAL_BGE_PATH, INDEX_PATH, STORE_PATH

# --- BGE-M3 전역 싱글턴 로더 ---
_bge = None


def _get_bge():
    global _bge
    if _bge is None:
        _bge = BGEM3FlagModel(LOCAL_BGE_PATH, use_fp16=False)
    return _bge


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        raise ValueError("[embed_texts] 입력 텍스트가 비어 있습니다.")
    model = _get_bge()
    enc = model.encode(
        texts,
        batch_size=32,
        max_length=8192,
    )  # returns dict with 'dense_vecs' (and sparse/colbert if 설정)
    vecs = np.asarray(enc["dense_vecs"], dtype="float32")  # (N, d)

    # 코사인 유사도용 정규화
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms
    return vecs


def build_faiss(chunks: List[Dict], index_path=INDEX_PATH, store_path=STORE_PATH):
    if not chunks:
        raise ValueError("[build_faiss] 청크가 0개입니다.")
    texts = [c["chunk"] for c in chunks]
    vecs = embed_texts(texts)  # (N, d)
    if vecs.shape[0] == 0:
        raise ValueError("[build_faiss] 임베딩 결과가 비었습니다.")

    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)  # 정규화했으니 IP=cosine
    index.add(vecs)

    faiss.write_index(index, index_path)
    with open(store_path, "wb") as f:
        pickle.dump({"chunks": chunks}, f)


def load_faiss(index_path=INDEX_PATH, store_path=STORE_PATH):
    index = faiss.read_index(index_path)
    with open(store_path, "rb") as f:
        store = pickle.load(f)
    return index, store["chunks"]


# --- 아주 얇은 VectorStore 래퍼 (간단히 쓰기 위함) ---
class SimpleVectorStore:
    def __init__(self, index_path=INDEX_PATH, store_path=STORE_PATH):
        self.index, self.chunks = load_faiss(index_path, store_path)

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Dict]:
        # query_vec: shape (d,) 또는 (1, d)
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        scores, idxs = self.index.search(query_vec.astype("float32"), top_k)
        results = []
        for i, s in zip(idxs[0].tolist(), scores[0].tolist()):
            if i == -1:
                continue
            results.append({**self.chunks[i], "score": float(s)})
        return results
