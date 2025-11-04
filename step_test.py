from data_loader import load_texts_from_folder, chunk_docs
from indexing import build_faiss, load_faiss
from retriever import retrieve
from judge import extract_needed_facts, judge_relevance, map_covered_facts
from metrics import context_precision, context_recall

# 1) 문서 로드 + 청킹 + 인덱스 빌드
raw = load_texts_from_folder("./docs")
chunks = chunk_docs(raw)
build_faiss(chunks)

# 2) 검색
question = "대한민국의 수도는?"
gt = "대한민국의 수도는 서울이다."
results = retrieve(question, top_k=3)  # List[Dict]: doc_id, chunk_id, chunk, score

# 3) 필요한 팩트 추출(LLM)
facts = extract_needed_facts(question, gt)  # e.g. ["대한민국의 수도는 서울이다"]

# 4) 각 청크 관련성 판정(LLM)
labels = []
covered = set()
for r in results:
    labels.append(judge_relevance(question, facts, r["chunk"]))
    covered |= set(map_covered_facts(question, facts, r["chunk"]))

# 5) 지표
p = context_precision(labels)
r = context_recall(covered, len(facts))
print(p, r)
