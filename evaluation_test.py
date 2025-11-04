# ---- (A) 수작업 라벨 방식 ------------------------------------
def context_precision_recall_from_labels(retrieved_labels, gold_count):
    """
    retrieved_labels: 길이 |R| 리스트. 각 원소는 True(관련) / False(비관련)
    gold_count:      정답에 필요한 관련 컨텍스트 총개수 |G|
    """
    if len(retrieved_labels) == 0:
        precision = 0.0
    else:
        precision = sum(retrieved_labels) / len(retrieved_labels)

    recall = 0.0 if gold_count == 0 else sum(retrieved_labels) / gold_count
    return precision, recall


# 예제 ①: 단일 팩트 (위 손계산과 동일)
labels = [True, False, False]  # c1 관련, c2 비관련, c3 비관련
gold_count = 1  # 필요한 팩트 1개
p, r = context_precision_recall_from_labels(labels, gold_count)
print("Ex1 - precision:", p, "recall:", r)  # 0.333..., 1.0

# 예제 ②: 두 팩트 필요
labels = [True, False]  # cA만 맞음, cB는 못가져옴
gold_count = 2
p, r = context_precision_recall_from_labels(labels, gold_count)
print("Ex2 - precision:", p, "recall:", r)  # 0.5, 0.5


# ---- (B) 아주 단순한 키워드 매칭(연습용) -----------------------
# 주의: 실제 ragas는 LLM/엔테일먼트 등을 써서 더 정교합니다.
def mark_relevance_by_keywords(retrieved_contexts, needed_facts_keywords):
    """
    retrieved_contexts: 리스트[str]  (리트리버가 가져온 문서 조각들)
    needed_facts_keywords: 리스트[세트[str]]
        - 각 세트는 '한 팩트'를 대표하는 키워드 묶음 (AND로 모두 포함되면 맞았다고 보자)
    반환:
        retrieved_labels: 각 리트리브된 조각이 '어느 팩트라도' 만족하면 True
        gold_count: 필요한 팩트 개수 (len(needed_facts_keywords))
        covered_facts: 발견된 팩트 인덱스 집합
    """
    retrieved_labels = []
    covered_facts = set()

    for ctx in retrieved_contexts:
        tokens = set(ctx.lower().split())
        is_rel = False
        for i, fact_keys in enumerate(needed_facts_keywords):
            # 매우 단순하게: 모든 키워드가 토큰에 존재하면 그 팩트 충족
            if fact_keys.issubset(tokens):
                is_rel = True
                covered_facts.add(i)
        retrieved_labels.append(is_rel)

    gold_count = len(needed_facts_keywords)
    return retrieved_labels, gold_count, covered_facts


# 키워드 예제: "서울", "수도"가 동시에 있으면 수도 팩트 충족
retrieved = [
    "서울 은 대한민국 의 수도 다",
    "부산 은 대한민국 제2 의 도시 다",
    "대한민국 의 국기 는 태극기 다",
]
needed_facts = [{"서울", "수도"}]  # 필요한 팩트는 1개

labels, gold_count, covered = mark_relevance_by_keywords(retrieved, needed_facts)
p, r = context_precision_recall_from_labels(labels, gold_count)
print("KW - precision:", p, "recall:", r)
# 기대: precision=1/3, recall=1.0
