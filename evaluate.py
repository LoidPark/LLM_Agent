# evaluate.py
from typing import List, Set
from data_loader import load_texts_from_folder, chunk_docs
from indexing import build_faiss
from retriever import retrieve
from judge import extract_needed_facts, judge_relevance, map_covered_facts
from metrics import context_precision, context_recall
from config import DOC_DIR, TOP_K, INDEX_PATH, STORE_PATH
import os

from ragas_judge import run_ragas_eval


def ensure_index():
    # 인덱스 없으면 빌드
    need_build = not (os.path.exists(INDEX_PATH) and os.path.exists(STORE_PATH))
    if need_build:
        raw_docs = load_texts_from_folder(DOC_DIR)
        chunks = chunk_docs(raw_docs)
        build_faiss(chunks)


def run_eval(question: str, ground_truth: str, top_k: int = TOP_K):
    ensure_index()
    # 1) 검색
    results = retrieve(
        question, top_k=top_k
    )  # [{'chunk':..., 'doc_id':..., 'score':...}, ...]

    print(f"[DEBUG] Retrieved chunks: {len(results)}")
    print(
        f"[DEBUG] First chunk preview: {results[0]['chunk'][:100]}"
    )  # 검색 제대로 됐는지

    from llm import generate_with_context

    contexts_texts = [r["chunk"] for r in results]
    answer = generate_with_context(question, contexts_texts)

    # 2) 팩트 추출
    facts = extract_needed_facts(question, ground_truth)
    print(f"[DEBUG] Extracted facts: {facts}")

    # 전체 코퍼스에 해당 토큰이 실제로 있는지 빠르게 스캔
    from indexing import load_faiss

    _, _all_chunks = load_faiss()
    for f in facts:
        hit = any(f.lower() in c["chunk"].lower() for c in _all_chunks)
        print(f"[DEBUG] FACT '{f}' present_in_corpus? -> {hit}")

    # 3) LLM 관련성 판정 (chunk 단위)
    labels, covered = [], set()
    for r in results:
        is_rel = judge_relevance(question, facts, r["chunk"])
        print(f"[DEBUG] Chunk relevance: {is_rel}")  # True/False 찍힘 확인
        labels.append(is_rel)
        # (옵션) 팩트 커버 추정: 키워드 흉내(간단)
        covered |= set(map_covered_facts(question, facts, r["chunk"]))
    print(f"[DEBUG] Labels summary: {labels}")
    print(f"[DEBUG] Covered facts: {covered}")
    print(f"[DEBUG] Num total facts: {len(facts)}")

    # 4) 메트릭
    p = context_precision(labels)
    r = context_recall(covered, total_facts=len(facts))

    from judge import score_faithfulness, score_answer_relevancy

    faith = score_faithfulness(answer, contexts_texts)
    ans_rel = score_answer_relevancy(question, answer)

    # 출력
    print("\n=== RAG Retrieve ===")
    for i, rch in enumerate(results, 1):
        tag = "REL" if labels[i - 1] else "NON"
        print(
            f"[{i}/{top_k}] ({tag}) score={rch['score']:.3f} doc={rch['doc_id']} chunk#{rch['chunk_id']}"
        )
        print(
            rch["chunk"][:180].replace("\n", " ")
            + ("..." if len(rch["chunk"]) > 180 else "")
        )
        print("-" * 60)

    print(f"\n=== Generated Answer ===\n{answer}\n")

    print("\n=== Needed Facts ===")
    for i, f in enumerate(facts):
        print(f"  - ({i}) {f}")

    print("\n=== DEBUG: Inputs used for BOTH our eval and RAGAS ===")
    print(f"[Q] {question}")
    print(f"[GT] {ground_truth}")
    print(f"[A] {answer}")
    print(f"[contexts used: {len(contexts_texts)}]")
    for i, c in enumerate(contexts_texts):
        # prev = c.replace("\n", " ")[:200]
        prev = c.replace("\n", " ")
        print(f"  - ctx#{i} len={len(c)}: {prev}{'...' if len(c)>200 else ''}")

    print("\n=== DEBUG: Our relevance labels & covered facts ===")
    print(f"labels={labels}")  # e.g., [True, False, True]
    print(f"facts={facts}")  # e.g., ['SVM','DNN','Autoencoder','DT-CNN']")
    print(f"covered_facts={sorted(list(covered))} / total={len(facts)}")

    print(f"\ncontext_precision = {p:.3f}")
    print(f"context_recall    = {r:.3f}")
    print(f"faithfulness      = {faith:.3f}")
    print(f"answer_relevancy  = {ans_rel:.3f}")

    ragas_scores = run_ragas_eval(question, answer, contexts_texts, ground_truth)
    print("\n=== RAGAS Metrics ===")
    print(
        f"ragas.context_precision = {ragas_scores.get('context_precision', float('nan')):.3f}"
    )
    print(
        f"ragas.context_recall    = {ragas_scores.get('context_recall', float('nan')):.3f}"
    )
    print(
        f"ragas.faithfulness      = {ragas_scores.get('faithfulness', float('nan')):.3f}"
    )
    print(
        f"ragas.answer_relevancy  = {ragas_scores.get('answer_relevancy', float('nan')):.3f}"
    )

    from debug_probe import dump_bundle, probe_context_relevance, probe_answer_support

    dump_bundle(question, answer, contexts_texts, ground_truth)
    _ = probe_context_relevance(question, contexts_texts)
    _ = probe_answer_support(answer, contexts_texts)

    from debug_probe import (
        probe_answer_supported_by_contexts,
        probe_each_context_relevance,
    )

    probe_each_context_relevance(question, contexts_texts)
    probe_answer_supported_by_contexts(answer, contexts_texts)

    # ===============================================================
    def _norm(s):
        import re

        return re.sub(r"[^0-9a-z가-힣]+", "", s.lower())

    print("\n=== FACT PRESENCE BY CHUNK (substring audit) ===")
    for f in facts:
        row = []
        nf = _norm(f)
        for i, c in enumerate(contexts_texts):
            hit = nf in _norm(c)
            row.append(f"ctx#{i}:{'Y' if hit else 'N'}")
        print(f" - {f} -> " + ", ".join(row))

    # ===============================================================
    from llm import chat

    joined = "\n\n---\n\n".join(contexts_texts)

    def _supported(claim: str) -> bool:
        out = chat(
            [
                {"role": "system", "content": "Return only 'true' or 'false'."},
                {
                    "role": "user",
                    "content": f'Contexts:\n{joined}\n\nClaim:\n"{claim}"\n\nIs this claim explicitly supported by the contexts?',
                },
            ]
        )
        o = out.strip().lower()
        return o.startswith("t") and "false" not in o

    print("\n=== CLAIM-LEVEL SUPPORT AUDIT ===")
    claims = ["SVM이 있다.", "DNN이 있다.", "Autoencoder가 있다.", "DT-CNN이 있다."]
    for cl in claims:
        print(f" - {cl} -> {_supported(cl)}")
    # ===============================================================

    # from llm_as_a_judge import run_llm_as_a_judge

    # judge6 = run_llm_as_a_judge(question, answer, contexts_texts, ground_truth)

    # print("\n=== LLM-as-a-Judge (6 metrics) ===")
    # print(f"completeness     = {judge6['completeness']:.3f}")
    # print(f"usefulness       = {judge6['usefulness']:.3f}")
    # print(f"clarity          = {judge6['clarity']:.3f}")
    # print(f"relevance        = {judge6['relevance']:.3f}")
    # print(f"additional_value = {judge6['additional_value']:.3f}")
    # print(f"error_penalty    = {judge6['error_penalty']:.3f}")

    # ===============================================================

    # from llm_as_a_judge import run_llm_as_a_judge

    # judge = run_llm_as_a_judge(
    #     question,
    #     answer,
    #     contexts_texts,
    #     ground_truth,
    #     key_claims=facts,  # 예: ['SVM','DNN','Autoencoder','DT-CNN']
    #     weight_additional_value=0.15,
    #     penalty_softness=0.2,
    # )

    # print("\n=== LLM-as-a-Judge (6 metrics + final) ===")
    # for k, v in judge["scores"].items():
    #     print(f"{k:16} = {v:.3f}")
    # print(judge["details"])

    # ===============================================================

    # from llm_as_a_judge import run_llm_as_a_judge
    # from llm_as_a_judge_explain import explain_all

    # # ... (question, answer, contexts_texts, ground_truth, facts 등 준비)
    # judge = run_llm_as_a_judge(
    #     question, answer, contexts_texts, ground_truth, key_claims=facts
    # )

    # print("\n=== LLM-as-a-Judge (scores) ===")
    # for k, v in judge["scores"].items():
    #     print(f"{k:16} = {v:.3f}")

    # # 설명 출력
    # exp = explain_all(judge["details"], question, answer, ground_truth, contexts_texts)

    # from pprint import pprint

    # print("\n=== LLM-as-a-Judge EXPLANATIONS (why each score?) ===")
    # pprint(exp["explanations"]["completeness"])  # 필요 항목만 펼쳐서 보기
    # pprint(exp["explanations"]["usefulness"])
    # pprint(exp["explanations"]["clarity"])
    # pprint(exp["explanations"]["relevance"])
    # pprint(exp["explanations"]["additional_value"])
    # pprint(exp["explanations"]["error_penalty"])

    # ===============================================================

    from explain import explain_bundle

    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_json = f"./reports/judge_explain_{ts}.json"

    bundle = explain_bundle(
        question=question,
        answer=answer,
        contexts=contexts_texts,  # rag retrieve 상위 k개 청크(문장 전체 전달 추천)
        ground_truth=ground_truth,
        key_claims=facts,  # 우리가 추출한 핵심 주장 리스트(없으면 None)
        weight_additional_value=0.15,
        penalty_softness=0.20,
        save_path=save_json,
    )

    # ---- 요약 점수 출력 ----
    print("\n=== LLM-as-a-Judge (6 metrics + final) ===")
    for k, v in bundle["scores"].items():
        print(f"{k:16} = {v:.3f}")

    # ---- 세부 근거: details 일부(원하면 전체 pprint) ----
    from pprint import pprint

    print("\n--- details.completeness ---")
    pprint(bundle["details"]["completeness"])
    print("\n--- details.clarity ---")
    pprint(bundle["details"]["clarity"])
    print("\n--- details.additional_value ---")
    pprint(bundle["details"]["additional_value"])
    print("\n--- details.error_penalty ---")
    pprint(bundle["details"]["error_penalty"])

    # ---- 왜 이런 점수가 나왔는지 설명(각 metric 한 장씩) ----
    exps = bundle["explanations"]
    print("\n=== EXPLANATIONS (why each score?) ===")
    for m in [
        "completeness",
        "usefulness",
        "clarity",
        "relevance",
        "additional_value",
        "error_penalty",
    ]:
        exp = exps.get(m, {})
        print(f"\n[{m}] summary: {exp.get('summary','')}")
        print(f"[{m}] score  : {exp.get('score', None)}")
        # 주요 drivers 2~3개만 요약
        drivers = exp.get("drivers", [])[:3]
        for d in drivers:
            typ = d.get("type", "")
            reason = d.get("reason", "")
            evs = d.get("evidence", [])[:1]
            ev_ref = evs[0].get("ref", "") if evs else ""
            ev_quote = evs[0].get("quote", "") if evs else ""
            print(f'  - {typ}: {reason} | {ev_ref} :: "{ev_quote}"')

    print(f"\n[ saved JSON ] {save_json}")


if __name__ == "__main__":
    # 예시 질의/정답
    q = "고장 예지 및 데이터 처리에 적용되는 기법은 무엇인가?"
    gt = "SVM, DNN, Autoencoder, DT-CNN 이 있다."
    run_eval(q, gt, top_k=3)
