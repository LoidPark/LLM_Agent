# explain.py
# 목적: (1) LLM-as-a-Judge 평가(점수/디테일) 실행 → (2) 왜 그런 점수인지 설명 생성 → (3) 선택적으로 JSON 저장
# 요구: llm_as_a_judge.py, llm_as_a_judge_explain.py, llm.py(chat) 가 같은 폴더에 존재

from typing import List, Dict, Any, Optional
import json, os
from datetime import datetime

from llm_as_a_judge import run_llm_as_a_judge
from llm_as_a_judge_explain import explain_all


def explain_bundle(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
    key_claims: Optional[List[str]] = None,
    weight_additional_value: float = 0.15,
    penalty_softness: float = 0.20,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    평가 + 설명을 한 번에 실행하고, 필요 시 JSON 파일로 저장.

    반환 구조:
    {
      "scores": {..., "final": 0.xxx},
      "details": {... per metric ...},
      "explanations": {metric_name: {...}},
      "formula": "...",
      "meta": {"question": ..., "answer": ..., "ground_truth": ..., "num_contexts": ...}
    }
    """
    # 1) 점수 + 디테일
    judge = run_llm_as_a_judge(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
        key_claims=key_claims,
        weight_additional_value=weight_additional_value,
        penalty_softness=penalty_softness,
    )

    # 2) 왜 그런 점수인지 설명
    exps = explain_all(
        judge_details=judge["details"],
        question=question,
        answer=answer,
        ground_truth=ground_truth,
        contexts=contexts,
        facts=key_claims,
    )

    bundle = {
        "scores": judge["scores"],
        "details": judge["details"],  # requirements/subscores 등 세부 근거
        "explanations": exps.get("explanations", {}),
        "formula": judge.get("formula", ""),
        "meta": {
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "num_contexts": len(contexts),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "weights": {
                "weight_additional_value": weight_additional_value,
                "penalty_softness": penalty_softness,
            },
        },
    }

    # 3) 저장(옵션)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False, indent=2)

    return bundle


# 단독 실행 테스트용 (옵션)
if __name__ == "__main__":
    # 간단 샘플 (실전에서는 evaluate.py에서 호출)
    q = "고장 예지 및 데이터 처리에 적용되는 기법은 무엇인가?"
    gt = "SVM, DNN, Autoencoder, DT-CNN 이 있다."
    ctxs = ["예시 컨텍스트 1", "예시 컨텍스트 2"]
    ans = "고장 예지 및 데이터 처리에 적용되는 기법은 SVM, DNN, Autoencoder, DT-CNN이 있다."
    facts = ["SVM", "DNN", "Autoencoder", "DT-CNN"]

    out = explain_bundle(
        question=q,
        answer=ans,
        contexts=ctxs,
        ground_truth=gt,
        key_claims=facts,
        save_path="./reports/demo_judge_explain.json",
    )
    print("[OK] wrote ./reports/demo_judge_explain.json")
