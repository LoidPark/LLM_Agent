# llm_as_a_judge_explain.py
# 목적: llm_as_a_judge의 metrics/detail에 대해
#       "왜 그런 점수가 나왔는지"를 LLM이 증거 기반으로 설명.
# 의존: llm.py(chat)

from typing import List, Dict, Any, Optional
import json, re
from llm import chat

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def _safe_json(out: str) -> Dict[str, Any]:
    if not out:
        return {"ok": False, "error": "empty_output"}
    try:
        data = json.loads(out)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    m = _NUM_RE.search(out or "")
    return {"ok": False, "raw": out, "hint_number": m.group() if m else None}


def _sentence_split(text: str) -> List[str]:
    # 간단/안전한 문장 분할(마침표/물음표/느낌표/개행 기준). 과도한 정교화는 피함.
    if not text:
        return []
    import re

    sents = re.split(r"(?<=[\.!?])\s+|\n+", text.strip())
    return [s.strip() for s in sents if s.strip()]


def _make_catalog(
    question: str, answer: str, gt: str, contexts: List[str]
) -> Dict[str, List[str]]:
    cat = {
        "Q": _sentence_split(question),
        "A": _sentence_split(answer),
        "GT": _sentence_split(gt),
        "CTX0": _sentence_split(contexts[0]) if len(contexts) > 0 else [],
        "CTX1": _sentence_split(contexts[1]) if len(contexts) > 1 else [],
        "CTX2": _sentence_split(contexts[2]) if len(contexts) > 2 else [],
        "CTX3": _sentence_split(contexts[3]) if len(contexts) > 3 else [],
        "CTX4": _sentence_split(contexts[4]) if len(contexts) > 4 else [],
    }
    return cat


def _catalog_to_prompt(cat: Dict[str, List[str]]) -> str:
    # "문장 번호가 매겨진 인덱스"를 프롬프트로 구성
    lines = []
    for section in ["Q", "A", "GT", "CTX0", "CTX1", "CTX2", "CTX3", "CTX4"]:
        sents = cat.get(section, [])
        if sents:
            lines.append(f"{section}:")
            for i, s in enumerate(sents, 1):
                lines.append(f"  ({i}) {s}")
    return "\n".join(lines) if lines else "(no text)"


# ========= 공통 시스템 프롬프트 =========
_SYS = """You are an explanation generator for evaluation metrics.
You must return STRICT JSON with specific fields. Do not include any prose outside JSON.
When citing evidence, ALWAYS point to specific sentences by section+index, e.g., "A(1)", "CTX2(3)", "GT(2)", "Q(1)".
Prefer concise quotes (<= 20 words) from those sentences.
Be honest and precise; if uncertain, say so explicitly in JSON."""

# ========= 지표별 사용자 프롬프트 =========
_USR_GENERAL = """We evaluated an answer with multiple metrics. 
Explain WHY each score was assigned, citing evidence from the indexed sentences.

Indexed texts:
{catalog}

Metric details (as provided by the judge):
{metric_detail_json}

Your task:
- For the metric "{metric_name}", analyze the provided detail object and the indexed texts.
- Return JSON with:
{{
  "metric": "{metric_name}",
  "summary": "<1-2 sentence summary of why the score was assigned>",
  "score": <number if present in detail else null>,
  "drivers": [
     {{
       "type": "positive" | "negative" | "neutral",
       "reason": "<short reason>",
       "evidence": [
          {{
            "ref": "A(1)" | "CTX0(2)" | "GT(1)" | "Q(1)" | ...,
            "quote": "<<=20 words quote from that sentence>",
            "interpretation": "<why this evidence matters>"
          }}
       ]
     }}
  ],
  "edge_cases": "<optional: where the metric could have differed>",
  "confidence": 0.0-1.0
}}"""


# ========= 마스터 함수: 하나의 metric 설명 =========
def explain_metric(
    metric_name: str, detail: Dict[str, Any], catalog: Dict[str, List[str]]
) -> Dict[str, Any]:
    cat_prompt = _catalog_to_prompt(catalog)
    detail_json = json.dumps(detail, ensure_ascii=False, indent=2)
    out = chat(
        [
            {"role": "system", "content": _SYS},
            {
                "role": "user",
                "content": _USR_GENERAL.format(
                    catalog=cat_prompt,
                    metric_detail_json=detail_json,
                    metric_name=metric_name,
                ),
            },
        ]
    )
    data = _safe_json(out)
    data["metric"] = data.get("metric", metric_name)
    return data


# ========= 번들 호출: 6개 지표 일괄 설명 =========
def explain_all(
    judge_details: Dict[str, Any],
    question: str,
    answer: str,
    ground_truth: str,
    contexts: List[str],
) -> Dict[str, Any]:
    """
    judge_details: run_llm_as_a_judge(... )가 반환한 객체의 ["details"]
    반환: {
      "explanations": {metric_name: {...json...}},
      "ok": true
    }
    """
    cat = _make_catalog(question, answer, ground_truth, contexts)
    exps = {}
    # metrics 순서는 보기 좋게 고정
    order = [
        "completeness",
        "usefulness",
        "clarity",
        "relevance",
        "additional_value",
        "error_penalty",
    ]
    for m in order:
        detail = judge_details.get(m, {})
        exps[m] = explain_metric(m, detail, cat)
    return {"ok": True, "explanations": exps}
