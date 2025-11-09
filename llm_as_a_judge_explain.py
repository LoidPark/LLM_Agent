# llm_as_a_judge_explain.py
# 목적: LLM-as-a-Judge metrics에 대해
#       - 왜 이런 점수가 나왔는지
#       - 어떤 fact들이 커버/누락 되었는지
#       - 어떻게 개선하면 되는지
#       - 영어/한국어 설명
# 을 JSON으로 반환.

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
        return {"ok": False, "error": "empty_output", "raw": out}
    try:
        data = json.loads(out)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    m = _NUM_RE.search(out or "")
    return {"ok": False, "raw": out, "hint_number": m.group() if m else None}


def _sentence_split(text: str) -> List[str]:
    if not text:
        return []
    import re

    sents = re.split(r"(?<=[\.!?])\s+|\n+", text.strip())
    return [s.strip() for s in sents if s.strip()]


def _make_catalog(
    question: str,
    answer: str,
    ground_truth: str,
    contexts: List[str],
    facts: Optional[List[str]] = None,
) -> str:
    lines: List[str] = []
    # Q / GT / A
    q_s = _sentence_split(question)
    gt_s = _sentence_split(ground_truth)
    a_s = _sentence_split(answer)

    if q_s:
        lines.append("Q:")
        for i, s in enumerate(q_s, 1):
            lines.append(f"  ({i}) {s}")
    if gt_s:
        lines.append("GT:")
        for i, s in enumerate(gt_s, 1):
            lines.append(f"  ({i}) {s}")
    if a_s:
        lines.append("A:")
        for i, s in enumerate(a_s, 1):
            lines.append(f"  ({i}) {s}")

    # FACTS (optional)
    if facts:
        lines.append("FACTS (from ground_truth):")
        for i, f in enumerate(facts, 1):
            lines.append(f"  [{i}] {f}")

    # CONTEXTS (지금 비교 러너에서는 비어있지만, 일반 RAG 파이프라인에서는 사용 가능)
    for ci, c in enumerate(contexts):
        cs = _sentence_split(c)
        if cs:
            lines.append(f"CTX{ci}:")
            for j, s in enumerate(cs, 1):
                lines.append(f"  ({j}) {s}")

    return "\n".join(lines) if lines else "(no text)"


# ========= 시스템 프롬프트 =========
_SYS = """You are an evaluation explanation generator for LLM answers.

You MUST:
- Return STRICT JSON (no extra text).
- Explain WHY a metric got its score.
- Give explanations in BOTH English and Korean.
- When possible, ground your explanation in:
  - query (Q)
  - ground_truth (GT)
  - facts (FACTS)
  - answer (A)
- If facts list is provided, use it for coverage/missing analysis.
"""

# ========= 사용자 프롬프트(메트릭별) =========
_USR_GENERAL = """
We evaluated an answer with multiple metrics.
Now we focus on ONE metric: "{metric_name}".

Indexed texts and facts:
{catalog}

Metric detail object:
{metric_detail_json}

If available, the completeness/error_penalty detail may contain:
- "requirements": list of fact strings
- "covered": list of booleans aligned with requirements
Use that to compute coverage/missing if relevant.

Your task:
Return STRICT JSON with this structure:

{{
  "metric": "{metric_name}",
  "score": {score_or_null},

  "summary_en": "<1-2 sentence summary in English>",
  "summary_ko": "<위 내용을 한국어로 1-2문장 요약>",

  "root_cause": {{
    "main_factors_en": "<what main factors caused this score (English)>",
    "main_factors_ko": "<위 내용을 한국어로 설명>",
    "facts_total": <int or null>,
    "facts_covered": <int or null>,
    "facts_missing": [ "<fact1>", "<fact2>" ],
    "facts_covered_list": [ "<fact1>", "<fact2>" ]
  }},

  "improvement_en": "<concrete suggestions in English on how to improve the metric>",
  "improvement_ko": "<위 제안 사항을 한국어로 표현>",

  "drivers": [
    {{
      "type": "positive" | "negative" | "neutral",
      "reason_en": "<short reason in English>",
      "reason_ko": "<same reason in Korean>",
      "evidence": [
        {{
          "ref": "Q(1)" | "GT(1)" | "A(1)" | "FACTS[1]" | "CTX0(1)" | ...,
          "quote": "<<=20 words from that sentence (can be Korean or English)>",
          "interpretation_en": "<why this evidence matters (English)>",
          "interpretation_ko": "<왜 이 근거가 중요한지 한국어로 설명>"
        }}
      ]
    }}
  ],

  "edge_cases": "<optional: when the score could have been higher/lower>",
  "confidence": 0.0-1.0
}}

Rules:
- If facts are not relevant for this metric (e.g., clarity), set facts_* fields to null or [].
- Be concise but specific. Avoid generic comments.
"""


def explain_metric(
    metric_name: str,
    detail: Dict[str, Any],
    catalog: str,
) -> Dict[str, Any]:
    # 기존 detail 안의 score를 기본값으로 넘김 (없으면 null)
    base_score = detail.get("score", None)
    score_literal = "null" if base_score is None else float(base_score)

    out = chat(
        [
            {"role": "system", "content": _SYS},
            {
                "role": "user",
                "content": _USR_GENERAL.format(
                    catalog=catalog,
                    metric_detail_json=json.dumps(detail, ensure_ascii=False, indent=2),
                    metric_name=metric_name,
                    score_or_null=score_literal,
                ),
            },
        ]
    )
    data = _safe_json(out)
    data["metric"] = data.get("metric", metric_name)
    # score 필드 정리
    if "score" in data and data["score"] is not None:
        data["score"] = float(_clamp01(data["score"]))
    else:
        data["score"] = float(_clamp01(base_score if base_score is not None else 0.0))
    return data


def explain_all(
    judge_details: Dict[str, Any],
    question: str,
    answer: str,
    ground_truth: str,
    contexts: List[str],
    facts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    judge_details: run_llm_as_a_judge(...)가 반환한 ["details"]
    facts: ground_truth에서 추출한 key facts 리스트 (있으면 루트 원인 분석에 활용)
    반환: {"ok": True, "explanations": {metric_name: {...}}}
    """
    catalog = _make_catalog(question, answer, ground_truth, contexts, facts)
    exps: Dict[str, Any] = {}
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
        exps[m] = explain_metric(m, detail, catalog)

    return {"ok": True, "explanations": exps}
