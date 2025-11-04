# llm_as_a_judge.py
# 6 metrics + final score combiner
# - completeness, usefulness, clarity, relevance, additional_value, error_penalty
# - usefulness / relevance: 지정된 rubric 그대로 사용
# - error_penalty: key_claims 누락/모순/비근거 주장 등 감점
# Requires: llm.py(chat)

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
        return {"score": 0.0}
    try:
        data = json.loads(out)
        if isinstance(data, dict):
            if "score" in data:
                data["score"] = _clamp01(data["score"])
            return data
    except Exception:
        pass
    m = _NUM_RE.search(out or "")
    return {"score": _clamp01(float(m.group()))} if m else {"score": 0.0}


def _join_ctx(contexts: List[str], max_ctx: int = 5, max_chars: int = 4500) -> str:
    # 간단 상한: 평가 LLM 토큰 초과 방지
    acc, parts = 0, []
    for c in contexts[:max_ctx]:
        c = (c or "").strip()
        if not c:
            continue
        if acc + len(c) > max_chars:
            parts.append(c[: max(0, max_chars - acc)])
            break
        parts.append(c)
        acc += len(c)
    return "\n\n---\n\n".join(parts)


# =========================
# 1) COMPLETENESS
# =========================
_SYS_COMPLETENESS = """You are a meticulous evaluation judge.
Grade COMPLETENESS in [0.0,1.0] as the fraction of covered atomic requirements.
Return strict JSON only:
{
  "requirements": [<strings>],
  "covered": [<booleans aligned with requirements>],
  "coverage": <0..1>,
  "score": <0..1>
}
Rubric:
- 0.0: Missing most required elements; incomplete or irrelevant.
- 0.25: Covers a few aspects, but many gaps remain.
- 0.50: Covers about half of the expected content.
- 0.75: Covers most major points; minor omissions.
- 1.00: Fully complete; all core elements present and accurate.
Detailed rubric:
- Identify minimal, non-overlapping atomic requirements necessary to fully answer the question.
- Use Ground Truth (GT) as authoritative; use Contexts only to disambiguate or fill ellipses in GT.
- A requirement is 'covered' only if the Answer states it explicitly or via an unambiguous paraphrase.
- Do NOT credit implied/assumed content not present in the Answer.
- coverage = (#covered / #requirements); score = coverage.
- Be strict but fair; avoid double-counting overlapping requirements.
"""

_USR_COMPLETENESS = """Question:
{q}

Ground Truth:
{gt}

Contexts (optional):
{ctx}

Answer:
{ans}

Steps:
1) List atomic requirements (2–8 items).
2) Mark covered true/false for each requirement based on explicit or clear paraphrase in the Answer.
3) Compute coverage=(#covered/#requirements) and set score=coverage.
Return JSON only."""


def score_completeness(
    q: str, ans: str, gt: str, contexts: List[str]
) -> Dict[str, Any]:
    ctx = _join_ctx(contexts)
    out = chat(
        [
            {"role": "system", "content": _SYS_COMPLETENESS},
            {
                "role": "user",
                "content": _USR_COMPLETENESS.format(q=q, gt=gt, ctx=ctx, ans=ans),
            },
        ]
    )
    data = _safe_json(out)
    reqs = data.get("requirements", [])
    covs = data.get("covered", [])
    if (
        isinstance(reqs, list)
        and isinstance(covs, list)
        and len(reqs) == len(covs)
        and len(reqs) > 0
    ):
        cov_ratio = sum(1 for x in covs if bool(x)) / len(reqs)
        data["coverage"] = round(_clamp01(cov_ratio), 3)
        data["score"] = round(_clamp01(data.get("score", cov_ratio)), 3)
    else:
        data["coverage"] = round(
            _clamp01(data.get("coverage", data.get("score", 0.0))), 3
        )
        data["score"] = round(_clamp01(data.get("score", 0.0)), 3)
    return data


# =========================
# 2) USEFULNESS  (Rubric as provided)
# =========================
_SYS_USEFUL = """You are a pragmatic evaluation judge.
Grade USEFULNESS in [0.0,1.0] using the rubric below.
Return strict JSON only: {"score": <0..1>}
Rubric (choose a value then optionally interpolate):
- 0.0: Useless or wrong or non-answer.
- 0.25: Vague, generic, or lacks specifics; low practical value.
- 0.50: Some useful info; partially actionable but missing key specifics or caveats.
- 0.75: Largely useful; clear, specific, and actionable with minor gaps or caveats.
- 1.00: Highly useful; directly answers core need with precise details, constraints, or next steps, minimal fluff.
"""

_USR_USEFUL = """Question:
{q}

Answer:
{ans}

Give only JSON: {{"score": number in [0,1]}}"""


def score_usefulness(q: str, ans: str) -> Dict[str, Any]:
    out = chat(
        [
            {"role": "system", "content": _SYS_USEFUL},
            {"role": "user", "content": _USR_USEFUL.format(q=q, ans=ans)},
        ]
    )
    data = _safe_json(out)
    data["score"] = round(_clamp01(data.get("score", 0.0)), 3)
    return data


# =========================
# 3) CLARITY (detailed rubric)
# =========================
_SYS_CLAR = """You are a style and clarity judge.
Grade CLARITY in [0.0,1.0] based on detailed subscores, return strict JSON only:
{
  "subscores": {
    "readability_structure": <0..1>,   // logical organization, lists, sentence flow
    "terminology_correctness": <0..1>, // correct & consistent technical terms
    "ambiguity_absence": <0..1>,       // clear referents; minimal vague phrasing
    "conciseness_balance": <0..1>      // concise yet sufficiently informative
  },
  "score": <0..1>
}
Scoring guide:
- 0.0–0.25: Hard to read; poor structure; misused terms; highly ambiguous or verbose.
- 0.5: Generally readable; occasional ambiguity or term misuse; verbosity exists.
- 0.75: Clear and well-structured; minor ambiguity/verbosity only.
- 1.0: Very clear, well-structured, precise terminology, minimal redundancy.
Score = mean(subscores)."""

_USR_CLAR = """Answer:
{ans}

Return JSON only."""


def score_clarity(ans: str) -> Dict[str, Any]:
    out = chat(
        [
            {"role": "system", "content": _SYS_CLAR},
            {"role": "user", "content": _USR_CLAR.format(ans=ans)},
        ]
    )
    data = _safe_json(out)
    subs = data.get("subscores", {})
    if isinstance(subs, dict) and subs:
        ks = [
            "readability_structure",
            "terminology_correctness",
            "ambiguity_absence",
            "conciseness_balance",
        ]
        vals = [_clamp01(subs.get(k, 0.0)) for k in ks]
        for k, v in zip(ks, vals):
            subs[k] = round(v, 3)
        meanv = sum(vals) / len(vals) if vals else 0.0
        data["score"] = round(_clamp01(data.get("score", meanv)), 3)
        data["subscores"] = subs
    else:
        data["score"] = round(_clamp01(data.get("score", 0.0)), 3)
    return data


# =========================
# 4) RELEVANCE (Rubric as provided)
# =========================
_SYS_REL = """You are a strict evaluation judge.
Grade RELEVANCE in [0.0,1.0] using the rubric below.
Return strict JSON only: {"score": <0..1>}
Rubric:
- 0.0: Irrelevant or mostly off-topic.
- 0.25: Weakly related; substantial digressions.
- 0.50: Partially addresses the question, with noticeable off-topic or missing focus.
- 0.75: Mostly on-topic; minor tangents or missing pieces.
- 1.00: Directly on-topic and focused; no needless digressions.
"""

_USR_REL = """Question:
{q}

Contexts (optional):
{ctx}

Answer:
{ans}

Return JSON only."""


def score_relevance(q: str, ans: str, contexts: List[str]) -> Dict[str, Any]:
    ctx = _join_ctx(contexts)
    out = chat(
        [
            {"role": "system", "content": _SYS_REL},
            {"role": "user", "content": _USR_REL.format(q=q, ctx=ctx, ans=ans)},
        ]
    )
    data = _safe_json(out)
    data["score"] = round(_clamp01(data.get("score", 0.0)), 3)
    return data


# =========================
# 5) ADDITIONAL VALUE (detailed rubric)
# =========================
_SYS_ADD = """You are an evaluation judge for ADDITIONAL VALUE.
Grade in [0.0,1.0] with subscores; return strict JSON only:
{
  "subscores": {
    "insight_novelty": <0..1>,         // non-obvious insights beyond bare answer
    "examples_illustration": <0..1>,   // concrete examples or micro-cases
    "comparisons_tradeoffs": <0..1>,   // concise comparisons; pros/cons; selection criteria
    "references_pointers": <0..1>      // pointers to standards, datasets, docs, reproducible steps
  },
  "score": <0..1>
}
Rubric guide:
- 0.0: No extra value beyond minimal answer; boilerplate.
- 0.25: Slight hints; mostly generic.
- 0.50: Some concrete additions that help understanding or application.
- 0.75: Clear, helpful extras (examples/comparisons/references) with minor gaps.
- 1.00: Strong added value with multiple helpful facets; concise and relevant.
Score = mean(subscores)."""

_USR_ADD = """Question:
{q}

Answer:
{ans}

Return JSON only."""


def score_additional_value(q: str, ans: str) -> Dict[str, Any]:
    out = chat(
        [
            {"role": "system", "content": _SYS_ADD},
            {"role": "user", "content": _USR_ADD.format(q=q, ans=ans)},
        ]
    )
    data = _safe_json(out)
    subs = data.get("subscores", {})
    if isinstance(subs, dict) and subs:
        ks = [
            "insight_novelty",
            "examples_illustration",
            "comparisons_tradeoffs",
            "references_pointers",
        ]
        vals = [_clamp01(subs.get(k, 0.0)) for k in ks]
        for k, v in zip(ks, vals):
            subs[k] = round(v, 3)
        meanv = sum(vals) / len(vals) if vals else 0.0
        data["score"] = round(_clamp01(data.get("score", meanv)), 3)
        data["subscores"] = subs
    else:
        data["score"] = round(_clamp01(data.get("score", 0.0)), 3)
    return data


# =========================
# 6) ERROR PENALTY (detailed rubric + key_claims check)
# =========================
_SYS_ERR = """You are an error and risk auditor for ERROR_PENALTY (a 'cleanliness' score).
Return strict JSON only:
{
  "subscores": {
    "factual_accuracy": <0..1>,          // correctness of claims wrt GT/Contexts
    "contradiction_to_contexts": <0..1>, // no contradictions vs provided Contexts/GT
    "unsupported_inference": <0..1>,     // avoids unevidenced assertions
    "safety_and_risk": <0..1>,           // avoids unsafe/biased/inappropriate guidance
    "key_claims_presence": <0..1>        // fraction of key claims present in the Answer
  },
  "score": <0..1>
}
Rubric (lower is better):
- 0.0: No factual errors, all key claims covered.
- 0.25: Minor factual imprecision or one missing key claim.
- 0.50: Some factual issues or key omissions.
- 0.75: Many incorrect statements or missing most key claims.
- 1.00: Major factual errors or completely incorrect content.
Detailed rubric:
- factual_accuracy: Penalize incorrect statements; 1.0 means no detected inaccuracies.
- contradiction_to_contexts: Penalize contradictions with GT/Contexts; 1.0 means none.
- unsupported_inference: Penalize claims lacking evidence in GT/Contexts; 1.0 means claims are supported or cautious.
- safety_and_risk: Penalize unsafe/biased content; 1.0 means clean.
- key_claims_presence: If a list of key claims is provided, compute (#present / #key_claims). If no key claims provided, set to 1.0.
Overall score = mean(subscores) (higher is cleaner; used as a mild penalty multiplier later)."""

_USR_ERR = """Question:
{q}

Ground Truth:
{gt}

Contexts (optional):
{ctx}

Answer:
{ans}

Key claims (optional list; if empty, ignore):
{key_claims}

Return JSON only."""


def score_error_penalty(
    q: str, ans: str, gt: str, contexts: List[str], key_claims: Optional[List[str]]
) -> Dict[str, Any]:
    ctx = _join_ctx(contexts)
    kc = key_claims or []
    kc_text = "\n".join(f"- {k}" for k in kc) if kc else "(none)"
    out = chat(
        [
            {"role": "system", "content": _SYS_ERR},
            {
                "role": "user",
                "content": _USR_ERR.format(
                    q=q, gt=gt, ctx=ctx, ans=ans, key_claims=kc_text
                ),
            },
        ]
    )
    data = _safe_json(out)
    subs = data.get("subscores", {})
    if isinstance(subs, dict) and subs:
        ks = [
            "factual_accuracy",
            "contradiction_to_contexts",
            "unsupported_inference",
            "safety_and_risk",
            "key_claims_presence",
        ]
        vals = [_clamp01(subs.get(k, 0.0)) for k in ks]
        for k, v in zip(ks, vals):
            subs[k] = round(v, 3)
        meanv = sum(vals) / len(vals) if vals else 0.0
        data["score"] = round(_clamp01(data.get("score", meanv)), 3)
        data["subscores"] = subs
    else:
        # key_claims가 주어졌다면, 최소한 key_claims_presence를 강제로 반영할 수도 있음(옵션)
        data["score"] = round(_clamp01(data.get("score", 0.0)), 3)
    return data


# =========================
# FINAL: Orchestrator
# =========================
def run_llm_as_a_judge(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
    key_claims: Optional[List[str]] = None,
    weight_additional_value: float = 0.15,
    penalty_softness: float = 0.2,  # error_penalty가 0일 때 0.8배, 1일 때 1.0배
) -> Dict[str, Any]:
    """
    Returns:
    {
      "scores": {
        "completeness": 1.0,
        "usefulness": 0.75,
        "clarity": 0.9,
        "relevance": 1.0,
        "additional_value": 0.5,
        "error_penalty": 0.95,
        "final": 0.93
      },
      "details": { ... per-metric objects ... },
      "formula": "final = ((avg(C,U,Cl,R)*(1-w) + AV*w) * (0.8 + 0.2*EP))"
    }
    """
    comp = score_completeness(question, answer, ground_truth, contexts)
    usef = score_usefulness(question, answer)
    clar = score_clarity(answer)
    rel = score_relevance(question, answer, contexts)
    addv = score_additional_value(question, answer)
    err = score_error_penalty(question, answer, ground_truth, contexts, key_claims)

    C = _clamp01(comp.get("score", 0.0))
    U = _clamp01(usef.get("score", 0.0))
    Cl = _clamp01(clar.get("score", 0.0))
    R = _clamp01(rel.get("score", 0.0))
    AV = _clamp01(addv.get("score, ", addv.get("score", 0.0)))
    EP = _clamp01(err.get("score", 0.0))

    base = (C + U + Cl + R) / 4.0
    w = max(0.0, min(0.5, weight_additional_value))  # 안전범위
    combined = base * (1 - w) + AV * w

    # "약간"의 감점: factor = 1 - softness*(1-EP) = (1-softness) + softness*EP
    softness = max(0.0, min(0.5, penalty_softness))  # 0~0.5 권장
    penalty_factor = (
        1.0 - softness
    ) + softness * EP  # EP=0 -> (1-softness), EP=1 -> 1.0
    final = _clamp01(combined * penalty_factor)

    return {
        "scores": {
            "completeness": round(C, 3),
            "usefulness": round(U, 3),
            "clarity": round(Cl, 3),
            "relevance": round(R, 3),
            "additional_value": round(AV, 3),
            "error_penalty": round(EP, 3),
            "final": round(final, 3),
        },
        "details": {
            "completeness": comp,
            "usefulness": usef,
            "clarity": clar,
            "relevance": rel,
            "additional_value": addv,
            "error_penalty": err,
        },
        "formula": f"final = ((avg(C,U,Cl,R)*(1-{w:.2f}) + AV*{w:.2f}) * ((1-{softness:.2f}) + {softness:.2f}*EP))",
    }


# # llm_as_a_judge.py
# # 목적: LLM-as-a-Judge 평가 (6개 지표)
# # - completeness (requirements 기반, coverage%)
# # - usefulness (실용성; subscores 포함)
# # - clarity (명료성; subscores 포함)
# # - relevance (질문/문맥 적합성; subscores 포함)
# # - additional_value (부가가치; subscores 포함)
# # - error_penalty (오류/모순/안전 문제; subscores 포함, 감점 성격)
# #
# # 출력 스케일: 모든 score는 0.0 ~ 1.0
# # 의존: llm.py의 chat(), config.OPENAI_MODEL 간접 사용
# # 사용: from llm_as_a_judge import run_llm_as_a_judge

# from typing import List, Dict, Any
# import json, re
# from llm import chat

# # -------------------- 공통 유틸 --------------------

# _NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


# def _clamp01(x: float) -> float:
#     try:
#         v = float(x)
#     except Exception:
#         return 0.0
#     if v < 0.0:
#         return 0.0
#     if v > 1.0:
#         return 1.0
#     return v


# def _safe_json_parse(out: str) -> Dict[str, Any]:
#     """
#     가능한 한 JSON으로 파싱.
#     실패 시 {"score": float} 정도라도 복구.
#     """
#     if not out:
#         return {"score": 0.0}
#     try:
#         data = json.loads(out)
#         # 점수만 있어도 통과
#         if isinstance(data, dict):
#             if "score" in data:
#                 data["score"] = _clamp01(data["score"])
#             return data
#     except Exception:
#         pass
#     # 숫자만 추출해서 score 로 반환
#     m = _NUM_RE.search(out)
#     if m:
#         return {"score": _clamp01(float(m.group()))}
#     return {"score": 0.0}


# def _join_ctx(contexts: List[str], max_ctx: int = 5, max_chars: int = 4000) -> str:
#     # 판정 LLM 토큰 초과 방지를 위한 컨텍스트 합치기
#     snippets, acc = [], 0
#     for c in contexts[:max_ctx]:
#         c = (c or "").strip()
#         if not c:
#             continue
#         if acc + len(c) + 8 > max_chars:
#             remain = max_chars - acc - 8
#             if remain > 0:
#                 snippets.append(c[:remain])
#                 acc += remain
#             break
#         snippets.append(c)
#         acc += len(c) + 8
#     return "\n\n---\n\n".join(snippets)


# # -------------------- COMPLETENESS --------------------
# # 정의: 질문/GT가 요구하는 원자적 요구사항을 Answer가 얼마나 충족했는가 (coverage).
# # 출력: requirements[], covered[], coverage, score

# _SYS_COMPLETENESS = """You are a meticulous evaluation judge.
# Grade COMPLETENESS on [0.0,1.0] as the fraction of covered atomic requirements.
# Output strict JSON only:
# {"requirements":[<strings>],
#  "covered":[<booleans aligned with requirements>],
#  "coverage": <0..1>,
#  "score": <0..1>}
# Rules:
# - Extract minimal, non-overlapping atomic requirements to fully answer the Question.
# - Use Ground Truth as authoritative; use Contexts only to disambiguate if GT is terse.
# - Count covered=true only if the Answer explicitly states or clearly paraphrases the requirement.
# - No credit for implied assumptions not present in the Answer.
# - coverage = (#covered / #requirements); score = coverage."""

# _USR_COMPLETENESS = """Question:
# {q}

# Ground Truth:
# {gt}

# Contexts (optional):
# {ctx}

# Answer:
# {ans}

# Steps:
# 1) List atomic requirements (2–8 items).
# 2) Mark covered true/false for each requirement based on the Answer.
# 3) Compute coverage=(#covered/#requirements) and set score=coverage.
# Return JSON only."""


# def score_completeness(
#     question: str, answer: str, ground_truth: str, contexts: List[str]
# ) -> Dict[str, Any]:
#     ctx = _join_ctx(contexts)
#     out = chat(
#         [
#             {"role": "system", "content": _SYS_COMPLETENESS},
#             {
#                 "role": "user",
#                 "content": _USR_COMPLETENESS.format(
#                     q=question, gt=ground_truth, ctx=ctx, ans=answer
#                 ),
#             },
#         ]
#     )
#     data = _safe_json_parse(out)
#     # 방어
#     reqs = data.get("requirements", [])
#     covs = data.get("covered", [])
#     if (
#         isinstance(reqs, list)
#         and isinstance(covs, list)
#         and len(reqs) == len(covs)
#         and len(reqs) > 0
#     ):
#         covered_cnt = sum(1 for x in covs if bool(x))
#         coverage = covered_cnt / len(reqs)
#         data["coverage"] = round(_clamp01(coverage), 3)
#         data["score"] = round(_clamp01(data.get("score", coverage)), 3)
#     else:
#         data["coverage"] = round(
#             _clamp01(data.get("coverage", data.get("score", 0.0))), 3
#         )
#         data["score"] = round(_clamp01(data.get("score", 0.0)), 3)
#     return data


# # -------------------- USEFULNESS --------------------
# # 정의: 실무적으로 "쓸모"가 있는가. 상세 루브릭을 subscores 로 제공.
# # subscores: clarity_for_use(이 지표에서의 실무적 명료성), specificity, actionability,
# #            constraints_caveats, guidance_next_steps, nonfluff
# # score = mean(subscores)

# _SYS_USEFULNESS = """You are a pragmatic evaluation judge for USEFULNESS.
# Return strict JSON only:
# {
#  "subscores": {
#    "clarity_for_use": <0..1>,
#    "specificity": <0..1>,
#    "actionability": <0..1>,
#    "constraints_caveats": <0..1>,
#    "guidance_next_steps": <0..1>,
#    "nonfluff": <0..1>
#  },
#  "score": <0..1>
# }
# Rubric:
# - clarity_for_use: Language is unambiguous and task-oriented.
# - specificity: Concrete details/examples/criteria (not just labels).
# - actionability: Provides steps, decision rules, or how-to guidance.
# - constraints_caveats: States assumptions, limitations, risks, or scope.
# - guidance_next_steps: Advises what to do next (e.g., baseline setup, evaluation plan).
# - nonfluff: Penalize filler/hallucination; reward concise substance.
# Score = mean(subscores)."""

# _USR_USEFULNESS = """Question:
# {q}

# Answer:
# {ans}

# Evaluate practical helpfulness for a practitioner who must *use* the answer.
# Fill all subscores in [0,1], set score=mean(subscores). Return JSON only."""


# def score_usefulness(question: str, answer: str) -> Dict[str, Any]:
#     out = chat(
#         [
#             {"role": "system", "content": _SYS_USEFULNESS},
#             {"role": "user", "content": _USR_USEFULNESS.format(q=question, ans=answer)},
#         ]
#     )
#     data = _safe_json_parse(out)
#     subs = data.get("subscores", {})
#     if isinstance(subs, dict) and subs:
#         vals = []
#         for k in [
#             "clarity_for_use",
#             "specificity",
#             "actionability",
#             "constraints_caveats",
#             "guidance_next_steps",
#             "nonfluff",
#         ]:
#             vals.append(_clamp01(subs.get(k, 0.0)))
#             subs[k] = round(_clamp01(subs.get(k, 0.0)), 3)
#         meanv = sum(vals) / len(vals) if vals else 0.0
#         data["score"] = round(_clamp01(data.get("score", meanv)), 3)
#         data["subscores"] = subs
#     else:
#         data["score"] = round(_clamp01(data.get("score", 0.0)), 3)
#     return data


# # -------------------- CLARITY --------------------
# # 정의: 명료성 자체(읽기 쉬움). usefulness와 분리된 독립 지표.
# # subscores: readability_structure, terminology_correctness, ambiguity_absence, conciseness_balance
# # score = mean(subscores)

# _SYS_CLARITY = """You are a style and clarity judge for CLARITY.
# Return strict JSON only:
# {
#  "subscores": {
#    "readability_structure": <0..1>,
#    "terminology_correctness": <0..1>,
#    "ambiguity_absence": <0..1>,
#    "conciseness_balance": <0..1>
#  },
#  "score": <0..1>
# }
# Rubric:
# - readability_structure: Organization, headings/lists, sentence flow.
# - terminology_correctness: Correct/consistent technical terms and notation.
# - ambiguity_absence: Minimal ambiguity; clear referents; avoids vague phrasing.
# - conciseness_balance: Concise yet sufficiently informative; avoids redundancy.
# Score = mean(subscores)."""

# _USR_CLARITY = """Answer:
# {ans}

# Evaluate clarity irrespective of usefulness.
# Fill all subscores in [0,1], set score=mean(subscores). Return JSON only."""


# def score_clarity(answer: str) -> Dict[str, Any]:
#     out = chat(
#         [
#             {"role": "system", "content": _SYS_CLARITY},
#             {"role": "user", "content": _USR_CLARITY.format(ans=answer)},
#         ]
#     )
#     data = _safe_json_parse(out)
#     subs = data.get("subscores", {})
#     if isinstance(subs, dict) and subs:
#         vals = []
#         for k in [
#             "readability_structure",
#             "terminology_correctness",
#             "ambiguity_absence",
#             "conciseness_balance",
#         ]:
#             vals.append(_clamp01(subs.get(k, 0.0)))
#             subs[k] = round(_clamp01(subs.get(k, 0.0)), 3)
#         meanv = sum(vals) / len(vals) if vals else 0.0
#         data["score"] = round(_clamp01(data.get("score", meanv)), 3)
#         data["subscores"] = subs
#     else:
#         data["score"] = round(_clamp01(data.get("score", 0.0)), 3)
#     return data


# # -------------------- RELEVANCE --------------------
# # 정의: 질문/문맥에 대한 관련성.
# # subscores: topical_alignment, focus_no_digressions, contextual_consistency
# # score = mean(subscores)

# _SYS_RELEVANCE = """You are a strict evaluation judge for RELEVANCE.
# Return strict JSON only:
# {
#  "subscores": {
#    "topical_alignment": <0..1>,
#    "focus_no_digressions": <0..1>,
#    "contextual_consistency": <0..1>
#  },
#  "score": <0..1>
# }
# Rubric:
# - topical_alignment: Addresses the core topic asked by the Question.
# - focus_no_digressions: Avoids off-topic tangents and needless additions.
# - contextual_consistency: If Contexts provided, stays consistent (no contradictions).
# Score = mean(subscores)."""

# _USR_RELEVANCE = """Question:
# {q}

# Contexts (optional):
# {ctx}

# Answer:
# {ans}

# Grade strictly per rubric. Fill all subscores in [0,1], set score=mean(subscores). Return JSON only."""


# def score_relevance(question: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
#     ctx = _join_ctx(contexts)
#     out = chat(
#         [
#             {"role": "system", "content": _SYS_RELEVANCE},
#             {
#                 "role": "user",
#                 "content": _USR_RELEVANCE.format(q=question, ctx=ctx, ans=answer),
#             },
#         ]
#     )
#     data = _safe_json_parse(out)
#     subs = data.get("subscores", {})
#     if isinstance(subs, dict) and subs:
#         vals = []
#         for k in [
#             "topical_alignment",
#             "focus_no_digressions",
#             "contextual_consistency",
#         ]:
#             vals.append(_clamp01(subs.get(k, 0.0)))
#             subs[k] = round(_clamp01(subs.get(k, 0.0)), 3)
#         meanv = sum(vals) / len(vals) if vals else 0.0
#         data["score"] = round(_clamp01(data.get("score", meanv)), 3)
#         data["subscores"] = subs
#     else:
#         data["score"] = round(_clamp01(data.get("score", 0.0)), 3)
#     return data


# # -------------------- ADDITIONAL VALUE --------------------
# # 정의: 직접 요구 밖에서 제공된 유익한 부가가치(예: 핵심 비교표, 예시, 실험 팁, 참고문헌 포인터 등)
# # subscores: insight_novelty, examples_illustration, comparisons_tradeoffs, references_pointers
# # score = mean(subscores) (단, 전부 0일 수도 있음)

# _SYS_ADDED = """You are an evaluation judge for ADDITIONAL VALUE.
# Return strict JSON only:
# {
#  "subscores": {
#    "insight_novelty": <0..1>,
#    "examples_illustration": <0..1>,
#    "comparisons_tradeoffs": <0..1>,
#    "references_pointers": <0..1>
#  },
#  "score": <0..1>
# }
# Rubric:
# - insight_novelty: Non-obvious insights beyond simply restating the answer.
# - examples_illustration: Useful examples or micro-cases that aid understanding.
# - comparisons_tradeoffs: Brief comparisons, pros/cons, or selection criteria.
# - references_pointers: Pointers to standards, datasets, docs, or reproducible steps.
# Score = mean(subscores). If no extra value beyond the direct answer, low scores are expected."""

# _USR_ADDED = """Question:
# {q}

# Answer:
# {ans}

# If there is meaningful additional value beyond the minimum needed to answer, reflect it in subscores.
# Fill [0,1] for each, set score=mean(subscores). Return JSON only."""


# def score_additional_value(question: str, answer: str) -> Dict[str, Any]:
#     out = chat(
#         [
#             {"role": "system", "content": _SYS_ADDED},
#             {"role": "user", "content": _USR_ADDED.format(q=question, ans=answer)},
#         ]
#     )
#     data = _safe_json_parse(out)
#     subs = data.get("subscores", {})
#     if isinstance(subs, dict) and subs:
#         vals = []
#         for k in [
#             "insight_novelty",
#             "examples_illustration",
#             "comparisons_tradeoffs",
#             "references_pointers",
#         ]:
#             vals.append(_clamp01(subs.get(k, 0.0)))
#             subs[k] = round(_clamp01(subs.get(k, 0.0)), 3)
#         meanv = sum(vals) / len(vals) if vals else 0.0
#         data["score"] = round(_clamp01(data.get("score", meanv)), 3)
#         data["subscores"] = subs
#     else:
#         data["score"] = round(_clamp01(data.get("score", 0.0)), 3)
#     return data


# # -------------------- ERROR PENALTY --------------------
# # 정의: 오류/모순/위험/비근거 주장 등으로 인한 감점 성격의 지표.
# # 해석: 1.0 = 오류 없음(감점 0), 0.0 = 심각한 오류(감점 극대). 점수는 "깨끗함" 정도.
# # subscores: factual_accuracy, contradiction_to_contexts, safety_and_risk, unsupported_inference
# # score = mean(subscores)

# _SYS_ERR = """You are an error and risk auditor for ERROR PENALTY (actually a 'cleanliness' score).
# Return strict JSON only:
# {
#  "subscores": {
#    "factual_accuracy": <0..1>,
#    "contradiction_to_contexts": <0..1>,
#    "safety_and_risk": <0..1>,
#    "unsupported_inference": <0..1>
#  },
#  "score": <0..1>
# }
# Interpretation:
# - factual_accuracy: Are claims correct?
# - contradiction_to_contexts: No contradictions with provided Contexts/GT.
# - safety_and_risk: Avoids unsafe/biased/inappropriate guidance.
# - unsupported_inference: Avoids making claims without evidence.
# Score = mean(subscores). 1.0 means no penalty (clean); 0.0 means severe issues."""

# _USR_ERR = """Question:
# {q}

# Ground Truth:
# {gt}

# Contexts (optional):
# {ctx}

# Answer:
# {ans}

# Audit the answer’s errors or risks considering GT and Contexts when available.
# Fill all subscores in [0,1], set score=mean(subscores). Return JSON only."""


# def score_error_penalty(
#     question: str, answer: str, ground_truth: str, contexts: List[str]
# ) -> Dict[str, Any]:
#     ctx = _join_ctx(contexts)
#     out = chat(
#         [
#             {"role": "system", "content": _SYS_ERR},
#             {
#                 "role": "user",
#                 "content": _USR_ERR.format(
#                     q=question, gt=ground_truth, ctx=ctx, ans=answer
#                 ),
#             },
#         ]
#     )
#     data = _safe_json_parse(out)
#     subs = data.get("subscores", {})
#     if isinstance(subs, dict) and subs:
#         vals = []
#         for k in [
#             "factual_accuracy",
#             "contradiction_to_contexts",
#             "safety_and_risk",
#             "unsupported_inference",
#         ]:
#             vals.append(_clamp01(subs.get(k, 0.0)))
#             subs[k] = round(_clamp01(subs.get(k, 0.0)), 3)
#         meanv = sum(vals) / len(vals) if vals else 0.0
#         data["score"] = round(_clamp01(data.get("score", meanv)), 3)
#         data["subscores"] = subs
#     else:
#         data["score"] = round(_clamp01(data.get("score", 0.0)), 3)
#     return data


# # -------------------- 일괄 실행 --------------------


# def run_llm_as_a_judge(
#     question: str, answer: str, contexts: List[str], ground_truth: str
# ) -> Dict[str, Any]:
#     """
#     반환 구조 예:
#     {
#       "completeness": {"score": 1.0, "requirements":[...], "covered":[...], "coverage": 1.0},
#       "usefulness":   {"score": 0.58, "subscores": {...}},
#       "clarity":      {"score": 0.91, "subscores": {...}},
#       "relevance":    {"score": 1.0,  "subscores": {...}},
#       "additional_value": {"score": 0.35, "subscores": {...}},
#       "error_penalty":    {"score": 0.92, "subscores": {...}}
#     }
#     """
#     comp = score_completeness(question, answer, ground_truth, contexts)
#     usef = score_usefulness(question, answer)
#     clar = score_clarity(answer)
#     rel = score_relevance(question, answer, contexts)
#     addv = score_additional_value(question, answer)
#     err = score_error_penalty(question, answer, ground_truth, contexts)

#     # 점수만 빠르게 보고 싶을 때를 위해 상위 레벨에도 요약 제공
#     summary = {
#         "completeness": round(_clamp01(comp.get("score", 0.0)), 3),
#         "usefulness": round(_clamp01(usef.get("score", 0.0)), 3),
#         "clarity": round(_clamp01(clar.get("score", 0.0)), 3),
#         "relevance": round(_clamp01(rel.get("score", 0.0)), 3),
#         "additional_value": round(_clamp01(addv.get("score", 0.0)), 3),
#         "error_penalty": round(_clamp01(err.get("score", 0.0)), 3),
#     }

#     return {
#         **summary,
#         "details": {
#             "completeness": comp,
#             "usefulness": usef,
#             "clarity": clar,
#             "relevance": rel,
#             "additional_value": addv,
#             "error_penalty": err,
#         },
#     }


# # # llm-as-a-judge.py
# # # 목적: LLM-as-a-Judge 평가 (completeness, usefulness, relevance) 0~1 점수
# # # 의존: 같은 폴더의 llm.py (chat 함수), config.OPENAI_MODEL(간접)

# # from typing import List, Dict
# # import json, re
# # from llm import chat


# # # --------- 공통 유틸 ----------
# # def _clamp01(x: float) -> float:
# #     try:
# #         v = float(x)
# #     except Exception:
# #         return 0.0
# #     if v < 0.0:
# #         return 0.0
# #     if v > 1.0:
# #         return 1.0
# #     return v


# # _NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


# # def _parse_score(out: str) -> float:
# #     """
# #     우선 JSON {"score": <float>} 파싱, 실패 시 문자열 내 최초 float 추출.
# #     """
# #     if not out:
# #         return 0.0
# #     try:
# #         data = json.loads(out)
# #         if isinstance(data, dict) and "score" in data:
# #             return _clamp01(data["score"])
# #     except Exception:
# #         pass
# #     m = _NUM_RE.search(out)
# #     if m:
# #         return _clamp01(m.group())
# #     return 0.0


# # def _join_ctx(contexts: List[str], max_ctx: int = 5, max_chars: int = 4000) -> str:
# #     # 상한을 두어 판정 LLM의 토큰 초과 방지
# #     snippets, acc = [], 0
# #     for c in contexts[:max_ctx]:
# #         c = c.strip()
# #         if not c:
# #             continue
# #         if acc + len(c) + 8 > max_chars:
# #             remain = max_chars - acc - 8
# #             if remain > 0:
# #                 snippets.append(c[:remain])
# #                 acc += remain
# #             break
# #         snippets.append(c)
# #         acc += len(c) + 8
# #     return "\n\n---\n\n".join(snippets)


# # # --------- Completeness ----------
# # # 정의: 질문/GT/문맥이 요구하는 핵심 “요건(atomic requirements)”을
# # # 답변이 얼마나 빠짐없이 충족했는지의 비율.
# # # 채점 방식: LLM이 요건 리스트를 추출한 뒤, 각 요건을 답이 충족했는지 체크 → (#충족 / #요건)

# # _SYS_COMPLETENESS = """You are a meticulous evaluation judge.
# # Your job is to grade COMPLETENESS on a 0.0~1.0 scale as a fraction of covered atomic requirements.
# # Be strict but fair.
# # You MUST output strict JSON: {"score": <float in [0,1]>}
# # NO prose, NO trailing text."""

# # _USR_COMPLETENESS = """Task: Grade completeness of the Answer.

# # Question:
# # {q}

# # Ground Truth (may be short or list-like):
# # {gt}

# # Contexts (optional; may help recover requirements):
# # {ctx}

# # Answer:
# # {ans}

# # Instructions:
# # 1) Extract a minimal list of atomic requirements needed to fully answer the Question, using the Ground Truth and optionally Contexts if GT is short/elliptical.
# # 2) For each requirement, check whether the Answer explicitly satisfies it (exactly or via clear paraphrase).
# # 3) Compute coverage_ratio = (# satisfied requirements) / (total requirements).
# # 4) Return JSON only: {{"score": coverage_ratio}}"""


# # def score_completeness(
# #     question: str, answer: str, ground_truth: str, contexts: List[str]
# # ) -> float:
# #     ctx_joined = _join_ctx(contexts)
# #     out = chat(
# #         [
# #             {"role": "system", "content": _SYS_COMPLETENESS},
# #             {
# #                 "role": "user",
# #                 "content": _USR_COMPLETENESS.format(
# #                     q=question, gt=ground_truth, ctx=ctx_joined, ans=answer
# #                 ),
# #             },
# #         ]
# #     )
# #     return _parse_score(out)


# # # --------- Usefulness ----------
# # # 정의: 사용자가 실제로 “쓴다”고 했을 때의 실용성(명료/구체/행동가능/주의점/범위 명시).
# # # 루브릭: 0=무용, 0.25=모호, 0.5=부분적 도움, 0.75=대체로 유용(구체+제약/한계 일부),
# # # 1.0=매우 유용(핵심 직접답+필요 맥락/조건/한계 명시, 과도한 헛소리 없음)

# # _SYS_USEFULNESS = """You are a pragmatic evaluation judge.
# # Your job is to grade USEFULNESS (practical helpfulness for the user) on a 0.0~1.0 scale.
# # Output strict JSON: {"score": <float in [0,1]>}
# # Consider clarity, specificity, actionability, stated assumptions/limits, and absence of fluff/hallucination."""

# # _USR_USEFULNESS = """Task: Grade usefulness of the Answer.

# # Question:
# # {q}

# # Answer:
# # {ans}

# # Rubric (choose a value then optionally interpolate):
# # - 0.0: Useless or wrong or non-answer.
# # - 0.25: Vague, generic, or lacks specifics; low practical value.
# # - 0.50: Some useful info; partially actionable but missing key specifics or caveats.
# # - 0.75: Largely useful; clear, specific, and actionable with minor gaps or caveats.
# # - 1.00: Highly useful; directly answers core need with precise details, constraints, or next steps, minimal fluff.

# # Return JSON only: {{"score": <number>}}"""


# # def score_usefulness(question: str, answer: str, contexts: List[str]) -> float:
# #     # contexts는 참고용(여기선 사용 안 해도 됨); 필요 시 q/ans만 판단
# #     out = chat(
# #         [
# #             {"role": "system", "content": _SYS_USEFULNESS},
# #             {"role": "user", "content": _USR_USEFULNESS.format(q=question, ans=answer)},
# #         ]
# #     )
# #     return _parse_score(out)


# # # --------- Relevance ----------
# # # 정의: 답이 질문에 얼마나 온전히 초점을 맞추는지(주제일치/탈선없음/불필요한 추가 최소화/문맥일치).
# # # 루브릭: 0=무관, 0.25=주제 벗어남 多, 0.5=부분관련, 0.75=대체로 관련, 1.0=완전 관련.

# # _SYS_RELEVANCE = """You are a strict evaluation judge.
# # Your job is to grade RELEVANCE to the Question on a 0.0~1.0 scale.
# # Output strict JSON: {"score": <float in [0,1]>}
# # Judge topical alignment, focus, and absence of off-topic content; optionally leverage Contexts to resolve ambiguity."""

# # _USR_RELEVANCE = """Task: Grade relevance of the Answer to the Question.

# # Question:
# # {q}

# # Contexts (optional disambiguation):
# # {ctx}

# # Answer:
# # {ans}

# # Rubric:
# # - 0.0: Irrelevant or mostly off-topic.
# # - 0.25: Weakly related; substantial digressions.
# # - 0.50: Partially addresses the question, with noticeable off-topic or missing focus.
# # - 0.75: Mostly on-topic; minor tangents or missing pieces.
# # - 1.00: Directly on-topic and focused; no needless digressions.

# # Return JSON only: {{"score": <number>}}"""


# # def score_relevance(question: str, answer: str, contexts: List[str]) -> float:
# #     ctx_joined = _join_ctx(contexts)
# #     out = chat(
# #         [
# #             {"role": "system", "content": _SYS_RELEVANCE},
# #             {
# #                 "role": "user",
# #                 "content": _USR_RELEVANCE.format(
# #                     q=question, ctx=ctx_joined, ans=answer
# #                 ),
# #             },
# #         ]
# #     )
# #     return _parse_score(out)


# # # --------- 일괄 호출 ----------
# # def run_llm_as_a_judge(
# #     question: str, answer: str, contexts: List[str], ground_truth: str
# # ) -> Dict[str, float]:
# #     """
# #     반환 예:
# #     {
# #       'completeness': 0.875,
# #       'usefulness': 0.750,
# #       'relevance': 1.000
# #     }
# #     """
# #     comp = score_completeness(question, answer, ground_truth, contexts)
# #     usef = score_usefulness(question, answer, contexts)
# #     rel = score_relevance(question, answer, contexts)
# #     # 소수점 3자리 스냅(표시 목적; 내부는 float)
# #     return {
# #         "completeness": round(_clamp01(comp), 3),
# #         "usefulness": round(_clamp01(usef), 3),
# #         "relevance": round(_clamp01(rel), 3),
# #     }
