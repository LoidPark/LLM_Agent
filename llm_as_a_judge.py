# llm_as_a_judge.py
# 목적: LLM-as-a-Judge 평가 (6개 지표)
# - completeness (requirements 기반, coverage%)
# - usefulness (실용성; subscores 포함)
# - clarity (명료성; subscores 포함)
# - relevance (질문/문맥 적합성; subscores 포함)
# - additional_value (부가가치; subscores 포함)
# - error_penalty (오류/모순/안전 문제; subscores 포함, 감점 성격)
#
# 출력 스케일: 모든 score는 0.0 ~ 1.0
# 의존: llm.py의 chat(), config.OPENAI_MODEL 간접 사용
# 사용: from llm_as_a_judge import run_llm_as_a_judge

from typing import List, Dict, Any
import json, re
from llm import chat

# -------------------- 공통 유틸 --------------------

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _safe_json_parse(out: str) -> Dict[str, Any]:
    """
    가능한 한 JSON으로 파싱.
    실패 시 {"score": float} 정도라도 복구.
    """
    if not out:
        return {"score": 0.0}
    try:
        data = json.loads(out)
        # 점수만 있어도 통과
        if isinstance(data, dict):
            if "score" in data:
                data["score"] = _clamp01(data["score"])
            return data
    except Exception:
        pass
    # 숫자만 추출해서 score 로 반환
    m = _NUM_RE.search(out)
    if m:
        return {"score": _clamp01(float(m.group()))}
    return {"score": 0.0}


def _join_ctx(contexts: List[str], max_ctx: int = 5, max_chars: int = 4000) -> str:
    # 판정 LLM 토큰 초과 방지를 위한 컨텍스트 합치기
    snippets, acc = [], 0
    for c in contexts[:max_ctx]:
        c = (c or "").strip()
        if not c:
            continue
        if acc + len(c) + 8 > max_chars:
            remain = max_chars - acc - 8
            if remain > 0:
                snippets.append(c[:remain])
                acc += remain
            break
        snippets.append(c)
        acc += len(c) + 8
    return "\n\n---\n\n".join(snippets)


# -------------------- COMPLETENESS --------------------
# 정의: 질문/GT가 요구하는 원자적 요구사항을 Answer가 얼마나 충족했는가 (coverage).
# 출력: requirements[], covered[], coverage, score

_SYS_COMPLETENESS = """You are a meticulous evaluation judge.
Grade COMPLETENESS on [0.0,1.0] as the fraction of covered atomic requirements.
Output strict JSON only:
{"requirements":[<strings>],
 "covered":[<booleans aligned with requirements>],
 "coverage": <0..1>,
 "score": <0..1>}
Rules:
- Extract minimal, non-overlapping atomic requirements to fully answer the Question.
- Use Ground Truth as authoritative; use Contexts only to disambiguate if GT is terse.
- Count covered=true only if the Answer explicitly states or clearly paraphrases the requirement.
- No credit for implied assumptions not present in the Answer.
- coverage = (#covered / #requirements); score = coverage."""

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
2) Mark covered true/false for each requirement based on the Answer.
3) Compute coverage=(#covered/#requirements) and set score=coverage.
Return JSON only."""


def score_completeness(
    question: str, answer: str, ground_truth: str, contexts: List[str]
) -> Dict[str, Any]:
    ctx = _join_ctx(contexts)
    out = chat(
        [
            {"role": "system", "content": _SYS_COMPLETENESS},
            {
                "role": "user",
                "content": _USR_COMPLETENESS.format(
                    q=question, gt=ground_truth, ctx=ctx, ans=answer
                ),
            },
        ]
    )
    data = _safe_json_parse(out)
    # 방어
    reqs = data.get("requirements", [])
    covs = data.get("covered", [])
    if (
        isinstance(reqs, list)
        and isinstance(covs, list)
        and len(reqs) == len(covs)
        and len(reqs) > 0
    ):
        covered_cnt = sum(1 for x in covs if bool(x))
        coverage = covered_cnt / len(reqs)
        data["coverage"] = round(_clamp01(coverage), 3)
        data["score"] = round(_clamp01(data.get("score", coverage)), 3)
    else:
        data["coverage"] = round(
            _clamp01(data.get("coverage", data.get("score", 0.0))), 3
        )
        data["score"] = round(_clamp01(data.get("score", 0.0)), 3)
    return data


# -------------------- USEFULNESS --------------------
# 정의: 실무적으로 "쓸모"가 있는가. 상세 루브릭을 subscores 로 제공.
# subscores: clarity_for_use(이 지표에서의 실무적 명료성), specificity, actionability,
#            constraints_caveats, guidance_next_steps, nonfluff
# score = mean(subscores)

_SYS_USEFULNESS = """You are a pragmatic evaluation judge for USEFULNESS.
Return strict JSON only:
{
 "subscores": {
   "clarity_for_use": <0..1>,
   "specificity": <0..1>,
   "actionability": <0..1>,
   "constraints_caveats": <0..1>,
   "guidance_next_steps": <0..1>,
   "nonfluff": <0..1>
 },
 "score": <0..1>
}
Rubric:
- clarity_for_use: Language is unambiguous and task-oriented.
- specificity: Concrete details/examples/criteria (not just labels).
- actionability: Provides steps, decision rules, or how-to guidance.
- constraints_caveats: States assumptions, limitations, risks, or scope.
- guidance_next_steps: Advises what to do next (e.g., baseline setup, evaluation plan).
- nonfluff: Penalize filler/hallucination; reward concise substance.
Score = mean(subscores)."""

_USR_USEFULNESS = """Question:
{q}

Answer:
{ans}

Evaluate practical helpfulness for a practitioner who must *use* the answer.
Fill all subscores in [0,1], set score=mean(subscores). Return JSON only."""


def score_usefulness(question: str, answer: str) -> Dict[str, Any]:
    out = chat(
        [
            {"role": "system", "content": _SYS_USEFULNESS},
            {"role": "user", "content": _USR_USEFULNESS.format(q=question, ans=answer)},
        ]
    )
    data = _safe_json_parse(out)
    subs = data.get("subscores", {})
    if isinstance(subs, dict) and subs:
        vals = []
        for k in [
            "clarity_for_use",
            "specificity",
            "actionability",
            "constraints_caveats",
            "guidance_next_steps",
            "nonfluff",
        ]:
            vals.append(_clamp01(subs.get(k, 0.0)))
            subs[k] = round(_clamp01(subs.get(k, 0.0)), 3)
        meanv = sum(vals) / len(vals) if vals else 0.0
        data["score"] = round(_clamp01(data.get("score", meanv)), 3)
        data["subscores"] = subs
    else:
        data["score"] = round(_clamp01(data.get("score", 0.0)), 3)
    return data


# -------------------- CLARITY --------------------
# 정의: 명료성 자체(읽기 쉬움). usefulness와 분리된 독립 지표.
# subscores: readability_structure, terminology_correctness, ambiguity_absence, conciseness_balance
# score = mean(subscores)

_SYS_CLARITY = """You are a style and clarity judge for CLARITY.
Return strict JSON only:
{
 "subscores": {
   "readability_structure": <0..1>,
   "terminology_correctness": <0..1>,
   "ambiguity_absence": <0..1>,
   "conciseness_balance": <0..1>
 },
 "score": <0..1>
}
Rubric:
- readability_structure: Organization, headings/lists, sentence flow.
- terminology_correctness: Correct/consistent technical terms and notation.
- ambiguity_absence: Minimal ambiguity; clear referents; avoids vague phrasing.
- conciseness_balance: Concise yet sufficiently informative; avoids redundancy.
Score = mean(subscores)."""

_USR_CLARITY = """Answer:
{ans}

Evaluate clarity irrespective of usefulness.
Fill all subscores in [0,1], set score=mean(subscores). Return JSON only."""


def score_clarity(answer: str) -> Dict[str, Any]:
    out = chat(
        [
            {"role": "system", "content": _SYS_CLARITY},
            {"role": "user", "content": _USR_CLARITY.format(ans=answer)},
        ]
    )
    data = _safe_json_parse(out)
    subs = data.get("subscores", {})
    if isinstance(subs, dict) and subs:
        vals = []
        for k in [
            "readability_structure",
            "terminology_correctness",
            "ambiguity_absence",
            "conciseness_balance",
        ]:
            vals.append(_clamp01(subs.get(k, 0.0)))
            subs[k] = round(_clamp01(subs.get(k, 0.0)), 3)
        meanv = sum(vals) / len(vals) if vals else 0.0
        data["score"] = round(_clamp01(data.get("score", meanv)), 3)
        data["subscores"] = subs
    else:
        data["score"] = round(_clamp01(data.get("score", 0.0)), 3)
    return data


# -------------------- RELEVANCE --------------------
# 정의: 질문/문맥에 대한 관련성.
# subscores: topical_alignment, focus_no_digressions, contextual_consistency
# score = mean(subscores)

_SYS_RELEVANCE = """You are a strict evaluation judge for RELEVANCE.
Return strict JSON only:
{
 "subscores": {
   "topical_alignment": <0..1>,
   "focus_no_digressions": <0..1>,
   "contextual_consistency": <0..1>
 },
 "score": <0..1>
}
Rubric:
- topical_alignment: Addresses the core topic asked by the Question.
- focus_no_digressions: Avoids off-topic tangents and needless additions.
- contextual_consistency: If Contexts provided, stays consistent (no contradictions).
Score = mean(subscores)."""

_USR_RELEVANCE = """Question:
{q}

Contexts (optional):
{ctx}

Answer:
{ans}

Grade strictly per rubric. Fill all subscores in [0,1], set score=mean(subscores). Return JSON only."""


def score_relevance(question: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
    ctx = _join_ctx(contexts)
    out = chat(
        [
            {"role": "system", "content": _SYS_RELEVANCE},
            {
                "role": "user",
                "content": _USR_RELEVANCE.format(q=question, ctx=ctx, ans=answer),
            },
        ]
    )
    data = _safe_json_parse(out)
    subs = data.get("subscores", {})
    if isinstance(subs, dict) and subs:
        vals = []
        for k in [
            "topical_alignment",
            "focus_no_digressions",
            "contextual_consistency",
        ]:
            vals.append(_clamp01(subs.get(k, 0.0)))
            subs[k] = round(_clamp01(subs.get(k, 0.0)), 3)
        meanv = sum(vals) / len(vals) if vals else 0.0
        data["score"] = round(_clamp01(data.get("score", meanv)), 3)
        data["subscores"] = subs
    else:
        data["score"] = round(_clamp01(data.get("score", 0.0)), 3)
    return data


# -------------------- ADDITIONAL VALUE --------------------
# 정의: 직접 요구 밖에서 제공된 유익한 부가가치(예: 핵심 비교표, 예시, 실험 팁, 참고문헌 포인터 등)
# subscores: insight_novelty, examples_illustration, comparisons_tradeoffs, references_pointers
# score = mean(subscores) (단, 전부 0일 수도 있음)

_SYS_ADDED = """You are an evaluation judge for ADDITIONAL VALUE.
Return strict JSON only:
{
 "subscores": {
   "insight_novelty": <0..1>,
   "examples_illustration": <0..1>,
   "comparisons_tradeoffs": <0..1>,
   "references_pointers": <0..1>
 },
 "score": <0..1>
}
Rubric:
- insight_novelty: Non-obvious insights beyond simply restating the answer.
- examples_illustration: Useful examples or micro-cases that aid understanding.
- comparisons_tradeoffs: Brief comparisons, pros/cons, or selection criteria.
- references_pointers: Pointers to standards, datasets, docs, or reproducible steps.
Score = mean(subscores). If no extra value beyond the direct answer, low scores are expected."""

_USR_ADDED = """Question:
{q}

Answer:
{ans}

If there is meaningful additional value beyond the minimum needed to answer, reflect it in subscores.
Fill [0,1] for each, set score=mean(subscores). Return JSON only."""


def score_additional_value(question: str, answer: str) -> Dict[str, Any]:
    out = chat(
        [
            {"role": "system", "content": _SYS_ADDED},
            {"role": "user", "content": _USR_ADDED.format(q=question, ans=answer)},
        ]
    )
    data = _safe_json_parse(out)
    subs = data.get("subscores", {})
    if isinstance(subs, dict) and subs:
        vals = []
        for k in [
            "insight_novelty",
            "examples_illustration",
            "comparisons_tradeoffs",
            "references_pointers",
        ]:
            vals.append(_clamp01(subs.get(k, 0.0)))
            subs[k] = round(_clamp01(subs.get(k, 0.0)), 3)
        meanv = sum(vals) / len(vals) if vals else 0.0
        data["score"] = round(_clamp01(data.get("score", meanv)), 3)
        data["subscores"] = subs
    else:
        data["score"] = round(_clamp01(data.get("score", 0.0)), 3)
    return data


# -------------------- ERROR PENALTY --------------------
# 정의: 오류/모순/위험/비근거 주장 등으로 인한 감점 성격의 지표.
# 해석: 1.0 = 오류 없음(감점 0), 0.0 = 심각한 오류(감점 극대). 점수는 "깨끗함" 정도.
# subscores: factual_accuracy, contradiction_to_contexts, safety_and_risk, unsupported_inference
# score = mean(subscores)

_SYS_ERR = """You are an error and risk auditor for ERROR PENALTY (actually a 'cleanliness' score).
Return strict JSON only:
{
 "subscores": {
   "factual_accuracy": <0..1>,
   "contradiction_to_contexts": <0..1>,
   "safety_and_risk": <0..1>,
   "unsupported_inference": <0..1>
 },
 "score": <0..1>
}
Interpretation:
- factual_accuracy: Are claims correct?
- contradiction_to_contexts: No contradictions with provided Contexts/GT.
- safety_and_risk: Avoids unsafe/biased/inappropriate guidance.
- unsupported_inference: Avoids making claims without evidence.
Score = mean(subscores). 1.0 means no penalty (clean); 0.0 means severe issues."""

_USR_ERR = """Question:
{q}

Ground Truth:
{gt}

Contexts (optional):
{ctx}

Answer:
{ans}

Audit the answer’s errors or risks considering GT and Contexts when available.
Fill all subscores in [0,1], set score=mean(subscores). Return JSON only."""


def score_error_penalty(
    question: str, answer: str, ground_truth: str, contexts: List[str]
) -> Dict[str, Any]:
    ctx = _join_ctx(contexts)
    out = chat(
        [
            {"role": "system", "content": _SYS_ERR},
            {
                "role": "user",
                "content": _USR_ERR.format(
                    q=question, gt=ground_truth, ctx=ctx, ans=answer
                ),
            },
        ]
    )
    data = _safe_json_parse(out)
    subs = data.get("subscores", {})
    if isinstance(subs, dict) and subs:
        vals = []
        for k in [
            "factual_accuracy",
            "contradiction_to_contexts",
            "safety_and_risk",
            "unsupported_inference",
        ]:
            vals.append(_clamp01(subs.get(k, 0.0)))
            subs[k] = round(_clamp01(subs.get(k, 0.0)), 3)
        meanv = sum(vals) / len(vals) if vals else 0.0
        data["score"] = round(_clamp01(data.get("score", meanv)), 3)
        data["subscores"] = subs
    else:
        data["score"] = round(_clamp01(data.get("score", 0.0)), 3)
    return data


# -------------------- 일괄 실행 --------------------


def run_llm_as_a_judge(
    question: str, answer: str, contexts: List[str], ground_truth: str
) -> Dict[str, Any]:
    """
    반환 구조 예:
    {
      "completeness": {"score": 1.0, "requirements":[...], "covered":[...], "coverage": 1.0},
      "usefulness":   {"score": 0.58, "subscores": {...}},
      "clarity":      {"score": 0.91, "subscores": {...}},
      "relevance":    {"score": 1.0,  "subscores": {...}},
      "additional_value": {"score": 0.35, "subscores": {...}},
      "error_penalty":    {"score": 0.92, "subscores": {...}}
    }
    """
    comp = score_completeness(question, answer, ground_truth, contexts)
    usef = score_usefulness(question, answer)
    clar = score_clarity(answer)
    rel = score_relevance(question, answer, contexts)
    addv = score_additional_value(question, answer)
    err = score_error_penalty(question, answer, ground_truth, contexts)

    # 점수만 빠르게 보고 싶을 때를 위해 상위 레벨에도 요약 제공
    summary = {
        "completeness": round(_clamp01(comp.get("score", 0.0)), 3),
        "usefulness": round(_clamp01(usef.get("score", 0.0)), 3),
        "clarity": round(_clamp01(clar.get("score", 0.0)), 3),
        "relevance": round(_clamp01(rel.get("score", 0.0)), 3),
        "additional_value": round(_clamp01(addv.get("score", 0.0)), 3),
        "error_penalty": round(_clamp01(err.get("score", 0.0)), 3),
    }

    return {
        **summary,
        "details": {
            "completeness": comp,
            "usefulness": usef,
            "clarity": clar,
            "relevance": rel,
            "additional_value": addv,
            "error_penalty": err,
        },
    }


# # llm-as-a-judge.py
# # 목적: LLM-as-a-Judge 평가 (completeness, usefulness, relevance) 0~1 점수
# # 의존: 같은 폴더의 llm.py (chat 함수), config.OPENAI_MODEL(간접)

# from typing import List, Dict
# import json, re
# from llm import chat


# # --------- 공통 유틸 ----------
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


# _NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


# def _parse_score(out: str) -> float:
#     """
#     우선 JSON {"score": <float>} 파싱, 실패 시 문자열 내 최초 float 추출.
#     """
#     if not out:
#         return 0.0
#     try:
#         data = json.loads(out)
#         if isinstance(data, dict) and "score" in data:
#             return _clamp01(data["score"])
#     except Exception:
#         pass
#     m = _NUM_RE.search(out)
#     if m:
#         return _clamp01(m.group())
#     return 0.0


# def _join_ctx(contexts: List[str], max_ctx: int = 5, max_chars: int = 4000) -> str:
#     # 상한을 두어 판정 LLM의 토큰 초과 방지
#     snippets, acc = [], 0
#     for c in contexts[:max_ctx]:
#         c = c.strip()
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


# # --------- Completeness ----------
# # 정의: 질문/GT/문맥이 요구하는 핵심 “요건(atomic requirements)”을
# # 답변이 얼마나 빠짐없이 충족했는지의 비율.
# # 채점 방식: LLM이 요건 리스트를 추출한 뒤, 각 요건을 답이 충족했는지 체크 → (#충족 / #요건)

# _SYS_COMPLETENESS = """You are a meticulous evaluation judge.
# Your job is to grade COMPLETENESS on a 0.0~1.0 scale as a fraction of covered atomic requirements.
# Be strict but fair.
# You MUST output strict JSON: {"score": <float in [0,1]>}
# NO prose, NO trailing text."""

# _USR_COMPLETENESS = """Task: Grade completeness of the Answer.

# Question:
# {q}

# Ground Truth (may be short or list-like):
# {gt}

# Contexts (optional; may help recover requirements):
# {ctx}

# Answer:
# {ans}

# Instructions:
# 1) Extract a minimal list of atomic requirements needed to fully answer the Question, using the Ground Truth and optionally Contexts if GT is short/elliptical.
# 2) For each requirement, check whether the Answer explicitly satisfies it (exactly or via clear paraphrase).
# 3) Compute coverage_ratio = (# satisfied requirements) / (total requirements).
# 4) Return JSON only: {{"score": coverage_ratio}}"""


# def score_completeness(
#     question: str, answer: str, ground_truth: str, contexts: List[str]
# ) -> float:
#     ctx_joined = _join_ctx(contexts)
#     out = chat(
#         [
#             {"role": "system", "content": _SYS_COMPLETENESS},
#             {
#                 "role": "user",
#                 "content": _USR_COMPLETENESS.format(
#                     q=question, gt=ground_truth, ctx=ctx_joined, ans=answer
#                 ),
#             },
#         ]
#     )
#     return _parse_score(out)


# # --------- Usefulness ----------
# # 정의: 사용자가 실제로 “쓴다”고 했을 때의 실용성(명료/구체/행동가능/주의점/범위 명시).
# # 루브릭: 0=무용, 0.25=모호, 0.5=부분적 도움, 0.75=대체로 유용(구체+제약/한계 일부),
# # 1.0=매우 유용(핵심 직접답+필요 맥락/조건/한계 명시, 과도한 헛소리 없음)

# _SYS_USEFULNESS = """You are a pragmatic evaluation judge.
# Your job is to grade USEFULNESS (practical helpfulness for the user) on a 0.0~1.0 scale.
# Output strict JSON: {"score": <float in [0,1]>}
# Consider clarity, specificity, actionability, stated assumptions/limits, and absence of fluff/hallucination."""

# _USR_USEFULNESS = """Task: Grade usefulness of the Answer.

# Question:
# {q}

# Answer:
# {ans}

# Rubric (choose a value then optionally interpolate):
# - 0.0: Useless or wrong or non-answer.
# - 0.25: Vague, generic, or lacks specifics; low practical value.
# - 0.50: Some useful info; partially actionable but missing key specifics or caveats.
# - 0.75: Largely useful; clear, specific, and actionable with minor gaps or caveats.
# - 1.00: Highly useful; directly answers core need with precise details, constraints, or next steps, minimal fluff.

# Return JSON only: {{"score": <number>}}"""


# def score_usefulness(question: str, answer: str, contexts: List[str]) -> float:
#     # contexts는 참고용(여기선 사용 안 해도 됨); 필요 시 q/ans만 판단
#     out = chat(
#         [
#             {"role": "system", "content": _SYS_USEFULNESS},
#             {"role": "user", "content": _USR_USEFULNESS.format(q=question, ans=answer)},
#         ]
#     )
#     return _parse_score(out)


# # --------- Relevance ----------
# # 정의: 답이 질문에 얼마나 온전히 초점을 맞추는지(주제일치/탈선없음/불필요한 추가 최소화/문맥일치).
# # 루브릭: 0=무관, 0.25=주제 벗어남 多, 0.5=부분관련, 0.75=대체로 관련, 1.0=완전 관련.

# _SYS_RELEVANCE = """You are a strict evaluation judge.
# Your job is to grade RELEVANCE to the Question on a 0.0~1.0 scale.
# Output strict JSON: {"score": <float in [0,1]>}
# Judge topical alignment, focus, and absence of off-topic content; optionally leverage Contexts to resolve ambiguity."""

# _USR_RELEVANCE = """Task: Grade relevance of the Answer to the Question.

# Question:
# {q}

# Contexts (optional disambiguation):
# {ctx}

# Answer:
# {ans}

# Rubric:
# - 0.0: Irrelevant or mostly off-topic.
# - 0.25: Weakly related; substantial digressions.
# - 0.50: Partially addresses the question, with noticeable off-topic or missing focus.
# - 0.75: Mostly on-topic; minor tangents or missing pieces.
# - 1.00: Directly on-topic and focused; no needless digressions.

# Return JSON only: {{"score": <number>}}"""


# def score_relevance(question: str, answer: str, contexts: List[str]) -> float:
#     ctx_joined = _join_ctx(contexts)
#     out = chat(
#         [
#             {"role": "system", "content": _SYS_RELEVANCE},
#             {
#                 "role": "user",
#                 "content": _USR_RELEVANCE.format(
#                     q=question, ctx=ctx_joined, ans=answer
#                 ),
#             },
#         ]
#     )
#     return _parse_score(out)


# # --------- 일괄 호출 ----------
# def run_llm_as_a_judge(
#     question: str, answer: str, contexts: List[str], ground_truth: str
# ) -> Dict[str, float]:
#     """
#     반환 예:
#     {
#       'completeness': 0.875,
#       'usefulness': 0.750,
#       'relevance': 1.000
#     }
#     """
#     comp = score_completeness(question, answer, ground_truth, contexts)
#     usef = score_usefulness(question, answer, contexts)
#     rel = score_relevance(question, answer, contexts)
#     # 소수점 3자리 스냅(표시 목적; 내부는 float)
#     return {
#         "completeness": round(_clamp01(comp), 3),
#         "usefulness": round(_clamp01(usef), 3),
#         "relevance": round(_clamp01(rel), 3),
#     }
