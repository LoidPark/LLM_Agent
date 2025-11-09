# judge.py
import json
from typing import List, Dict, Tuple
from llm import chat

SYS_FACTS = """You are an information extraction assistant.
Given a question and a ground-truth answer, you must decompose the answer into a minimal list of atomic facts needed to answer the question correctly.
Return a strict JSON array of strings called 'facts'. No commentary.
Example output:
{"facts": ["Fact A", "Fact B"]}"""

USR_FACTS_TMPL = """Question: {q}
Ground truth answer: {gt}
Extract minimal atomic facts required to answer the question correctly.
Return JSON like: {{"facts": ["...", "..."]}}"""

SYS_REL = """You are a strict relevance judge.
Decide if the provided CONTEXT contains information that SUPPORTS AT LEAST ONE of the required facts for answering the question correctly.
Return only 'true' or 'false' (lowercase), no explanation."""

USR_REL_TMPL = """Question: {q}
Required facts: {facts}
Context:
\"\"\"{ctx}\"\"\""""


def extract_needed_facts(question: str, ground_truth: str | None = None) -> List[str]:
    msg = [
        {"role": "system", "content": SYS_FACTS},
        {"role": "user", "content": USR_FACTS_TMPL.format(q=question, gt=ground_truth)},
    ]
    out = chat(msg)
    print("[DEBUG][extract_needed_facts] raw LLM output:", out)
    # 관대한 파서: JSON만 뽑기
    try:
        data = json.loads(out)
        facts = [f.strip() for f in data.get("facts", []) if isinstance(f, str)]
    except Exception:
        # fallback: 한 줄씩
        facts = [s.strip("-• ").strip() for s in out.split("\n") if s.strip()]
    return [f for f in facts if f]


def judge_relevance(question: str, facts: List[str], context: str) -> bool:
    msg = [
        {"role": "system", "content": SYS_REL},
        {
            "role": "user",
            "content": USR_REL_TMPL.format(q=question, facts=facts, ctx=context),
        },
    ]
    out = chat(msg).lower()
    print("[DEBUG][judge_relevance] raw LLM output:", out)
    return "true" in out and "false" not in out


import re


def _norm(s: str) -> str:
    # 소문자화 + 영숫자만 남김 (DT-CNN → dtcnn)
    return re.sub(r"[^0-9a-z]+", "", s.lower())


def map_covered_facts(question: str, facts: List[str], context: str) -> List[int]:
    covered = []
    ctx_n = _norm(context)
    for i, f in enumerate(facts):
        if not f:
            continue
        if _norm(f) in ctx_n:
            covered.append(i)
    return covered


# judge.py (맨 아래에 추가)
import json


def score_faithfulness(answer: str, contexts: list[str]) -> float:
    ctx = "\n\n---\n\n".join(contexts[:5])
    sys = 'You are a strict fact-checker. Rate how well the Answer is supported by the Contexts (0=not supported, 1=fully supported). Return JSON: {"score": float between 0 and 1}. No extra text.'
    usr = f"Contexts:\n{ctx}\n\nAnswer:\n{answer}\n\nReturn only JSON."
    out = chat([{"role": "system", "content": sys}, {"role": "user", "content": usr}])
    try:
        return float(json.loads(out).get("score", 0.0))
    except Exception:
        # 관대한 파싱 폴백
        try:
            return float(out.strip())
        except Exception:
            return 0.0


def score_answer_relevancy(question: str, answer: str) -> float:
    sys = 'You are a relevance grader. Rate how directly the Answer addresses the Question (0=irrelevant, 1=fully relevant). Return JSON: {"score": float between 0 and 1}. No extra text.'
    usr = f"Question:\n{question}\n\nAnswer:\n{answer}\n\nReturn only JSON."
    out = chat([{"role": "system", "content": sys}, {"role": "user", "content": usr}])
    try:
        return float(json.loads(out).get("score", 0.0))
    except Exception:
        try:
            return float(out.strip())
        except Exception:
            return 0.0
