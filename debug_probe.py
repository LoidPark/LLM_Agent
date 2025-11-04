# debug_probe.py
from typing import List
from llm import chat

REL_SYS = "Answer strictly 'true' or 'false'. No extra text."
REL_USR = """Question:
{q}

Context:
\"\"\"{ctx}\"\"\"

Does this context help answer the question?"""

SUPP_SYS = "Answer strictly 'true' or 'false'. No extra text."
SUPP_USR = """Contexts:
{ctxs}

Answer:
{ans}

Is the answer fully supported by the contexts?"""


def _tf(s: str) -> bool:
    s = (s or "").strip().lower()
    return s.startswith("t") and "false" not in s


def probe_context_relevance(question: str, contexts: List[str]) -> List[bool]:
    out = []
    for i, c in enumerate(contexts):
        resp = chat(
            [
                {"role": "system", "content": REL_SYS},
                {"role": "user", "content": REL_USR.format(q=question, ctx=c[:2000])},
            ]
        )
        val = _tf(resp)
        print(f"[DBG] ctx#{i} relevance -> {val} | raw='{resp}'")
        out.append(val)
    return out


def probe_answer_support(answer: str, contexts: List[str]) -> bool:
    joined = "\n\n---\n\n".join([c[:2000] for c in contexts[:5]])
    resp = chat(
        [
            {"role": "system", "content": SUPP_SYS},
            {"role": "user", "content": SUPP_USR.format(ctxs=joined, ans=answer)},
        ]
    )
    val = _tf(resp)
    print(f"[DBG] answer supported by contexts -> {val} | raw='{resp}'")
    return val


def dump_bundle(question: str, answer: str, contexts: List[str], ground_truth: str):
    print("\n=== DEBUG BUNDLE ===")
    print(f"Q: {question}")
    print(f"GT: {ground_truth}")
    print(
        f"Answer(len={len(answer)}): {answer[:400]}{'...' if len(answer)>400 else ''}"
    )
    print(f"#contexts={len(contexts)}")
    for i, c in enumerate(contexts):
        preview = c.replace("\n", " ")
        print(f"- ctx#{i} (len={len(c)}): {preview}{'...' if len(c)>200 else ''}")


from llm import chat


def _tf(x: str) -> bool:
    s = (x or "").strip().lower()
    return s.startswith("t") and "false" not in s


def probe_answer_supported_by_contexts(answer: str, contexts: list[str]):
    ctx = "\n\n---\n\n".join([c[:2000] for c in contexts[:5]])
    out = chat(
        [
            {"role": "system", "content": "Return only 'true' or 'false'."},
            {
                "role": "user",
                "content": f"Contexts:\n{ctx}\n\nAnswer:\n{answer}\n\nIs every claim in the Answer fully supported by the Contexts?",
            },
        ]
    )
    print(f"[DBG] supported? -> {_tf(out)} | raw='{out}'")


def probe_each_context_relevance(question: str, contexts: list[str]):
    for i, c in enumerate(contexts):
        out = chat(
            [
                {"role": "system", "content": "Return only 'true' or 'false'."},
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nContext:\n{c[:2000]}\n\nDoes this context help answer the question?",
                },
            ]
        )
        print(f"[DBG] ctx#{i} relevant? -> {_tf(out)} | raw='{out}'")
