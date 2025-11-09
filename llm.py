# llm.py
import os
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL


def get_client():
    api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
    return OpenAI(api_key=api_key)


def chat(messages, model=OPENAI_MODEL) -> str:
    """
    messages = [
      {"role":"system","content":"..."},
      {"role":"user","content":"..."}
    ]
    """
    client = get_client()
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return resp.choices[0].message.content.strip()


def generate_with_context(
    question: str, contexts: list[str], model=OPENAI_MODEL
) -> str:
    joined_ctx = "\n\n---\n\n".join(contexts[:5])  # 상위 몇 개만
    messages = [
        {
            "role": "system",
            "content": "Answer the user's question using ONLY the provided context. If insufficient, say you don't know.",
        },
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nContext:\n{joined_ctx}\n\nGive a concise answer.",
        },
    ]
    return chat(messages, model=model)
