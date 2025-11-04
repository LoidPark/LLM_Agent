import os
from typing import List, Dict, Any
from config import OPENAI_API_KEY, OPENAI_MODEL
from config import LOCAL_BGE_PATH

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_MODEL"] = OPENAI_MODEL


def run_ragas_eval(
    question: str, answer: str, contexts: List[str], ground_truth: str
) -> Dict[str, float]:
    """
    Ragas로 context_precision, context_recall, faithfulness, answer_relevancy 계산.
    반환: {'context_precision': x, 'context_recall': y, 'faithfulness': z, 'answer_relevancy': w}
    """
    # 1) 입력 → HF Dataset
    try:
        from datasets import Dataset
    except Exception as e:
        raise RuntimeError(
            "`datasets` 패키지가 필요합니다. `pip install datasets` 후 다시 시도하세요."
        ) from e

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],  # list[str]
        "ground_truth": [ground_truth],  # str
    }
    ds = Dataset.from_dict(data)

    # 2) Ragas 임포트 (신/구 버전 호환)
    try:
        # 신규 API (ragas>=0.1.x~0.2.x)
        from ragas import evaluate
        from ragas.metrics import (
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        )

        # LLM / Embedding 백엔드
        try:
            # ragas 내장 백엔드 (있으면 이걸 사용)
            from ragas.llms import OpenAI as RagasOpenAI
            from ragas.embeddings import SentenceTransformerEmbeddings

            llm = RagasOpenAI(
                api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_KEY),
                model=os.getenv("OPENAI_MODEL", OPENAI_MODEL),
            )
            emb = SentenceTransformerEmbeddings(model_name=LOCAL_BGE_PATH)
            res = evaluate(
                ds,
                metrics=[
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_relevancy,
                ],
                llm=llm,
                embeddings=emb,
            )
        except Exception:
            # langchain 백엔드 폴백
            from langchain_openai import ChatOpenAI
            from langchain_community.embeddings import HuggingFaceEmbeddings

            llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", OPENAI_MODEL), temperature=0.0
            )
            emb = HuggingFaceEmbeddings(model_name=LOCAL_BGE_PATH)
            res = evaluate(
                ds,
                metrics=[
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_relevancy,
                ],
                llm=llm,
                embeddings=emb,
            )
    except Exception as e:
        raise RuntimeError(
            "Ragas 임포트/호출에 실패했습니다. `pip install ragas` 또는 버전을 확인하세요."
        ) from e

    # 3) 결과 파싱
    try:
        # ragas>=0.1: result.to_pandas() 가 점수 컬럼을 포함
        df = res.to_pandas()
        out = {}
        for k in [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
        ]:
            # 컬럼 이름이 바뀐 경우를 대비해 유사 키 검색
            cand = [c for c in df.columns if k in c]
            if cand:
                out[k] = float(df[cand[0]].iloc[0])
        return out
    except Exception:
        # 혹시나 해서 dict 형태도 시도
        try:
            return {
                "context_precision": float(res["context_precision"]),
                "context_recall": float(res["context_recall"]),
                "faithfulness": float(res["faithfulness"]),
                "answer_relevancy": float(res["answer_relevancy"]),
            }
        except Exception:
            raise
