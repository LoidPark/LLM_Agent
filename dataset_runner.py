# dataset_runner.py
# 엑셀(evaluation_dataset.xlsx)에서 (query, ground_truth) 반복 평가
# 요구: pandas, openpyxl (또는 xlrd/xlsxwriter) 설치

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
from evaluate import run_eval  # 위에서 추가한 함수
import os
import time

INPUT_XLSX = "./evaluation_dataset.xlsx"  # 엑셀 파일 경로
SHEET_NAME = 0  # 시트 인덱스 또는 이름
COL_QUERY = "query"
COL_GT = "ground_truth"
TOP_K = 3

OUT_DIR = Path("./reports/dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: str, sheet=0) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)  # pip install openpyxl
    # 컬럼 이름 정규화
    rename = {c: c.strip().lower() for c in df.columns}
    df.rename(columns=rename, inplace=True)
    if COL_QUERY not in df.columns or COL_GT not in df.columns:
        raise ValueError(
            f"엑셀에 '{COL_QUERY}', '{COL_GT}' 컬럼이 있어야 합니다. 현재 cols={list(df.columns)}"
        )
    # 앞/뒤 공백 제거
    df[COL_QUERY] = df[COL_QUERY].astype(str).str.strip()
    df[COL_GT] = df[COL_GT].astype(str).str.strip()
    # 빈 행 제거
    df = df[(df[COL_QUERY] != "") & (df[COL_GT] != "")]
    df.reset_index(drop=True, inplace=True)
    return df


def run_batch(df: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n = len(df)
    for i, row in df.iterrows():
        q = row[COL_QUERY]
        gt = row[COL_GT]
        print(f"\n[{i+1}/{n}] Evaluating: {q[:80]}{'...' if len(q)>80 else ''}")
        try:
            result = run_eval(q, gt, top_k=top_k)
        except Exception as e:
            print(f"[ERROR] row {i+1}: {e}")
            result = {
                "question": q,
                "ground_truth": gt,
                "answer": "",
                "num_contexts": 0,
                "our_context_precision": float("nan"),
                "our_context_recall": float("nan"),
                "our_faithfulness": float("nan"),
                "our_answer_relevancy": float("nan"),
                "ragas_context_precision": float("nan"),
                "ragas_context_recall": float("nan"),
                "ragas_faithfulness": float("nan"),
                "ragas_answer_relevancy": float("nan"),
                "judge_completeness": float("nan"),
                "judge_usefulness": float("nan"),
                "judge_clarity": float("nan"),
                "judge_relevance": float("nan"),
                "judge_additional_value": float("nan"),
                "judge_error_penalty": float("nan"),
                "judge_final": float("nan"),
                "error": str(e),
            }
        rows.append(result)
        # API rate-limit/안정화용 소잠깐 대기(필요 시)
        time.sleep(0.25)
    return pd.DataFrame(rows)


def save_outputs(df_out: pd.DataFrame):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"summary_{ts}.csv"
    xlsx_path = OUT_DIR / f"summary_{ts}.xlsx"
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df_out.to_excel(xlsx_path, index=False)
    except Exception as e:
        print(f"[WARN] Excel 저장 실패: {e} (CSV는 저장됨)")
    print(f"\n[Saved] {csv_path}")
    if xlsx_path.exists():
        print(f"[Saved] {xlsx_path}")


if __name__ == "__main__":
    print(f"[LOAD] {INPUT_XLSX}")
    df = load_dataset(INPUT_XLSX, sheet=SHEET_NAME)
    print(f"[ROWS] {len(df)} rows")

    df_out = run_batch(df, top_k=TOP_K)
    save_outputs(df_out)
