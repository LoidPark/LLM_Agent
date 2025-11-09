from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
import time
import json
import os
import math

from explain import explain_bundle

# from llm_as_a_judge_explain import generate_explanation, compare_reasoning

# (선택) 사실상 key_claims를 자동 추출하려면 이 함수가 있으면 사용하고,
# 없으면 None으로 보냄.
try:
    from judge import extract_needed_facts  # 또는 facts_extractor 모듈명

    HAS_FACTS = True
except Exception:
    HAS_FACTS = False

# INPUT_XLSX = "./evaluation_dataset_comparison.xlsx"
INPUT_XLSX = "./evaluation_dataset_comparison(sample).xlsx"
SHEET_NAME = 0

COL_Q = "query"
COL_GT = "ground_truth"
COL_NA = "naive_answer"
COL_PA = "parser_answer"

OUT_DIR = Path("./reports/comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _to_str(s) -> str:
    return "" if s is None else str(s).strip()


def load_dataset(path: str, sheet=0) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)  # pip install openpyxl
    # 컬럼 소문자화
    rename = {c: c.strip().lower() for c in df.columns}
    df.rename(columns=rename, inplace=True)

    # 필수 컬럼 존재 확인
    req = {COL_Q, COL_GT, COL_NA, COL_PA}
    if not req.issubset(set(df.columns)):
        raise ValueError(
            f"엑셀에 {sorted(list(req))} 컬럼이 모두 있어야 합니다. 현재 cols={list(df.columns)}"
        )

    # 정리
    for c in [COL_Q, COL_GT, COL_NA, COL_PA]:
        df[c] = df[c].apply(_to_str)

    # 최소: q, gt는 비어있으면 제외
    df = df[(df[COL_Q] != "") & (df[COL_GT] != "")]
    df.reset_index(drop=True, inplace=True)
    return df


def judge_one(
    question: str, ground_truth: str, answer: str, save_json_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    컨텍스트 없이(빈 리스트) LLM-as-a-Judge만 수행.
    explanations(why)까지 함께 생성하여 JSON으로 저장(옵션).
    """
    contexts = []
    if HAS_FACTS:
        try:
            facts = extract_needed_facts(question, ground_truth)  # 있으면 사용
        except Exception:
            facts = None
    else:
        facts = None

    bundle = explain_bundle(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
        key_claims=facts,
        weight_additional_value=0.15,
        penalty_softness=0.20,
        save_path=save_json_path,  # None이면 저장 안함
    )
    return bundle  # scores, details, explanations, formula, meta


def run_batch(df: pd.DataFrame, save_json_each: bool = False) -> pd.DataFrame:
    rows = []
    n = len(df)
    ts_all = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _score(bundle: Dict[str, Any], key: str) -> float:
        try:
            v = float(bundle.get("scores", {}).get(key, float("nan")))
            return v
        except Exception:
            return float("nan")

    def _reason(exps: Dict[str, Any], metric: str, field: str) -> str:
        # field: "summary_en" / "summary_ko" / "improvement_en" / "improvement_ko"
        obj = exps.get(metric, {})
        return str(obj.get(field, ""))

    for i, row in df.iterrows():
        q = row[COL_Q]
        gt = row[COL_GT]
        na = row[COL_NA]
        pa = row[COL_PA]

        print(
            f"\n[{i+1}/{n}] Evaluating pair — Q: {q[:70]}{'...' if len(q)>70 else ''}"
        )

        save_na = (
            str(OUT_DIR / f"naive_judge_{ts_all}_{i+1:04d}.json")
            if save_json_each
            else None
        )
        save_pa = (
            str(OUT_DIR / f"parser_judge_{ts_all}_{i+1:04d}.json")
            if save_json_each
            else None
        )

        # --- naive ---
        try:
            na_bundle = judge_one(q, gt, na, save_json_path=save_na)
        except Exception as e:
            print(f"[WARN] naive_answer judge failed: {e}")
            na_bundle = {"scores": {}, "explanations": {}}

        # --- parser ---
        try:
            pa_bundle = judge_one(q, gt, pa, save_json_path=save_pa)
        except Exception as e:
            print(f"[WARN] parser_answer judge failed: {e}")
            pa_bundle = {"scores": {}, "explanations": {}}

        # def s(bundle: Dict[str, Any], key: str) -> float:
        #     return float(bundle.get("scores", {}).get(key, float("nan")))

        metrics = [
            "completeness",
            "usefulness",
            "clarity",
            "relevance",
            "additional_value",
            "error_penalty",
            "final",
        ]

        na_scores = {k: _score(na_bundle, k) for k in metrics}
        pa_scores = {k: _score(pa_bundle, k) for k in metrics}

        # 설명 추출 (영/한)
        na_exps = na_bundle.get("explanations", {})
        pa_exps = pa_bundle.get("explanations", {})

        # delta 계산
        def _delta(m: str) -> float:
            a = na_scores.get(m, float("nan"))
            b = pa_scores.get(m, float("nan"))
            if math.isnan(a) or math.isnan(b):
                return float("nan")
            return b - a

        # def get_reason(exps: Dict[str, Any], metric: str, field: str) -> str:
        #     # field: "summary_en" / "summary_ko" / "improvement_en" / ...
        #     obj = exps.get(metric, {})
        #     return str(obj.get(field, ""))

        # 최종 승자
        def _winner(a: float, b: float) -> str:
            if math.isnan(a) and math.isnan(b):
                return "tie_nan"
            if math.isnan(a):
                return "parser"
            if math.isnan(b):
                return "naive"
            if b > a:
                return "parser"
            if b < a:
                return "naive"
            return "tie"

        winner = _winner(na_scores["final"], pa_scores["final"])

        # ---- 결과 행 구성 ----
        row_out: Dict[str, Any] = {
            "query": q,
            "ground_truth": gt,
            "naive_answer": na,
            "parser_answer": pa,
            # naive 점수
            "naive_completeness": round(na_scores["completeness"], 3),
            "naive_usefulness": round(na_scores["usefulness"], 3),
            "naive_clarity": round(na_scores["clarity"], 3),
            "naive_relevance": round(na_scores["relevance"], 3),
            "naive_additional_value": round(na_scores["additional_value"], 3),
            "naive_error_penalty": round(na_scores["error_penalty"], 3),
            "naive_final": round(na_scores["final"], 3),
            # parser 점수
            "parser_completeness": round(pa_scores["completeness"], 3),
            "parser_usefulness": round(pa_scores["usefulness"], 3),
            "parser_clarity": round(pa_scores["clarity"], 3),
            "parser_relevance": round(pa_scores["relevance"], 3),
            "parser_additional_value": round(pa_scores["additional_value"], 3),
            "parser_error_penalty": round(pa_scores["error_penalty"], 3),
            "parser_final": round(pa_scores["final"], 3),
            # delta & winner
            "delta_completeness": round(_delta("completeness"), 3),
            "delta_usefulness": round(_delta("usefulness"), 3),
            "delta_clarity": round(_delta("clarity"), 3),
            "delta_relevance": round(_delta("relevance"), 3),
            "delta_additional_value": round(_delta("additional_value"), 3),
            "delta_error_penalty": round(_delta("error_penalty"), 3),
            "delta_final": round(_delta("final"), 3),
            "winner(final)": winner,
        }

        # ---- metric별 영/한 summary & 개선점 ----
        for m in [
            "completeness",
            "usefulness",
            "clarity",
            "relevance",
            "additional_value",
            "error_penalty",
        ]:
            row_out[f"naive_{m}_summary_en"] = _reason(na_exps, m, "summary_en")
            row_out[f"naive_{m}_summary_ko"] = _reason(na_exps, m, "summary_ko")
            row_out[f"naive_{m}_improve_en"] = _reason(na_exps, m, "improvement_en")
            row_out[f"naive_{m}_improve_ko"] = _reason(na_exps, m, "improvement_ko")

            row_out[f"parser_{m}_summary_en"] = _reason(pa_exps, m, "summary_en")
            row_out[f"parser_{m}_summary_ko"] = _reason(pa_exps, m, "summary_ko")
            row_out[f"parser_{m}_improve_en"] = _reason(pa_exps, m, "improvement_en")
            row_out[f"parser_{m}_improve_ko"] = _reason(pa_exps, m, "improvement_ko")

        rows.append(row_out)
        time.sleep(0.2)  # 레이트리밋 완충

    return pd.DataFrame(rows)


# def run_batch(df: pd.DataFrame, save_json_each: bool = False) -> pd.DataFrame:
#     rows = []
#     n = len(df)
#     ts_all = datetime.now().strftime("%Y%m%d_%H%M%S")

#     for i, row in df.iterrows():
#         q = row[COL_Q]
#         gt = row[COL_GT]
#         na = row[COL_NA]  # naive_answer
#         pa = row[COL_PA]  # parser_answer

#         print(
#             f"\n[{i+1}/{n}] Evaluating pair — Q: {q[:70]}{'...' if len(q)>70 else ''}"
#         )

#         # 개별 JSON 저장 경로(옵션)
#         save_na = (
#             str(OUT_DIR / f"naive_judge_{ts_all}_{i+1:04d}.json")
#             if save_json_each
#             else None
#         )
#         save_pa = (
#             str(OUT_DIR / f"parser_judge_{ts_all}_{i+1:04d}.json")
#             if save_json_each
#             else None
#         )

#         # 빈 답변 처리: LLM-as-a-Judge는 빈 답변도 채점 가능하지만,
#         # 실무상 0에 수렴하므로 명시적으로 빈 문자열은 그대로 전달.
#         try:
#             na_bundle = judge_one(q, gt, na, save_json_path=save_na)
#         except Exception as e:
#             print(f"[WARN] naive_answer judge failed: {e}")
#             na_bundle = {
#                 "scores": {
#                     k: float("nan")
#                     for k in [
#                         "completeness",
#                         "usefulness",
#                         "clarity",
#                         "relevance",
#                         "additional_value",
#                         "error_penalty",
#                         "final",
#                     ]
#                 }
#             }

#         try:
#             pa_bundle = judge_one(q, gt, pa, save_json_path=save_pa)
#         except Exception as e:
#             print(f"[WARN] parser_answer judge failed: {e}")
#             pa_bundle = {
#                 "scores": {
#                     k: float("nan")
#                     for k in [
#                         "completeness",
#                         "usefulness",
#                         "clarity",
#                         "relevance",
#                         "additional_value",
#                         "error_penalty",
#                         "final",
#                     ]
#                 }
#             }

#         # 스코어 추출
#         def s(bundle: Dict[str, Any], key: str) -> float:
#             return float(bundle.get("scores", {}).get(key, float("nan")))

#         # 개별 점수
#         na_scores = {
#             k: s(na_bundle, k)
#             for k in [
#                 "completeness",
#                 "usefulness",
#                 "clarity",
#                 "relevance",
#                 "additional_value",
#                 "error_penalty",
#                 "final",
#             ]
#         }
#         pa_scores = {
#             k: s(pa_bundle, k)
#             for k in [
#                 "completeness",
#                 "usefulness",
#                 "clarity",
#                 "relevance",
#                 "additional_value",
#                 "error_penalty",
#                 "final",
#             ]
#         }

#         # 델타(파서 - 네이브)
#         delta = {f"delta_{k}": (pa_scores[k] - na_scores[k]) for k in na_scores.keys()}

#         # 승자 (final 기준)
#         def _winner(n_final: float, p_final: float) -> str:
#             if pd.isna(n_final) and pd.isna(p_final):
#                 return "tie_nan"
#             if pd.isna(n_final):
#                 return "parser"
#             if pd.isna(p_final):
#                 return "naive"
#             if p_final > n_final:
#                 return "parser"
#             if p_final < n_final:
#                 return "naive"
#             return "tie"

#         winner = _winner(na_scores["final"], pa_scores["final"])

#         result_row = {
#             "query": q,
#             "ground_truth": gt,
#             "naive_answer": na,
#             "parser_answer": pa,
#             # naive
#             "naive_completeness": round(na_scores["completeness"], 3),
#             "naive_usefulness": round(na_scores["usefulness"], 3),
#             "naive_clarity": round(na_scores["clarity"], 3),
#             "naive_relevance": round(na_scores["relevance"], 3),
#             "naive_additional_value": round(na_scores["additional_value"], 3),
#             "naive_error_penalty": round(na_scores["error_penalty"], 3),
#             "naive_final": round(na_scores["final"], 3),
#             # parser
#             "parser_completeness": round(pa_scores["completeness"], 3),
#             "parser_usefulness": round(pa_scores["usefulness"], 3),
#             "parser_clarity": round(pa_scores["clarity"], 3),
#             "parser_relevance": round(pa_scores["relevance"], 3),
#             "parser_additional_value": round(pa_scores["additional_value"], 3),
#             "parser_error_penalty": round(pa_scores["error_penalty"], 3),
#             "parser_final": round(pa_scores["final"], 3),
#             # deltas
#             **{k: round(v, 3) for k, v in delta.items()},
#             "winner(final)": winner,
#         }

#         rows.append(result_row)
#         time.sleep(0.2)  # API 완충

#     return pd.DataFrame(rows)


def save_outputs(df_out: pd.DataFrame):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"summary_{ts}.csv"
    xlsx_path = OUT_DIR / f"summary_{ts}.xlsx"
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df_out.to_excel(xlsx_path, index=False)
    except Exception as e:
        print(f"[WARN] Excel 저장 실패: {e} (CSV는 저장됨)")
    print(f"[Saved] {csv_path}")
    if xlsx_path.exists():
        print(f"[Saved] {xlsx_path}")


if __name__ == "__main__":
    print(f"[LOAD] {INPUT_XLSX}")
    df = load_dataset(INPUT_XLSX, sheet=SHEET_NAME)
    print(df)
    print(f"[ROWS] {len(df)} rows")

    out_df = run_batch(df, save_json_each=False)
    save_outputs(out_df)


# for idx, row in df.iterrows():
#     q = row["query"]
#     gt = row["ground_truth"]
#     naive = row["naive_answer"]
#     parser = row["parser_answer"]

#     naive_metrics = {
#         "completeness": row["naive_completeness"],
#         "usefulness": row["naive_usefulness"],
#         "clarity": row["naive_clarity"],
#         "relevance": row["naive_relevance"],
#         "additional_value": row["naive_additional_value"],
#         "error_penalty": row["naive_error_penalty"],
#         "final": row["naive_final"],
#     }

#     parser_metrics = {
#         "completeness": row["parser_completeness"],
#         "usefulness": row["parser_usefulness"],
#         "clarity": row["parser_clarity"],
#         "relevance": row["parser_relevance"],
#         "additional_value": row["parser_additional_value"],
#         "error_penalty": row["parser_error_penalty"],
#         "final": row["parser_final"],
#     }

#     naive_expl = generate_explanation(q, gt, naive, naive_metrics)
#     parser_expl = generate_explanation(q, gt, parser, parser_metrics)

#     reasoning = compare_reasoning(naive_expl, parser_expl)
#     df.loc[idx, "naive_explanation"] = json.dumps(naive_expl, ensure_ascii=False)
#     df.loc[idx, "parser_explanation"] = json.dumps(parser_expl, ensure_ascii=False)
#     df.loc[idx, "reason_comparison"] = json.dumps(reasoning, ensure_ascii=False)

# df.to_excel("reports/comparison_explained.xlsx", index=False)
# print("[Saved] reports/comparison_explained.xlsx")
