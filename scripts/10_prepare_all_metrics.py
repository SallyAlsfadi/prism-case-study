#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
import pandas as pd

def load_specificity(path):
    df = pd.read_csv(path)
    keep = [c for c in ["req_id","system","class","text","S_final_sys"] if c in df.columns]
    df = df[keep].copy()
    df = df.rename(columns={"S_final_sys":"S"})
    df["req_id"] = df["req_id"].astype(str)
    return df

def load_volatility(path):
    df = pd.read_csv(path)
    keep = [c for c in ["req_id","system","class","text","V_final"] if c in df.columns]
    df = df[keep].copy()
    df = df.rename(columns={"V_final":"V"})
    df["req_id"] = df["req_id"].astype(str)
    return df

def load_criticality(path):
    df = pd.read_csv(path)
    keep = [c for c in ["req_id","system","class","Criticality"] if c in df.columns]
    df = df[keep].copy()
    df = df.rename(columns={"Criticality":"C"})
    df["req_id"] = df["req_id"].astype(str)
    return df

def merge_all(sp_df, v_df, c_df, join_on_class=True):
    # keys
    keys = ["req_id","system","class"] if join_on_class else ["req_id","system"]

    # Avoid text collisions
    v_tmp = v_df.copy()
    if "text" in v_tmp.columns:
        v_tmp = v_tmp.rename(columns={"text":"text_vol"})

    # Merge
    m = c_df.merge(sp_df, on=keys, how="inner")
    m = m.merge(v_tmp, on=keys, how="inner")

    # Choose one text
    if "text" in m.columns and "text_vol" in m.columns:
        m["text"] = m["text"].fillna(m["text_vol"])
        m = m.drop(columns=["text_vol"])
    elif "text_vol" in m.columns and "text" not in m.columns:
        m = m.rename(columns={"text_vol":"text"})

    # If relaxed join, keep class from specificity if present, else from volatility/criticality
    if not join_on_class:
        if "class_x" in m.columns or "class_y" in m.columns:
            # in case pandas produced suffixes
            for cand in ["class", "class_x", "class_y"]:
                if cand in m.columns:
                    m["class"] = m[cand]
                    break
            drop = [c for c in ["class_x","class_y"] if c in m.columns]
            if drop: m = m.drop(columns=drop)

    # numeric safety
    for col in ["C","S","V"]:
        m[col] = pd.to_numeric(m[col], errors="coerce")

    return m

def audit(sp_df, v_df, c_df, merged, join_on_class=True):
    if join_on_class:
        keycols = ["req_id","system","class"]
    else:
        keycols = ["req_id","system"]

    def keyset(d):
        return set(tuple(x) for x in d[keycols].astype(str).itertuples(index=False, name=None))

    ids_sp = keyset(sp_df)
    ids_v  = keyset(v_df)
    ids_c  = keyset(c_df)
    ids_all = ids_sp & ids_v & ids_c

    return {
        "join_on_class": bool(join_on_class),
        "counts": {
            "specificity_rows": len(sp_df),
            "volatility_rows": len(v_df),
            "criticality_rows": len(c_df),
            "intersection_keys": len(ids_all),
            "merged_rows": len(merged),
        },
        "examples_missing": {
            "only_specificity": sorted(list(ids_sp - ids_all))[:30],
            "only_volatility":  sorted(list(ids_v  - ids_all))[:30],
            "only_criticality": sorted(list(ids_c  - ids_all))[:30],
        }
    }

def main():
    ap = argparse.ArgumentParser(description="Prepare unified input for final scoring")
    ap.add_argument("--specificity_csv", default="results_rerun/specificity/specificity_scored.csv")
    ap.add_argument("--volatility_csv",  default="results_rerun/volatility/volatility_scored.csv")
    ap.add_argument("--criticality_csv", default="results_rerun/criticality/criticality_scored.csv")
    ap.add_argument("--output_csv",      default="results_rerun/final_scoring/all_metrics_scored.csv")
    ap.add_argument("--report_json",     default="results_rerun/final_scoring/all_metrics_scored_audit.json")
    ap.add_argument("--join_on_class",   action="store_true",
                    help="If set, join on (req_id,system,class). If not set, join on (req_id,system) only.")
    args = ap.parse_args()

    sp = load_specificity(args.specificity_csv)
    v  = load_volatility(args.volatility_csv)
    c  = load_criticality(args.criticality_csv)

    merged = merge_all(sp, v, c, join_on_class=args.join_on_class)

    out_cols = ["req_id","system","class","C","S","V"] + (["text"] if "text" in merged.columns else [])
    merged[out_cols].to_csv(args.output_csv, index=False)

    rep = audit(sp, v, c, merged, join_on_class=args.join_on_class)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    print(f"[OK] Wrote {len(merged)} rows → {args.output_csv}")
    print(f"[OK] Audit → {args.report_json}")

if __name__ == "__main__":
    main()
