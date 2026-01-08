#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
import pandas as pd
import numpy as np

EPS = 1e-9
SPAN_MIN = 1e-6

def normalize_systemwise(df):
    out = []
    span_log = []
    for sys, g in df.groupby("system", sort=False):
        g = g.copy()

        # benefit transform: stability benefit
        g["R"] = 1.0 - g["V"].astype(float)

        for raw, xcol in [("C","x_C"), ("S","x_S"), ("R","x_R")]:
            lo = float(g[raw].min())
            hi = float(g[raw].max())
            span = hi - lo

            if span < SPAN_MIN:
                g[xcol] = 0.5
                span_log.append({"system": sys, "metric": raw, "min": lo, "max": hi,
                                 "span": span, "status": "neutralized_0.5"})
            else:
                g[xcol] = (g[raw] - lo) / (span + EPS)
                g[xcol] = g[xcol].clip(0.0, 1.0)
                span_log.append({"system": sys, "metric": raw, "min": lo, "max": hi,
                                 "span": span, "status": "ok"})
        out.append(g)

    return pd.concat(out, ignore_index=True), pd.DataFrame(span_log)

def main():
    ap = argparse.ArgumentParser(description="Benefit transform + normalize per system")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", default="results_rerun/final_scoring/step1_normalized.csv")
    ap.add_argument("--report_json", default="results_rerun/final_scoring/step1_span_report.json")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    needed = {"req_id","system","class","C","S","V"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    norm_df, span_df = normalize_systemwise(df)

    cols = ["req_id","system","class","C","S","V","R","x_C","x_S","x_R"]
    if "text" in norm_df.columns: cols.append("text")
    norm_df[cols].to_csv(args.output_csv, index=False)

    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump({"eps": EPS, "span_min": SPAN_MIN, "spans": span_df.to_dict("records")}, f, indent=2)

    print(f"[OK] Normalized → {args.output_csv} (rows={len(norm_df)})")
    print(f"[OK] Span report → {args.report_json}")

if __name__ == "__main__":
    main()
