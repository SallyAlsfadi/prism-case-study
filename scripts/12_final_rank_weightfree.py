#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os
import pandas as pd
import numpy as np

# ---------------------------
# Pareto fronts (3D) with dedup fix
# ---------------------------
def pareto_fronts(points):
    n = points.shape[0]
    dominated_counts = np.zeros(n, dtype=int)
    dominates = [set() for _ in range(n)]

    for i in range(n):
        pi = points[i]
        for j in range(i+1, n):
            pj = points[j]
            if np.all(pi >= pj) and np.any(pi > pj):
                dominates[i].add(j)
                dominated_counts[j] += 1
            elif np.all(pj >= pi) and np.any(pj > pi):
                dominates[j].add(i)
                dominated_counts[i] += 1

    front_label = np.zeros(n, dtype=int)
    current = [i for i in range(n) if dominated_counts[i] == 0]
    f = 1

    while current:
        # label current front
        for idx in current:
            front_label[idx] = f

        # build next front, deduplicated
        next_set = set()
        for p in current:
            for q in dominates[p]:
                dominated_counts[q] -= 1
                if dominated_counts[q] == 0:
                    next_set.add(q)

        current = list(next_set)
        f += 1

    # guard leftover
    if np.any(front_label == 0):
        front_label[front_label == 0] = f

    return front_label.tolist()

# ---------------------------
# Rank aggregation (weight-free)
# Best -> 0, Worst -> 1
# ---------------------------
def rank_percentiles(df, cols):
    out = {}
    n = len(df)
    denom = max(1, n - 1)
    for c in cols:
        r = df[c].rank(method="average", ascending=False)  # rank 1 is best
        out[c] = ((r - 1.0) / denom).astype(float)         # 0..1
    return out

def knee_score_row(xC, xS, xR):
    # closeness to ideal (1,1,1); higher is better
    gaps = [1 - xC, 1 - xS, 1 - xR]
    d_inf = max(gaps)
    d1 = sum(gaps)
    return 0.5 * (1 - d_inf) + 0.5 * (1 - d1 / 3.0)

def score_one_system(g):
    g = g.copy().reset_index(drop=True)

    pts = g[["x_C","x_S","x_R"]].to_numpy(float)
    g["F"] = pareto_fronts(pts)

    rp = rank_percentiles(g, ["x_C","x_S","x_R"])
    g["rra"] = (rp["x_C"] + rp["x_S"] + rp["x_R"]) / 3.0  # lower better

    g["K"] = [knee_score_row(a,b,c) for a,b,c in zip(g["x_C"], g["x_S"], g["x_R"])]

    if "text" in g.columns:
        g["text_len"] = g["text"].astype(str).str.len()
    else:
        g["text_len"] = np.nan

    g = g.sort_values(
        by=["F","rra","K","x_C","x_S","x_R","text_len","req_id"],
        ascending=[True, True, False, False, False, False, True, True],
        kind="mergesort"
    ).reset_index(drop=True)

    g["final_rank"] = np.arange(1, len(g) + 1)
    return g

def main():
    ap = argparse.ArgumentParser(description="System-level final scoring (weight-free)")
    ap.add_argument("--input_csv", default="results_rerun/final_scoring/step1_normalized.csv")
    ap.add_argument("--out_dir",   default="results_rerun/final_scoring")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.input_csv)

    needed = {"req_id","system","class","x_C","x_S","x_R"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    all_ranked = []
    fronts_summary = []

    for sys, g in df.groupby("system", sort=False):
        ranked = score_one_system(g)
        all_ranked.append(ranked)

        sys_safe = "".join(c if c.isalnum() or c in ("-","_") else "_" for c in str(sys))
        ranked.to_csv(os.path.join(args.out_dir, f"final_rankings_{sys_safe}.csv"), index=False)

        fronts_summary.append(
            ranked.groupby("F").size().reset_index(name="count").assign(system=sys)
        )

    all_ranked_df = pd.concat(all_ranked, ignore_index=True)
    all_ranked_df.to_csv(os.path.join(args.out_dir, "final_rankings_all.csv"), index=False)

    fs = pd.concat(fronts_summary, ignore_index=True)
    fs = fs[["system","F","count"]].sort_values(["system","F"])
    fs.to_csv(os.path.join(args.out_dir, "fronts_summary.csv"), index=False)

    print(f"[OK] Wrote {len(all_ranked_df)} rows → {os.path.join(args.out_dir,'final_rankings_all.csv')}")
    print(f"[OK] Fronts summary → {os.path.join(args.out_dir,'fronts_summary.csv')}")

if __name__ == "__main__":
    main()
