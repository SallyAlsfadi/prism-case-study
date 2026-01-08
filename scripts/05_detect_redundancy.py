#!/usr/bin/env python3
# utils/redundancy_detector.py
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Dict, List
from collections import defaultdict

# Thresholds (configurable)
COS_MIN = 0.86
LEN_RATIO_MIN = 0.60
JACCARD_MIN = 0.15
GROUP_AVG_COS = 0.88
GROUP_MIN_COS = 0.83

def jaccard_overlap(a: str, b: str) -> float:
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def length_ratio(a: str, b: str) -> float:
    la, lb = len(a.split()), len(b.split())
    return min(la, lb) / max(la, lb) if max(la, lb) > 0 else 0

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
    def find(self, x: int) -> int:
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx

def process_file(texts_csv: str, embs_npy: str, out_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(texts_csv)
    embs = np.load(embs_npy)
    ids, texts = df["req_id"].tolist(), df["text"].tolist()
    n = len(ids)

    base = os.path.splitext(os.path.basename(texts_csv))[0]
    os.makedirs(out_dir, exist_ok=True)

    # Edge case: fewer than 2 requirements
    if n < 2:
        print(f"   ⚠️ Not enough requirements in {texts_csv} (n={n}), skipping pairs.")
        pd.DataFrame(columns=["id1","text1","id2","text2","cosine","len_ratio","jaccard"]) \
            .to_csv(os.path.join(out_dir, f"{base}_all_pairs.csv"), index=False)
        pd.DataFrame().to_csv(os.path.join(out_dir, f"{base}_pairs.csv"), index=False)
        pd.DataFrame().to_csv(os.path.join(out_dir, f"{base}_proposed_merges.csv"), index=False)
        return pd.DataFrame(), pd.DataFrame()

    sims = cosine_similarity(embs)
    all_pairs = []

    # Step 1: compute ALL pairs (raw)
    for i, j in combinations(range(n), 2):
        cos = sims[i, j]
        lr = length_ratio(texts[i], texts[j])
        jac = jaccard_overlap(texts[i], texts[j])
        all_pairs.append({
            "id1": ids[i], "text1": texts[i],
            "id2": ids[j], "text2": texts[j],
            "cosine": cos, "len_ratio": lr, "jaccard": jac
        })

    all_df = pd.DataFrame(all_pairs)

    # Step 2: filter candidate pairs
    filt_df = all_df[
        (all_df["cosine"] >= COS_MIN) &
        (all_df["len_ratio"] >= LEN_RATIO_MIN) &
        (all_df["jaccard"] >= JACCARD_MIN)
    ].reset_index(drop=True)

    # Step 3: build groups with union-find
    idx_map = {rid: k for k, rid in enumerate(ids)}
    uf = UnionFind(n)
    for _, row in filt_df.iterrows():
        uf.union(idx_map[row["id1"]], idx_map[row["id2"]])
    groups = defaultdict(list)
    for rid, idx in idx_map.items():
        groups[uf.find(idx)].append(rid)

    # Step 4: evaluate groups for safe merges
    proposals = []
    for g_idx, members in enumerate(groups.values(), 1):
        if len(members) < 2:
            continue
        m_idx = [idx_map[m] for m in members]
        sub_sims = sims[np.ix_(m_idx, m_idx)]
        avg_cos = (sub_sims.sum() - len(members)) / (len(members) * (len(members) - 1))
        min_cos = (sub_sims + np.eye(len(members))).min()

        if avg_cos >= GROUP_AVG_COS and min_cos >= GROUP_MIN_COS:
            # Pick representative = longest text
            rep_id = max(members, key=lambda rid: len(df.loc[df.req_id == rid, "text"].values[0]))
            rep_text = df.loc[df.req_id == rep_id, "text"].values[0]
            proposals.append({
                "proposal_id": f"merge_{g_idx:03d}",
                "member_ids": "|".join(members),
                "suggested_representative_id": rep_id,
                "suggested_representative_text": rep_text,
                "avg_cos": round(avg_cos, 4),
                "min_cos": round(min_cos, 4),
                "size": len(members)
            })

    props_df = pd.DataFrame(proposals)

    # Step 5: save outputs
    all_df.to_csv(os.path.join(out_dir, f"{base}_all_pairs.csv"), index=False)
    filt_df.to_csv(os.path.join(out_dir, f"{base}_pairs.csv"), index=False)
    props_df.to_csv(os.path.join(out_dir, f"{base}_proposed_merges.csv"), index=False)

    return filt_df, props_df

def main():
    parser = argparse.ArgumentParser(description="Detect redundant requirements.")
    parser.add_argument("--input_dir", default="embeddings/domain_class", help="Input directory with *_texts.csv + .npy")
    parser.add_argument("--output_dir", default="results/redundancy", help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = [f for f in os.listdir(args.input_dir) if f.endswith("_texts.csv")]
    print(f" Found {len(files)} text files for redundancy detection")

    for fname in files:
        texts_csv = os.path.join(args.input_dir, fname)
        embs_npy = os.path.join(args.input_dir, fname.replace("_texts.csv", ".npy"))
        if not os.path.exists(embs_npy):
            print(f"⚠️ Missing embeddings for {fname}, skipping.")
            continue
        print(f" Processing {fname} …")
        filt_df, props_df = process_file(texts_csv, embs_npy, args.output_dir)
        print(f"   Candidate pairs (filtered): {len(filt_df)}, Proposed merges: {len(props_df)}")

if __name__ == "__main__":
    main()
