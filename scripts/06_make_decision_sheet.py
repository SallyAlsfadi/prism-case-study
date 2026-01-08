#!/usr/bin/env python3
# utils/make_decision_sheet.py
from __future__ import annotations
import os, argparse, glob
import pandas as pd

REQUIRED_COLS = {
    "proposal_id","member_ids","suggested_representative_id",
    "suggested_representative_text","avg_cos","min_cos","size"
}

def main():
    p = argparse.ArgumentParser(description="Aggregate proposed merges into a single decision sheet.")
    p.add_argument("--proposals_dir", default="results/redundancy", help="Dir containing *_proposed_merges.csv files")
    p.add_argument("--out", default="results/redundancy/decisions.csv", help="Output decision sheet CSV")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.proposals_dir, "*_proposed_merges.csv")))
    if not files:
        print("No *_proposed_merges.csv files found.")
        return

    rows = []
    skipped = []

    for fp in files:
        base = os.path.basename(fp).replace("_proposed_merges.csv", "")
        # infer system,class from base like ecommerce_us_texts
        sys_cls = base.replace("_texts", "").split("_")
        system = sys_cls[0] if len(sys_cls) > 0 else ""
        req_class = sys_cls[1] if len(sys_cls) > 1 else ""

        # Skip zero-byte files early
        if os.path.getsize(fp) == 0:
            skipped.append((fp, "zero-byte"))
            continue

        try:
            df = pd.read_csv(fp)
        except pd.errors.EmptyDataError:
            skipped.append((fp, "empty-no-header"))
            continue

        if df.empty or not REQUIRED_COLS.issubset(set(df.columns.str.lower())):
            # normalize and check again
            df.columns = df.columns.str.lower()
            if df.empty or not REQUIRED_COLS.issubset(set(df.columns)):
                skipped.append((fp, "missing-required-cols-or-empty"))
                continue

        df.columns = df.columns.str.lower()
        df["system"] = system
        df["class"] = req_class
        rows.append(df[list(REQUIRED_COLS) + ["system","class"]])

    if not rows:
        print("No usable proposals found (all empty or invalid).")
        if skipped:
            print("Skipped files:")
            for fp, reason in skipped:
                print(f"  - {fp} → {reason}")
        return

    merged = pd.concat(rows, ignore_index=True)

    # Normalize column order and add decision fields
    merged = merged[[
        "proposal_id","member_ids","suggested_representative_id",
        "suggested_representative_text","avg_cos","min_cos","size","system","class"
    ]].copy()
    merged.insert(4, "decision", "")     # accept_merge | keep_originals | pick_one | edit
    merged.insert(5, "pick_one_id", "")  # required when decision == pick_one
    merged.insert(6, "edited_text", "")  # optional when decision == edit
    merged["comment"] = ""

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"Decision sheet written → {args.out}")
    if skipped:
        print("Skipped files:")
        for fp, reason in skipped:
            print(f"  - {fp} → {reason}")

if __name__ == "__main__":
    main()
