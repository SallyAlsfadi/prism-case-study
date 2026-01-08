#!/usr/bin/env python3
# utils/data_splitter.py
from __future__ import annotations

import os
import re
import argparse
from typing import Tuple, Dict, List

import pandas as pd

RE_NONWORD = re.compile(r"[^\w\-]+")
RE_MULTI_US = re.compile(r"_+")


def safe_slug(s: str) -> str:
    """Lowercase, replace non-word chars with underscores, collapse dups, strip."""
    s = str(s).strip().lower()
    s = RE_NONWORD.sub("_", s)
    s = RE_MULTI_US.sub("_", s)
    return s.strip("_")


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize columns, trim whitespace, and drop rows with empty text.
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    required = {"text", "class", "system"}
    if not required.issubset(df.columns):
        missing = required.difference(df.columns)
        raise ValueError(f"CSV must contain columns {required}, missing: {missing}")

    # Strip and normalize blanks → NaN
    for col in ["text", "class", "system"]:
        mask = df[col].notna()
        df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip()
        df.loc[df[col].astype(str).eq(""), col] = pd.NA

    # Drop rows with empty/NaN text
    before = len(df)
    df = df[~df["text"].isna() & (df["text"].astype(str).str.len() > 0)].copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"⚠️ Dropped {dropped} rows with empty 'text'")

    return df[["text", "class", "system"]]


def split_data_by_class_and_system(
    input_csv_path: str,
    output_dir: str,
    text_only: bool = False,
) -> Tuple[int, Dict[str, int]]:
    """
    Split master CSV into {system}_{class}.csv files under output_dir.
    Each requirement gets a stable req_id.
    """
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv_path)
    df = normalize_df(df)

    grouped = df.groupby(["system", "class"], sort=True, dropna=False)

    total_written = 0
    per_file_counts: Dict[str, int] = {}
    manifest_rows: List[Dict[str, str | int]] = []

    print(f" Detected {len(grouped)} (system,class) groups.")

    for (system, req_class), g in grouped:
        if g.empty:
            continue

        sys_slug = safe_slug(system if pd.notna(system) else "nan")
        cls_slug = safe_slug(req_class if pd.notna(req_class) else "nan")

        # Add IDs: system_class_001 style
        g = g.reset_index(drop=True)
        g.insert(
            0,
            "req_id",
            [f"{sys_slug}_{cls_slug}_{i+1:03d}" for i in range(len(g))]
        )

        filename = f"{sys_slug}_{cls_slug}.csv"
        filepath = os.path.join(output_dir, filename)

        if text_only:
            out_df = g[["req_id", "text"]]
        else:
            out_df = g[["req_id", "text", "class", "system"]]

        out_df.to_csv(filepath, index=False)

        n = len(g)
        total_written += n
        per_file_counts[filename] = n
        print(f" Saved {n:4d} rows → {filepath}")

        manifest_rows.append({
            "system": system,
            "class": req_class,
            "filename": filename,
            "path": filepath,
            "rows": n,
            "text_only": text_only,
        })

    # Write manifest
    if manifest_rows:
        manifest = pd.DataFrame(manifest_rows).sort_values(["system", "class"], na_position="last")
        manifest_path = os.path.join(output_dir, "_manifest.csv")
        manifest.to_csv(manifest_path, index=False)
        print(f"  Manifest written → {manifest_path}")

    print(f" Split complete. Total rows written: {total_written}")
    return total_written, per_file_counts


def main():
    parser = argparse.ArgumentParser(description="Split dataset by (system, class).")
    parser.add_argument(
        "--input",
        default="data/requirements.csv",
        help="Path to master CSV (default: data/requirements.csv)",
    )
    parser.add_argument(
        "--outdir",
        default="data/split_by_class-system",
        help="Output directory (default: data/split_by_class-system)",
    )
    parser.add_argument(
        "--text_only",
        action="store_true",
        help="Write text-only shards (default off). Recommended for embedding.",
    )
    args = parser.parse_args()

    print("🔧 Splitting dataset by (system, class)…")
    total, perfile = split_data_by_class_and_system(
        args.input,
        args.outdir,
        text_only=args.text_only,
    )
    print(" Per-file counts:")
    for fname, n in sorted(perfile.items()):
        print(f"   - {fname}: {n}")
    print(" Done.")


if __name__ == "__main__":
    main()
