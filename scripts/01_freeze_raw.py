#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1A — Freeze + Standardize raw requirements.csv
===================================================

Input  : data/requirements.csv  (columns: text, class, system)
Output : data/frozen/requirements_frozen.csv
         data/frozen/requirements_manifest.json

Canonical schema:
  req_id, text, system, class, source

Notes:
- No semantic processing here (no embeddings, no merging).
- Only deterministic cleaning + ID assignment + integrity hash.
"""

from pathlib import Path
import hashlib, json
import pandas as pd

RAW_PATH = Path("data/requirements.csv")
OUT_DIR  = Path("data/frozen")
OUT_CSV  = OUT_DIR / "requirements_frozen.csv"
OUT_MAN  = OUT_DIR / "requirements_manifest.json"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def normalize_space(s: str) -> str:
    # collapse multiple spaces/tabs/newlines into single spaces
    return " ".join(str(s).replace("\u00a0", " ").split())

def main():
    if not RAW_PATH.exists():
        raise SystemExit(f"Missing input: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    # Validate minimal schema
    expected = {"text", "class", "system"}
    missing = expected - set(df.columns)
    if missing:
        raise SystemExit(f"requirements.csv is missing columns: {sorted(missing)}")

    # Standardize types + clean text
    df = df.copy()
    df["text"]   = df["text"].astype(str).map(normalize_space)
    df["class"]  = df["class"].astype(str).str.strip().str.lower()
    df["system"] = df["system"].astype(str).str.strip().str.lower()

    # Drop empty text rows
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    # Add canonical fields
    df.insert(0, "req_id", [f"raw_{i:04d}" for i in range(len(df))])
    df["source"] = "raw"

    # Reorder canonical schema
    df = df[["req_id", "text", "system", "class", "source"]]

    # Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    manifest = {
        "raw_input": str(RAW_PATH),
        "raw_sha256": sha256_file(RAW_PATH),
        "frozen_output": str(OUT_CSV),
        "n_rows_raw": int(pd.read_csv(RAW_PATH).shape[0]),
        "n_rows_frozen": int(df.shape[0]),
        "schema_raw": ["text", "class", "system"],
        "schema_frozen": ["req_id", "text", "system", "class", "source"],
        "notes": [
            "Only deterministic cleaning (strip + lower labels + collapse whitespace).",
            "No semantic changes, no deduplication, no merging."
        ]
    }
    OUT_MAN.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[OK] wrote", OUT_CSV)
    print("[OK] wrote", OUT_MAN)

if __name__ == "__main__":
    main()
