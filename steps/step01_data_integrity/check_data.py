import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text())

    data_csv = Path(cfg["paths"]["data_csv"])
    logs_dir = Path(cfg["paths"]["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    required_cols = set(cfg["data_expectations"]["required_columns"])
    min_rows = int(cfg["data_expectations"]["min_rows"])

    if not data_csv.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {data_csv}. "
            f"Place requirements.csv at {data_csv} (see data/README.md)."
        )

    file_hash = sha256_file(data_csv)
    df = pd.read_csv(data_csv)

    cols = list(df.columns)
    missing_required = sorted(list(required_cols - set(cols)))
    n_rows = int(df.shape[0])
    n_cols = int(df.shape[1])
    n_missing_total = int(df.isna().sum().sum())

    dup_text_total = int(df.duplicated(subset=["text"]).sum()) if "text" in df.columns else None
    dup_text_within_system = None
    if "text" in df.columns and "system" in df.columns:
        dup_text_within_system = int(df.duplicated(subset=["system", "text"]).sum())

    system_counts = df["system"].value_counts().to_dict() if "system" in df.columns else {}
    class_counts = df["class"].value_counts().to_dict() if "class" in df.columns else {}

    schema_ok = (len(missing_required) == 0) and (n_rows >= min_rows)

    report = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "config_path": str(config_path),
        "dataset_path": str(data_csv),
        "dataset_hash_sha256": file_hash,
        "shape": {"rows": n_rows, "cols": n_cols},
        "columns": cols,
        "missing_required_columns": missing_required,
        "missing_values_total": n_missing_total,
        "duplicates_exact_text_total": dup_text_total,
        "duplicates_exact_text_within_system": dup_text_within_system,
        "system_counts": system_counts,
        "class_counts": class_counts,
        "schema_ok": schema_ok,
    }

    report_json_path = Path(cfg["integrity"]["report_json"])
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    report_md_path = Path(cfg["integrity"]["report_md"])
    md = []
    md.append("# Step 1: Dataset Integrity Report\n")
    md.append(f"- Generated (UTC): **{report['generated_at_utc']}**")
    md.append(f"- Dataset: **{report['dataset_path']}**")
    md.append(f"- SHA-256: `{report['dataset_hash_sha256']}`\n")
    md.append("## Schema\n")
    md.append(f"- Rows: **{n_rows}**")
    md.append(f"- Columns: **{n_cols}**")
    md.append(f"- Required columns: **{sorted(list(required_cols))}**")
    md.append(f"- Missing required columns: **{missing_required}**")
    md.append(f"- Total missing values: **{n_missing_total}**")
    md.append(f"- Schema OK: **{schema_ok}**\n")
    md.append("## Duplicates (exact text)\n")
    md.append(f"- Exact duplicates across dataset (by `text`): **{dup_text_total}**")
    md.append(f"- Exact duplicates within the same system (`system`+`text`): **{dup_text_within_system}**\n")
    md.append("## System counts\n")
    if system_counts:
        for k, v in system_counts.items():
            md.append(f"- {k}: {v}")
    else:
        md.append("- (no `system` column)")
    md.append("\n## Class counts\n")
    if class_counts:
        for k, v in class_counts.items():
            md.append(f"- {k}: {v}")
    else:
        md.append("- (no `class` column)")
    md.append("")

    report_md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"[OK] Wrote: {report_json_path}")
    print(f"[OK] Wrote: {report_md_path}")
    if not schema_ok:
        raise ValueError("Schema checks failed. See report for details.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(Path(args.config))
