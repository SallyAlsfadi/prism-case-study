#!/usr/bin/env python3
# scripts/07_apply_decisions.py
#
# Apply stakeholder merge decisions to produce resolved requirements (single-system ecommerce case study).
# This is the SAME logic as the Sep_version apply_decisions.py, adapted to this repo layout.

from __future__ import annotations
import os
import argparse
import glob
import pandas as pd


def infer_system_class_from_reqid(req_id: str) -> tuple[str, str]:
    """
    Parse {system}_{class}_{nnn} where system may contain underscores.
    Example: field_service_us_001 -> ("field_service","us")
    """
    parts = str(req_id).split("_")
    if len(parts) >= 3:
        system = "_".join(parts[:-2])
        req_class = parts[-2]
        return system, req_class
    return "", ""


def load_master_texts(emb_dir: str) -> pd.DataFrame:
    """Load all *_texts.csv to map req_id -> text."""
    rows = []
    files = sorted(glob.glob(os.path.join(emb_dir, "*_texts.csv")))
    if not files:
        raise FileNotFoundError(f"No *_texts.csv found under {emb_dir}")

    for fp in files:
        df = pd.read_csv(fp)
        df.columns = df.columns.str.lower().str.strip()
        if not {"req_id", "text"}.issubset(df.columns):
            continue

        sys_cls = df["req_id"].apply(lambda x: pd.Series(infer_system_class_from_reqid(x)))
        sys_cls.columns = ["system", "class"]
        df = pd.concat([df[["req_id", "text"]], sys_cls], axis=1)
        rows.append(df)

    if not rows:
        raise FileNotFoundError(f"No valid *_texts.csv with columns req_id,text under {emb_dir}")

    master = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["req_id"])
    return master


def expand_members(member_ids_str: str) -> list[str]:
    return [x.strip() for x in str(member_ids_str).split("|") if x.strip()]


def apply_row(dec: pd.Series, master_map: dict[str, str]) -> list[dict]:
    """
    Returns list of dict rows for final output given one decision row.
    Always computes system/class from the resulting req_id(s).
    """
    out = []
    members = expand_members(dec["member_ids"])
    decision = str(dec.get("decision", "")).strip().lower()
    rep_id = dec["suggested_representative_id"]
    rep_text = dec["suggested_representative_text"]

    if decision in ("accept_merge", "a", "accept"):
        sys_, cls_ = infer_system_class_from_reqid(rep_id)
        out.append(
            {"req_id": rep_id, "text": rep_text, "system": sys_, "class": cls_, "source": "merge_accept"}
        )

    elif decision in ("keep_originals", "k", "keep", ""):
        for mid in members:
            sys_, cls_ = infer_system_class_from_reqid(mid)
            out.append(
                {"req_id": mid, "text": master_map.get(mid, ""), "system": sys_, "class": cls_, "source": "keep_originals"}
            )

    elif decision in ("pick_one", "p", "pick"):
        pick_id = str(dec.get("pick_one_id", "")).strip()
        if pick_id and pick_id in members:
            sys_, cls_ = infer_system_class_from_reqid(pick_id)
            out.append(
                {"req_id": pick_id, "text": master_map.get(pick_id, ""), "system": sys_, "class": cls_, "source": "pick_one"}
            )
        else:
            # fallback: keep originals
            for mid in members:
                sys_, cls_ = infer_system_class_from_reqid(mid)
                out.append(
                    {"req_id": mid, "text": master_map.get(mid, ""), "system": sys_, "class": cls_, "source": "keep_originals"}
                )

    elif decision in ("edit", "e"):
        edited = str(dec.get("edited_text", "")).strip()
        final_text = edited if edited else rep_text
        sys_, cls_ = infer_system_class_from_reqid(rep_id)
        out.append(
            {"req_id": rep_id, "text": final_text, "system": sys_, "class": cls_, "source": "merge_edit"}
        )

    else:
        # unknown decision → keep originals
        for mid in members:
            sys_, cls_ = infer_system_class_from_reqid(mid)
            out.append(
                {"req_id": mid, "text": master_map.get(mid, ""), "system": sys_, "class": cls_, "source": "keep_originals"}
            )

    return out


def main():
    p = argparse.ArgumentParser(description="Step 07: Apply redundancy decisions to produce resolved requirements.")
    p.add_argument(
        "--emb_dir",
        default="results_rerun/embeddings/domain_class",
        help="Dir with *_texts.csv (req_id,text). Default: results_rerun/embeddings/domain_class",
    )
    p.add_argument(
        "--decisions_csv",
        default="results_rerun/redundancy/decisions.csv",
        help="Decisions CSV. Default: results_rerun/redundancy/decisions.csv",
    )
    p.add_argument(
        "--out_csv",
        default="results_rerun/redundancy/resolved_requirements.csv",
        help="Output resolved CSV. Default: results_rerun/redundancy/resolved_requirements.csv",
    )
    p.add_argument(
        "--audit_csv",
        default="results_rerun/redundancy/_apply_audit.csv",
        help="Audit CSV. Default: results_rerun/redundancy/_apply_audit.csv",
    )
    args = p.parse_args()

    master = load_master_texts(args.emb_dir)
    master_map = dict(zip(master["req_id"], master["text"]))
    all_reqs = set(master["req_id"])

    dec = pd.read_csv(args.decisions_csv)
    dec.columns = dec.columns.str.lower().str.strip()

    # make sure decision columns exist (robust)
    for col in ["decision", "pick_one_id", "edited_text", "comment"]:
        if col not in dec.columns:
            dec[col] = ""

    final_rows: list[dict] = []
    seen = set()

    # Remove members from baseline; they'll be re-added per decision.
    for _, row in dec.iterrows():
        members = set(expand_members(row["member_ids"]))
        all_reqs -= members

        for r in apply_row(row, master_map):
            if r["req_id"] in seen:
                continue
            final_rows.append(r)
            seen.add(r["req_id"])

    # Add all untouched originals (not part of any proposal)
    untouched = master[master["req_id"].isin(sorted(all_reqs))].copy()
    if not untouched.empty:
        untouched = untouched.assign(source="unchanged")
        final_rows.extend(untouched.to_dict(orient="records"))

    final_df = pd.DataFrame(final_rows).drop_duplicates(subset=["req_id"])
    final_df = final_df[["req_id", "text", "system", "class", "source"]].sort_values(["system", "class", "req_id"])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    final_df.to_csv(args.out_csv, index=False)

    # Audit
    audit_rows = []
    for _, row in dec.iterrows():
        audit_rows.append(
            {
                "proposal_id": row.get("proposal_id", ""),
                "decision": row.get("decision", ""),
                "member_ids": row.get("member_ids", ""),
                "pick_one_id": row.get("pick_one_id", ""),
                "edited_text_len": len(str(row.get("edited_text", ""))),
            }
        )
    pd.DataFrame(audit_rows).to_csv(args.audit_csv, index=False)

    print(f"Resolved requirements written → {args.out_csv}  (rows={len(final_df)})")
    print(f"Audit written → {args.audit_csv}")

    # quick sanity print
    raw_n = len(master)
    res_n = len(final_df)
    print(f"Sanity: raw reqs={raw_n} → resolved reqs={res_n} (reduction={raw_n - res_n})")


if __name__ == "__main__":
    main()
