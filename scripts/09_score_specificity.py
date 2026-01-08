#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Specificity scoring (strict, defensible) — PER-SYSTEM EDITION (Expanded Lexicon + Validation)

Core idea:
  - L: Linguistic/testability cues (anchors) minus vagueness penalties, clamped to [0,1]
  - M: Semantic separability margin using per-system banks (SB vs VB) with shrink-to-neutral guard
  - Fuse (L, M) per system using Entropy Weight Method (EWM) + optional shrink/caps for stability

Input:
  resolved_requirements.csv with columns:
    req_id, text, system, class, source

Outputs:
  results/specificity/specificity_scored.csv
  results/specificity/specificity_meta.json
  results/specificity/banks/<system>/{vague_audit.csv,specific_audit.csv}
"""

import os, re, json, math, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# -----------------------------
# Expanded Lexicons
# -----------------------------
STRONG_MODALS = {"shall", "must"}
MEDIUM_MODALS = {"will"}
WEAK_MODALS   = {"should", "may", "could", "might", "can"}

VAGUE_TERMS = {
    "easy","easily","simple","simply","fast","faster","quick","quickly","slow",
    "flexible","scalable","reliable","robust","modern","intuitive","seamless",
    "efficient","efficiently","user-friendly","user friendly","appropriate",
    "adequate","sufficient","reasonable","normal","common","typical","standard",
    "generic","improved","better","best","optimal","convenient","timely","suitable",
    "soon","often","usually","frequently","sometimes","occasionally","periodically",
    "regularly","as needed","from time to time","when necessary",
    "etc.","and so on","if possible","as required","as appropriate","depending",
    "variable","unspecified"
}

EXPLICIT_QUANTIFIERS = {
    "at least","at most","no more than","no less than","not more than","not less than",
    "greater than","less than","equal to","exactly","not to exceed",
    "within","between","from","to","up to","minimum","maximum","bounded by","capped at",
    "ratio of","proportion of","fraction of","percentage of"
}

UNITS_KPI = {
    "ms","msec","s","sec","secs","second","seconds","minute","minutes","min",
    "hour","hours","hr","hrs","day","days","week","weeks","month","months",
    "year","years","cycle","bit","bits","kb","mb","gb","tb","byte","bytes",
    "mbps","gbps","iops","bandwidth","latency","response time","throughput",
    "transactions per second","tps","queries per second","qps","rps","fps",
    "uptime","availability","downtime","mttr","mtbf","sla","p95","p99",
    "failure rate","error rate","defect density","accuracy","precision","recall","f1",
    "coverage","completeness","correctness","key","keys","entropy","hash","digest",
    "salt","token length","cpu","memory","disk","storage","network","cache"
}

STANDARDS = {
    "aes-128","aes-256","tls 1.2","tls1.2","tls 1.3","tls1.3","mtls","m tls",
    "iso 27001","iso 27017","nist 800-53","owasp","pci-dss","hipaa","gdpr",
    "soc 2","fedramp","cis benchmark","fips 140-2","saml 2.0","oauth 2.0",
    "oauth 2.1","openid connect","oidc","wcag 2.0","wcag 2.1","wcag 2.2",
    "section 508","ada compliance","sox","sarbanes-oxley","basel iii","ferpa",
    "chrome","firefox","safari","edge","ios","android"
}

ROLE_SPECIFIC = {
    "system administrator","administrator","admin","supervisor","operator","auditor",
    "merchant","customer","cardmember","player","organizer","adjuster","estimator",
    "realtor","nursing staff","program administrator","student","instructor","teacher",
    "physician","doctor","nurse","patient","technician","engineer","dispatcher","buyer",
    "seller","manager","inspector","developer","tester"
}

GENERIC_ACTORS = {"user","client","end-user","end user","system","application","service","device","entity","component"}

ACTION_VERBS = {
    "create","record","store","update","delete","view","display","notify","search",
    "filter","edit","approve","assign","send","receive","transmit","upload","download",
    "import","export","log","encrypt","decrypt","hash","validate","parse","authenticate",
    "generate","calculate","compute","retry","backup","restore","failover","throttle",
    "cache","schedule","trigger","start","stop","enroll","grade","proctor","diagnose",
    "prescribe","charge","invoice","book","reserve"
}

COMPARATIVES = {"better","faster","higher","lower","greater","improved","more","less"}
OPEN_QUANTIFIERS = {"some","many","often","usually","frequently","commonly","various","different","multiple"}
STD_WEIGHT = 1.0

NUM_REGEX = re.compile(
    r"\b\d+(?:\.\d+)?%?\b"
    r"|(?:\d+x){1,}\d+\b"
    r"|\b\d+\s*-\s*\d+\b"
    r"|\b\d+\s*(?:ms|msec|s|sec|secs?|mins?|minutes?|hours?|hrs?)\b",
    flags=re.IGNORECASE
)

TOKEN_SPLIT = re.compile(r"[^\w\-\./]+")
FR_STRUCT_RE = re.compile(r"\ballow(?:s|ed)?\b.*\bto\b", re.IGNORECASE)

# -----------------------------
# Helper functions
# -----------------------------
def normalize_text(s): return str(s).strip().lower()
def tokenize(s): return [t for t in TOKEN_SPLIT.split(normalize_text(s)) if t]

def phrase_present(text, phrase_set): return any(p in text for p in phrase_set)
def phrase_count(text, phrase_set): return sum(text.count(p) for p in phrase_set)
def count_numbers(text): return len(NUM_REGEX.findall(normalize_text(text)))

def kpi_near_number(tokens, window=3):
    # detect token(s) that are numbers or number+unit
    num_idx = [i for i, t in enumerate(tokens) if NUM_REGEX.fullmatch(t or "") is not None]
    kpi_one = {w for w in UNITS_KPI if " " not in w}
    kpi_idx = [i for i, t in enumerate(tokens) if t in kpi_one]
    return any(abs(i - j) <= window for i in num_idx for j in kpi_idx)

def vague_hits_without_qualifier(tokens, window=3):
    txt = " ".join(tokens)
    vague_one = {w for w in VAGUE_TERMS if " " not in w}
    vague_pos = [i for i, t in enumerate(tokens) if t in vague_one]
    multi_hits = sum(1 for w in VAGUE_TERMS if " " in w and w in txt)

    num_idx = [i for i, t in enumerate(tokens) if NUM_REGEX.fullmatch(t or "") is not None]
    kpi_one = {w for w in UNITS_KPI if " " not in w}
    kpi_idx = [i for i, t in enumerate(tokens) if t in kpi_one]

    def qualified(i): return any(abs(i - j) <= window for j in num_idx + kpi_idx)
    penalized = sum(1 for i in vague_pos if not qualified(i))
    if multi_hits > 0 and len(num_idx) == 0:
        penalized += multi_hits
    return penalized

def comparatives_unanchored(text, tokens, window=5):
    comp_one = {w for w in COMPARATIVES if " " not in w}
    comp_idx = [i for i, t in enumerate(tokens) if t in comp_one]
    txt = " ".join(tokens)
    if " than " in f" {txt} ":
        return False

    num_idx = [i for i, t in enumerate(tokens) if NUM_REGEX.fullmatch(t or "") is not None]
    kpi_one = {w for w in UNITS_KPI if " " not in w}
    kpi_idx = [i for i, t in enumerate(tokens) if t in kpi_one]

    for i in comp_idx:
        if not any(abs(i - j) <= window for j in num_idx + kpi_idx):
            return True
    if (any(p in txt for p in COMPARATIVES)) and not num_idx and not kpi_idx:
        return True
    return False

def open_quantifier_present(text): return any(f" {w} " in f" {text} " for w in OPEN_QUANTIFIERS)

def detect_modality(tokens):
    s = set(tokens)
    if any(t in STRONG_MODALS for t in s): return 1.0
    if any(t in MEDIUM_MODALS for t in s): return 0.75
    if any(t in WEAK_MODALS for t in s): return 0.5
    return 0.0

def actor_specificity(text, system_has_multi_roles):
    txt = normalize_text(text)
    for p in ROLE_SPECIFIC:
        if p in txt: return 1.0
    for p in GENERIC_ACTORS:
        if p in txt: return 0.5 if not system_has_multi_roles else 0.0
    return 0.0

def functional_structure_bonus(text):
    t = normalize_text(text); bonus = 0.0
    if FR_STRUCT_RE.search(t): bonus += 0.5
    if any(f" {v} " in f" {t} " for v in ACTION_VERBS): bonus += 0.25
    return min(bonus, 0.75)

# -----------------------------
# EWM + stabilization
# -----------------------------
def entropy_weight_method(matrix_2d, eps=1e-12):
    X = np.array(matrix_2d, dtype=float)
    if X.ndim != 2:
        raise ValueError("entropy_weight_method expects 2D matrix")
    n, m = X.shape
    if n < 2:
        return np.ones(m) / m

    # avoid zeros
    X = np.clip(X, eps, None)
    colsum = X.sum(axis=0, keepdims=True)
    P = X / np.clip(colsum, eps, None)

    k = 1.0 / math.log(max(n, 2))
    E = -k * np.sum(P * np.log(np.clip(P, eps, None)), axis=0)
    d = 1.0 - E
    if float(d.sum()) <= eps:
        return np.ones(m) / m
    return d / d.sum()

def shrink_cap_weights(w, n_eff, kappa=1.5, wmin=0.20, wmax=0.80):
    # shrink toward uniform for small n, then cap and renormalize
    w = np.array(w, dtype=float)
    m = w.size
    u = np.ones(m) / m
    lam = min(1.0, kappa / max(np.sqrt(max(n_eff, 1)), 1.0))
    w = lam * u + (1.0 - lam) * w
    w = np.clip(w, wmin, wmax)
    w = w / np.clip(w.sum(), 1e-12, None)
    return w

# -----------------------------
# Bank rules
# -----------------------------
def vb_rule(row):
    any_vague_signal = (
        row["vague_lex_hits"] > 0 or row["weak_modal"] > 0 or
        row["comparative_unanchored"] or row["open_quantifier"]
    )
    no_anchors = (
        row["num_present"] == 0 and row["unit_present"] == 0 and
        row["kpi_near"] == 0 and row["std_present"] == 0
    )
    return any_vague_signal and no_anchors and (row["L"] <= 0.45)

def sb_rule(row):
    conds = 0
    if (row["num_present"] == 1) or (row["unit_present"] == 1):
        conds += 1
    if (row["kpi_near"] == 1) or (row["std_present"] == 1):
        conds += 1
    if (row["actor_specific"] == 1) or (row["strong_modal"] == 1):
        conds += 1
    no_vague = (row["vague_lex_hits"] == 0 and row["weak_modal"] == 0
                and not row["comparative_unanchored"] and not row["open_quantifier"])
    return (conds >= 2) and no_vague and (row["L"] >= 0.70)

# -----------------------------
# Validation helpers
# -----------------------------
def validate_system(sys_name, df_part, meta):
    print(f"\n=== Validation for system: {sys_name} ===")
    VB_size, SB_size = meta["VB_size"], meta["SB_size"]
    w_L, w_M = meta["w_L"], meta["w_M"]

    if VB_size == 0: print("⚠️ Warning: VB_size = 0")
    if SB_size == 0: print("⚠️ Warning: SB_size = 0")

    if w_L < 0 or w_M < 0:
        print(f"⚠️ Negative weight detected: w_L={w_L}, w_M={w_M}")
    if not (0.999 <= (w_L + w_M) <= 1.001):
        print(f"⚠️ Weights don’t sum to 1: w_L+w_M={w_L+w_M:.6f}")

    print("Top 3 most specific:")
    print(df_part.sort_values("S_final_sys", ascending=False).head(3)[["req_id","S_final_sys","text"]].to_string(index=False))
    print("\nTop 3 most vague:")
    print(df_part.sort_values("S_final_sys", ascending=True).head(3)[["req_id","S_final_sys","text"]].to_string(index=False))
    return df_part

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/redundancy/resolved_requirements.csv")
    ap.add_argument("--model", default="intfloat/e5-large")
    ap.add_argument("--tau_cap", type=float, default=5.0)
    ap.add_argument("--banks_dir", default="results/specificity/banks")
    ap.add_argument("--out_scores", default="results/specificity/specificity_scored.csv")
    ap.add_argument("--out_meta", default="results/specificity/specificity_meta.json")

    # stabilization for EWM
    ap.add_argument("--kappa", type=float, default=1.5)
    ap.add_argument("--wmin", type=float, default=0.20)
    ap.add_argument("--wmax", type=float, default=0.80)

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_scores), exist_ok=True)
    os.makedirs(args.banks_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ["req_id","text","system","class","source"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # detect whether system includes multiple specific roles (affects generic actor penalty)
    sys_multi_role = {}
    for sys_name, g in df.groupby("system"):
        texts = " ".join(normalize_text(x) for x in g["text"].astype(str))
        sys_multi_role[sys_name] = any(p in texts for p in ROLE_SPECIFIC)

    # -------- L (linguistic) --------
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Linguistic cues (L)"):
        rid, sysn, txt = row["req_id"], row["system"], str(row["text"])
        low, tokens = normalize_text(txt), tokenize(txt)

        modal_val = detect_modality(tokens)
        strong_modal = 1 if modal_val == 1.0 else 0
        weak_modal = 1 if modal_val == 0.5 else 0

        n_numbers = count_numbers(txt)

        # units count: single-token units + multiword units (substring)
        kpi_one = {w for w in UNITS_KPI if " " not in w}
        n_units = sum(1 for t in tokens if t in kpi_one)
        token_join = " ".join(tokens)
        for w in UNITS_KPI:
            if " " in w and w in token_join:
                n_units += 1

        num_present, unit_present = int(n_numbers > 0), int(n_units > 0)
        kpi_near = int(kpi_near_number(tokens, window=3))
        std_present = int(phrase_present(low, STANDARDS))

        actor_score = actor_specificity(txt, sys_multi_role.get(sysn, False))
        actor_specific = int(actor_score >= 1.0)

        quant_count = phrase_count(low, EXPLICIT_QUANTIFIERS)
        vague_hits = vague_hits_without_qualifier(tokens, window=3)
        comp_un = comparatives_unanchored(low, tokens, window=5)
        open_q = open_quantifier_present(low)
        fr_bonus = functional_structure_bonus(txt)

        pos = (
            n_numbers +
            n_units +
            quant_count +
            modal_val +
            actor_score +
            (STD_WEIGHT * std_present) +
            fr_bonus
        )
        L_val = min(max(0.0, (pos - vague_hits)) / args.tau_cap, 1.0)

        rows.append({
            "req_id": rid, "system": sysn, "class": row["class"], "source": row["source"], "text": txt,
            "numbers_count": n_numbers,
            "units_count": n_units,
            "explicit_quantifiers": quant_count,
            "modal_score": modal_val,
            "strong_modal": strong_modal,
            "weak_modal": weak_modal,
            "actor_score": actor_score,
            "actor_specific": actor_specific,
            "vague_lex_hits": vague_hits,
            "comparative_unanchored": comp_un,
            "open_quantifier": open_q,
            "num_present": num_present,
            "unit_present": unit_present,
            "kpi_near": kpi_near,
            "std_present": std_present,
            "cue_score": pos - vague_hits,
            "L": L_val
        })

    Ldf = pd.DataFrame(rows)

    # -------- embeddings once --------
    model = SentenceTransformer(args.model)
    all_vecs = model.encode(Ldf["text"].tolist(), batch_size=16, convert_to_numpy=True, show_progress_bar=True)
    Ldf["_row_idx"] = np.arange(len(Ldf))

    # init output columns
    out_cols_extra = [
        "sim_spec_sys","sim_vague_sys","M_raw_sys","M_sys","w_L_sys","w_M_sys",
        "bank_Q_sys","bank_shrink_a_sys","VB_size_sys","SB_size_sys","S_final_sys"
    ]
    for col in out_cols_extra:
        Ldf[col] = np.nan

    per_sys_meta = {}

    # -------- per-system M and fusion --------
    for sys_name, part in Ldf.groupby("system", sort=False):
        vecs = all_vecs[part["_row_idx"].to_numpy()]

        VB = part[part.apply(vb_rule, axis=1)]
        SB = part[part.apply(sb_rule, axis=1)]

        # fallback expansions (IMPORTANT: do both sides)
        if len(VB) < 10:
            VB = part.sort_values("L", ascending=True).head(min(30, len(part)))
        if len(SB) < 10:
            SB = part.sort_values("L", ascending=False).head(min(30, len(part)))

        # audit output
        sys_dir = os.path.join(args.banks_dir, sys_name)
        os.makedirs(sys_dir, exist_ok=True)
        VB.to_csv(os.path.join(sys_dir, "vague_audit.csv"), index=False)
        SB.to_csv(os.path.join(sys_dir, "specific_audit.csv"), index=False)

        def centroid(v): return np.mean(v, axis=0, dtype=np.float32) if v.shape[0] > 0 else None

        def avg_intra_sim(v):
            if v.shape[0] <= 1:
                return 1.0
            S = cosine_similarity(v)
            return float((S.sum() - v.shape[0]) / (v.shape[0] * (v.shape[0] - 1)))

        def avg_inter_sim(A, B):
            if A.shape[0] == 0 or B.shape[0] == 0:
                return 0.0
            return float(np.mean(cosine_similarity(A, B)))

        vb_pos = part.index.get_indexer(VB.index)
        sb_pos = part.index.get_indexer(SB.index)
        VB_vecs = vecs[vb_pos] if len(vb_pos) > 0 else np.zeros((0, vecs.shape[1]), dtype=np.float32)
        SB_vecs = vecs[sb_pos] if len(sb_pos) > 0 else np.zeros((0, vecs.shape[1]), dtype=np.float32)

        if VB_vecs.shape[0] < 2 or SB_vecs.shape[0] < 2:
            Q, a = None, 0.0
            simS = np.zeros(len(part), dtype=float)
            simV = np.zeros(len(part), dtype=float)
            M_raw = np.full(len(part), 0.5, dtype=float)
            M = np.full(len(part), 0.5, dtype=float)
        else:
            c_VB = centroid(VB_vecs)
            c_SB = centroid(SB_vecs)
            intra_V = avg_intra_sim(VB_vecs)
            intra_S = avg_intra_sim(SB_vecs)
            inter_VS = avg_inter_sim(VB_vecs, SB_vecs)

            Q = None if inter_VS <= 1e-12 else ((intra_V + intra_S) / 2.0) / inter_VS
            # map Q into shrink strength a in [0,1]
            a = 0.0 if Q is None else max(0.0, min(1.0, (Q - 1.0) / 0.5))

            simS = cosine_similarity(vecs, c_SB.reshape(1, -1)).ravel()
            simV = cosine_similarity(vecs, c_VB.reshape(1, -1)).ravel()

            margin = (simS - simV + 1.0) / 2.0  # [0,1]
            local = np.where(np.abs(simS - simV) < 0.05, 0.5, 1.0)  # soften ambiguous ones
            M_raw = margin
            M = 0.5 + (margin - 0.5) * (a * local)

        indicators = np.column_stack([part["L"].to_numpy(dtype=float), M.astype(float)])
        w = entropy_weight_method(indicators)
        w = shrink_cap_weights(w, n_eff=len(part), kappa=args.kappa, wmin=args.wmin, wmax=args.wmax)
        w_L, w_M = float(w[0]), float(w[1])

        S_sys = (w_L * part["L"].to_numpy(dtype=float) + w_M * M).astype(float)
        S_sys = np.clip(S_sys, 0.0, 1.0)

        # write back
        Ldf.loc[part.index, "sim_spec_sys"] = simS
        Ldf.loc[part.index, "sim_vague_sys"] = simV
        Ldf.loc[part.index, "M_raw_sys"] = M_raw
        Ldf.loc[part.index, "M_sys"] = M
        Ldf.loc[part.index, "w_L_sys"] = w_L
        Ldf.loc[part.index, "w_M_sys"] = w_M
        Ldf.loc[part.index, "bank_Q_sys"] = (np.nan if Q is None else float(Q))
        Ldf.loc[part.index, "bank_shrink_a_sys"] = float(a)
        Ldf.loc[part.index, "VB_size_sys"] = int(len(VB))
        Ldf.loc[part.index, "SB_size_sys"] = int(len(SB))
        Ldf.loc[part.index, "S_final_sys"] = S_sys

        per_sys_meta[sys_name] = {
            "VB_size": int(len(VB)),
            "SB_size": int(len(SB)),
            "Q": (None if Q is None else float(Q)),
            "shrink_a": float(a),
            "w_L": w_L,
            "w_M": w_M
        }

        # print validation summary (no mutation needed here)
        _ = validate_system(sys_name, Ldf.loc[part.index], per_sys_meta[sys_name])

    # final output
    out_cols = [
        "req_id","system","class","source","text",
        "numbers_count","units_count","explicit_quantifiers",
        "modal_score","strong_modal","weak_modal",
        "actor_score","actor_specific",
        "vague_lex_hits","comparative_unanchored","open_quantifier",
        "num_present","unit_present","kpi_near","std_present",
        "cue_score","L",
        "sim_spec_sys","sim_vague_sys","M_raw_sys","M_sys",
        "w_L_sys","w_M_sys",
        "bank_Q_sys","bank_shrink_a_sys","VB_size_sys","SB_size_sys",
        "S_final_sys"
    ]

    Ldf[out_cols].to_csv(args.out_scores, index=False)

    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump({
            "input": args.input,
            "model": args.model,
            "tau_cap": args.tau_cap,
            "ewm_stabilization": {"kappa": args.kappa, "wmin": args.wmin, "wmax": args.wmax},
            "per_system": per_sys_meta,
            "banks_root": os.path.abspath(args.banks_dir)
        }, f, indent=2)

if __name__ == "__main__":
    main()
