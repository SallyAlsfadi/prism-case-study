#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Volatility scoring (V1..V4) — PER-(SYSTEM, CLASS) with on-the-fly embeddings

Fixes vs previous version:
  1) measurable_present(): no longer treats any digit/version/date as “measurable”
  2) embed_on_the_fly(): correct device handling for SentenceTransformer
  3) eps_span default aligned with PRISM guards (0.05)
"""

import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# ------------- Defaults -------------
DEFAULTS = dict(
    k=5,
    alpha=0.5,
    alpha_small_n=0.7,
    lambda_v2=0.6,
    gamma_shrink=0.2,
    w_min=0.10,
    w_max=0.50,
    eps_span=0.05,      # ✅ aligned with Criticality/S guards
    eps=1e-12,
    corr_thresh=0.80,
    corr_damp=0.85,
    legal_cap=0.20,
    measurable_override=0.50,
    redundancy_sim_thresh=0.90,
    redundancy_discount=0.80,
    batch_size=64,
    device="auto"
)

# ---------- Cue banks ----------
HEDGE_PHRASES = [
    "as appropriate","as needed","if possible","where feasible",
    "to the extent possible","in general","in principle","by default"
]
HEDGE_TOKENS = [
    "may","might","could","should","would",
    "typically","generally","usually","often",
    "likely","possibly","probably",
    "approximately","roughly","around","about","nearly","almost",
    "ideally","preferably","optionally"
]
COND_TOKENS = [
    "if","unless","when","whenever","provided","assuming","in case",
    "contingent","subject","dependent","only","as long as","before","after"
]
COND_PHRASES = [
    "provided that","subject to","in case","only if","as long as","contingent on","dependent on"
]
TEMPORAL_VAGUE = [
    "soon","eventually","later","future","periodic","periodically",
    "from time to time","as soon as possible","asap","ongoing","at a later stage","in phases"
]
OPEN_ENDED_TAILS = [
    "etc.","etc","and so on","and others","including but not limited to","among others"
]
OPEN_ENDED_QUANT = [
    "as needed","as appropriate","where applicable","any","all","multiple","several","many","unlimited"
]
LEGAL_BOILER = [
    "subject to applicable law","subject to law","subject to regulatory approval"
]

# ---- Regexes (revised measurable detection) ----
RE_MONTH_MAY = re.compile(r'\bMay\s+\d{1,2}\b')  # avoid “May 12” as hedging
RE_DEFAULT_VALUE = re.compile(r'\bdefault\s*=\s*\w+', re.IGNORECASE)
RE_EXPLICIT_VERSION_DATE = re.compile(r'\bv\d+(\.\d+)*\b|\b20\d{2}-\d{2}-\d{2}\b')
RE_PLACEHOLDER = re.compile(
    r'\bT-?B-?D\b|\bto be determined\b|<[^>]+>|\bN\s+\w+\b|\bX\s+\w+\b|\bY\s+\w+\b|\bZ\s+\w+\b|\.\.\.$',
    re.IGNORECASE
)
CLAUSE_SPLIT = re.compile(r'[.;:?!]')

# “Measurable KPI” = number + (unit/kpi context) OR explicit comparator pattern
RE_KPI_MEASURABLE = re.compile(
    r'(\b\d+(\.\d+)?\s*(%|ms|msec|s|sec|secs|min|mins|minute|minutes|hour|hours|day|days)\b)'
    r'|(\b\d+(\.\d+)?\s*(kb|mb|gb|tb|kbps|mbps|gbps|rps|req/s|tps|qps|fps)\b)'
    r'|(\b(p95|p99|sla|uptime|availability|latency|response time|throughput|error rate|failure rate)\b.*\b\d+(\.\d+)?%?\b)'
    r'|(\b(>=|<=|>|<|between)\b.*\b\d+(\.\d+)?\b)',
    re.IGNORECASE
)

# ---------- Utils ----------
def safe_div(a, b, eps=1e-12): return a/(b+eps)
def text_lower(s): return (s or "").strip().lower()
def token_split(s): return re.findall(r'[a-zA-Z0-9%\/\+\-\._]+', (s or "").lower())

def count_multiword(text, phrases): return sum(text.count(p) for p in phrases if p in text)
def contains_any_phrase(text, phrases): return any(p in text for p in phrases)
def count_tokens_in(text, vocab): return sum(1 for t in token_split(text) if t in vocab)

def measurable_present(text: str) -> bool:
    """
    True only when we see KPI-like measurable content (units/comparators/KPI words).
    Ignores dates/versions by design.
    """
    t = (text or "")
    if RE_EXPLICIT_VERSION_DATE.search(t):
        # version/date present is NOT evidence of measurability
        pass
    return bool(RE_KPI_MEASURABLE.search(t))

def legal_boilerplate_strength(text): return 1.0 if contains_any_phrase(text, LEGAL_BOILER) else 0.0

def normalize_minmax(arr, eps_span=0.05):
    a_min = np.nanmin(arr); a_max = np.nanmax(arr); span = a_max - a_min
    if span < eps_span: return np.full_like(arr, 0.5, dtype=float), True
    out = np.clip((arr - a_min) / (span + 0.0), 0.0, 1.0)
    return out, False

def unit_normalize_rows(X, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return X / norms

# ---------- Components V1..V3 ----------
def v1_hedging_density(text):
    t0 = text or ""
    t_guard = RE_MONTH_MAY.sub("", t0).lower()
    mw_hits = count_multiword(t_guard, HEDGE_PHRASES)
    tok_hits = count_tokens_in(t_guard, HEDGE_TOKENS)
    dens = safe_div(mw_hits + tok_hits, max(1, len(token_split(t_guard))))
    if RE_DEFAULT_VALUE.search(t0): dens *= 0.5
    return float(np.clip(dens, 0.0, 1.0))

def v2_conditional_temporal(text, lam=0.6, legal_cap=0.20):
    t = text_lower(text)
    cond_count = count_multiword(t, COND_PHRASES) + count_tokens_in(t, COND_TOKENS)
    temp_count = sum(1 for cue in TEMPORAL_VAGUE if cue in t)
    dC = safe_div(cond_count, max(1, len(token_split(t))))
    dT = safe_div(temp_count, max(1, len(token_split(t))))
    v2 = lam*dC + (1-lam)*dT
    if legal_boilerplate_strength(t) > 0: v2 = min(v2, legal_cap)
    if RE_EXPLICIT_VERSION_DATE.search(text or ""): v2 *= 0.7
    return float(np.clip(v2, 0.0, 1.0))

def v3_open_ended(text):
    t = text_lower(text)
    clauses = [c.strip() for c in CLAUSE_SPLIT.split(t) if c.strip()] or [t]
    def clause_open(c):
        if contains_any_phrase(c, OPEN_ENDED_TAILS): return True
        if contains_any_phrase(c, OPEN_ENDED_QUANT): return True
        if RE_PLACEHOLDER.search(c): return True
        # explicit bounds with KPI evidence should not be treated as open-ended
        if (("at least" in c) or ("at most" in c)) and measurable_present(c): return False
        return False
    hits = sum(1 for c in clauses if clause_open(c))
    return float(np.clip(safe_div(hits, len(clauses)), 0.0, 1.0))

def measurable_override_factor(text, factor=0.5):
    return factor if measurable_present(text or "") else 1.0

# ---------- V4 structure ----------
def compute_c1_c2(embeds, k=5, eps=1e-12):
    n = embeds.shape[0]
    if n == 0: return np.array([]), np.array([])
    centroid = unit_normalize_rows(np.mean(embeds, axis=0, keepdims=True), eps=eps)
    C1 = cosine_similarity(embeds, centroid).ravel()
    S = cosine_similarity(embeds); np.fill_diagonal(S, -np.inf)
    kk = min(k, max(1, n-1))
    topk_idx = np.argpartition(-S, kth=kk-1, axis=1)[:, :kk]
    rows = np.arange(n)[:, None]
    topk_vals = S[rows, topk_idx]
    C2 = np.mean(topk_vals, axis=1)
    C1 = np.clip((C1 + 1.0)/2.0, 0.0, 1.0)
    C2 = np.clip((C2 + 1.0)/2.0, 0.0, 1.0)
    return C1, C2

def redundancy_discount_flags(embeds, sim_thresh=0.90):
    n = embeds.shape[0]
    if n <= 1: return np.zeros(n, dtype=bool)
    S = cosine_similarity(embeds); np.fill_diagonal(S, 0.0)
    return (S >= sim_thresh).any(axis=1)

# ---------- EWM + correlation ----------
def entropy_weights(X_norm, gamma=0.2, w_min=0.10, w_max=0.50, eps=1e-12):
    n, m = X_norm.shape
    if n == 0: return np.full(m, 1.0/m)
    col_sums = X_norm.sum(axis=0) + eps
    P = X_norm / col_sums
    E = np.zeros(m)
    for k in range(m):
        pk = P[:, k] + eps
        E[k] = - (pk*np.log(pk)).sum() / np.log(max(n, 2))
    d = 1.0 - E
    w0 = np.full(m, 1.0/m) if d.sum() <= eps else d/(d.sum()+eps)
    w = (1-gamma)*w0 + gamma*(1.0/m)
    w = np.clip(w, w_min, w_max)
    w = w/(w.sum()+eps)
    return w

def correlation_dampen(X_cols, names, thresh=0.80, damp=0.85):
    names_list = list(names); applied=[]
    if len(names_list) < 2: return X_cols, applied
    M = np.vstack([X_cols[n] for n in names_list]).T
    if M.shape[0] < 3: return X_cols, applied
    df = pd.DataFrame(M, columns=names_list)
    corr = df.corr().values; var = df.var().values
    for i in range(len(names_list)):
        for j in range(i+1, len(names_list)):
            rho = corr[i, j]
            if abs(rho) >= thresh:
                target = names_list[i] if var[i] >= var[j] else names_list[j]
                X_cols[target] = np.clip(X_cols[target]*damp, 0.0, 1.0)
                applied.append({"pair":[names_list[i],names_list[j]],"rho":float(rho),"damped":target,"factor":damp})
    return X_cols, applied

# ---------- Embedding IO ----------
def parse_vec_cell(cell):
    cell = str(cell).strip()
    if cell.startswith("[") and cell.endswith("]"):
        try: return np.array(json.loads(cell), dtype=float)
        except Exception: pass
    parts = cell.replace(",", " ").split()
    return np.array([float(x) for x in parts], dtype=float)

def load_embeddings_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    if not {"req_id","vec"}.issubset(df.columns):
        raise ValueError("embeddings_csv must have columns: req_id, vec")
    id2vec = {}
    for _, r in df.iterrows():
        id2vec[str(r["req_id"])] = parse_vec_cell(r["vec"])
    return id2vec

def load_embeddings_from_npy(npy_path, id_map_csv):
    E = np.load(npy_path); m = pd.read_csv(id_map_csv)
    if not {"req_id","row_index"}.issubset(m.columns):
        raise ValueError("id_map_csv must have columns: req_id,row_index")
    m = m.sort_values("row_index")
    if E.shape[0] < len(m):
        raise ValueError("embeddings_npy rows < id_map entries")
    id2row = dict(zip(m["req_id"].astype(str), m["row_index"].astype(int)))
    return E, id2row

# ---------- On-the-fly embedding ----------
def embed_on_the_fly(texts, model_name, batch_size=64, device="auto", cache_path=None, req_ids=None):
    cached = {}
    if cache_path and os.path.exists(cache_path):
        dfc = pd.read_csv(cache_path)
        if {"req_id","vec"}.issubset(dfc.columns):
            for _, r in dfc.iterrows():
                cached[str(r["req_id"])] = parse_vec_cell(r["vec"])

    if req_ids is None:
        req_ids = [str(i) for i in range(len(texts))]

    to_embed_idx = [i for i, rid in enumerate(req_ids) if rid not in cached]

    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except Exception as e:
        raise RuntimeError("sentence-transformers (and torch) are required for --embed_mode onfly") from e

    device_resolved = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
    model = SentenceTransformer(model_name)
    model.to(device_resolved)

    if to_embed_idx:
        to_embed_texts = [texts[i] for i in to_embed_idx]
        vecs = []
        for j in range(0, len(to_embed_texts), batch_size):
            chunk = to_embed_texts[j:j+batch_size]
            v = model.encode(chunk, batch_size=len(chunk), show_progress_bar=False,
                             normalize_embeddings=False, convert_to_numpy=True)
            vecs.append(v)
        new_vecs = np.vstack(vecs) if vecs else np.zeros((0, 384), dtype=float)
        for local_i, rid in enumerate([req_ids[i] for i in to_embed_idx]):
            cached[rid] = new_vecs[local_i]

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_df = pd.DataFrame({"req_id": list(cached.keys()), "vec": [json.dumps(v.tolist()) for v in cached.values()]})
        cache_df.to_csv(cache_path, index=False)

    E = np.vstack([cached[str(rid)] for rid in req_ids])
    return E

# ---------- Audit helpers ----------
def topk_unique(arr, k=5):
    idx = np.argsort(arr)[::-1]
    out = []
    for i in idx:
        out.append(i)
        if len(out) >= k: break
    return out

def bottomk_unique(arr, k=5):
    idx = np.argsort(arr)
    out = []
    for i in idx:
        out.append(i)
        if len(out) >= k: break
    return out

# ---------- Group scoring ----------
def score_group(df_g, embeds_mat, idxs, params):
    texts = df_g["text"].fillna("").tolist()
    n = len(texts)
    V1 = np.zeros(n); V2 = np.zeros(n); V3 = np.zeros(n)

    meas = np.array([measurable_override_factor(t, params["measurable_override"]) for t in texts])
    for i, t in enumerate(texts):
        V1[i] = v1_hedging_density(t) * meas[i]
        V2[i] = v2_conditional_temporal(t, lam=params["lambda_v2"], legal_cap=params["legal_cap"]) * meas[i]
        V3[i] = v3_open_ended(t) * meas[i]

    subE = unit_normalize_rows(embeds_mat[idxs, :], eps=params["eps"])
    C1, C2 = compute_c1_c2(subE, k=params["k"], eps=params["eps"])
    alpha = params["alpha_small_n"] if n < 5 else params["alpha"]
    V4 = np.clip(1.0 - (alpha*C1 + (1-alpha)*C2), 0.0, 1.0)

    flags = redundancy_discount_flags(subE, sim_thresh=params["redundancy_sim_thresh"])
    V4 = np.where(flags, V4*params["redundancy_discount"], V4)

    notes = {"collapsed": {}}
    V1n, c1 = normalize_minmax(V1, params["eps_span"])
    V2n, c2 = normalize_minmax(V2, params["eps_span"])
    V3n, c3 = normalize_minmax(V3, params["eps_span"])
    V4n, c4 = normalize_minmax(V4, params["eps_span"])
    notes["collapsed"].update({"V1":bool(c1),"V2":bool(c2),"V3":bool(c3),"V4":bool(c4)})

    cols = {"V1": V1n.copy(), "V2": V2n.copy(), "V3": V3n.copy(), "V4": V4n.copy()}
    corr_applied = []
    if n >= 6:
        cols, corr_applied = correlation_dampen(cols, ["V1","V2","V3","V4"],
                                                thresh=params["corr_thresh"], damp=params["corr_damp"])
    V1n, V2n, V3n, V4n = cols["V1"], cols["V2"], cols["V3"], cols["V4"]
    notes["correlation_damping"] = corr_applied

    X = np.vstack([V1n, V2n, V3n, V4n]).T
    weights = entropy_weights(X, gamma=params["gamma_shrink"], w_min=params["w_min"], w_max=params["w_max"], eps=params["eps"])
    notes["weights"] = {"V1": float(weights[0]), "V2": float(weights[1]), "V3": float(weights[2]), "V4": float(weights[3])}

    V_final = np.clip(X.dot(weights), 0.0, 1.0)

    k_show = min(5, max(1, n // 2)) if n >= 3 else min(1, n)
    audit = {
        "top_volatile_idx": topk_unique(V_final, k=k_show),
        "least_volatile_idx": bottomk_unique(V_final, k=k_show),
        "leaders": {
            "V1": topk_unique(V1n, k=min(3, n)),
            "V2": topk_unique(V2n, k=min(3, n)),
            "V3": topk_unique(V3n, k=min(3, n)),
            "V4": topk_unique(V4n, k=min(3, n)),
        }
    }

    return dict(
        V1=V1, V2=V2, V3=V3, V4=V4,
        V1n=V1n, V2n=V2n, V3n=V3n, V4n=V4n,
        V=V_final, weights=weights, notes=notes, audit=audit, n=n
    )

def main():
    ap = argparse.ArgumentParser(description="Volatility scoring (V1–V4) with on-the-fly embeddings")
    ap.add_argument("--input", default="results/redundancy/resolved_requirements.csv")
    ap.add_argument("--model", default="intfloat/e5-large")
    ap.add_argument("--out_scores", default="results/volatility/volatility_scored.csv")
    ap.add_argument("--out_meta", default="results/volatility/volatility_meta.json")

    ap.add_argument("--embed_mode", choices=["onfly","csv","npy"], default="onfly")
    ap.add_argument("--embeddings_csv", default="results/embeddings/requirements_e5_large.csv")
    ap.add_argument("--embeddings_npy", default=None)
    ap.add_argument("--id_map_csv", default=None)

    ap.add_argument("--embed_cache", default="results/embeddings/volatility_e5_cache.csv")
    ap.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--device", default=DEFAULTS["device"])

    ap.add_argument("--k", type=int, default=DEFAULTS["k"])
    ap.add_argument("--alpha", type=float, default=DEFAULTS["alpha"])
    ap.add_argument("--alpha_small_n", type=float, default=DEFAULTS["alpha_small_n"])
    ap.add_argument("--lambda_v2", type=float, default=DEFAULTS["lambda_v2"])
    ap.add_argument("--gamma_shrink", type=float, default=DEFAULTS["gamma_shrink"])
    ap.add_argument("--w_min", type=float, default=DEFAULTS["w_min"])
    ap.add_argument("--w_max", type=float, default=DEFAULTS["w_max"])
    ap.add_argument("--eps_span", type=float, default=DEFAULTS["eps_span"])
    ap.add_argument("--corr_thresh", type=float, default=DEFAULTS["corr_thresh"])
    ap.add_argument("--corr_damp", type=float, default=DEFAULTS["corr_damp"])
    ap.add_argument("--legal_cap", type=float, default=DEFAULTS["legal_cap"])
    ap.add_argument("--measurable_override", type=float, default=DEFAULTS["measurable_override"])
    ap.add_argument("--redundancy_sim_thresh", type=float, default=DEFAULTS["redundancy_sim_thresh"])
    ap.add_argument("--redundancy_discount", type=float, default=DEFAULTS["redundancy_discount"])
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_scores), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_meta), exist_ok=True)

    df = pd.read_csv(args.input)
    required = {"req_id","text","system","class"}
    if not required.issubset(df.columns):
        raise ValueError(f"--input must contain columns {required}")
    df["req_id"] = df["req_id"].astype(str)

    # ---- Embeddings ----
    if args.embed_mode == "onfly":
        E = embed_on_the_fly(
            texts=df["text"].fillna("").tolist(),
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device,
            cache_path=args.embed_cache,
            req_ids=df["req_id"].tolist()
        )
    elif args.embed_mode == "csv":
        id2vec = load_embeddings_from_csv(args.embeddings_csv)
        vecs = [id2vec[rid] for rid in df["req_id"]]
        E = np.vstack(vecs)
    elif args.embed_mode == "npy":
        if not (args.embeddings_npy and args.id_map_csv):
            raise ValueError("Provide --embeddings_npy and --id_map_csv for --embed_mode npy")
        E_all, id2row = load_embeddings_from_npy(args.embeddings_npy, args.id_map_csv)
        rows = [id2row[rid] for rid in df["req_id"]]
        E = E_all[rows, :]
    else:
        raise ValueError("Unknown --embed_mode")

    E = unit_normalize_rows(E, eps=DEFAULTS["eps"])

    params = {
        "k": args.k,
        "alpha": args.alpha,
        "alpha_small_n": args.alpha_small_n,
        "lambda_v2": args.lambda_v2,
        "gamma_shrink": args.gamma_shrink,
        "w_min": args.w_min,
        "w_max": args.w_max,
        "eps_span": args.eps_span,
        "eps": DEFAULTS["eps"],
        "corr_thresh": args.corr_thresh,
        "corr_damp": args.corr_damp,
        "legal_cap": args.legal_cap,
        "measurable_override": args.measurable_override,
        "redundancy_sim_thresh": args.redundancy_sim_thresh,
        "redundancy_discount": args.redundancy_discount,
    }

    out_rows = []
    meta = defaultdict(dict)
    df = df.reset_index(drop=True)

    for (sys_name, cls_name), df_g in tqdm(df.groupby(["system","class"], sort=False), desc="Scoring groups"):
        idx = df_g.index.values
        result = score_group(df_g, E, idx, params)

        for j in range(len(df_g)):
            out_rows.append({
                "req_id": df_g["req_id"].iloc[j],
                "system": sys_name,
                "class": cls_name,
                "text": df_g["text"].iloc[j],
                "V1_raw": float(result["V1"][j]),
                "V2_raw": float(result["V2"][j]),
                "V3_raw": float(result["V3"][j]),
                "V4_raw": float(result["V4"][j]),
                "V1": float(result["V1n"][j]),
                "V2": float(result["V2n"][j]),
                "V3": float(result["V3n"][j]),
                "V4": float(result["V4n"][j]),
                "V_final": float(result["V"][j]),
                "w_V1": float(result["weights"][0]),
                "w_V2": float(result["weights"][1]),
                "w_V3": float(result["weights"][2]),
                "w_V4": float(result["weights"][3]),
            })

        key = f"{sys_name}::{cls_name}"
        meta[key] = {
            "weights": result["notes"]["weights"],
            "collapsed_columns": result["notes"]["collapsed"],
            "correlation_damping": result["notes"]["correlation_damping"],
            "audit": {
                "top_volatile": [df_g["req_id"].tolist()[i] for i in result["audit"]["top_volatile_idx"]],
                "least_volatile": [df_g["req_id"].tolist()[i] for i in result["audit"]["least_volatile_idx"]],
                "leaders": {
                    "V1": [df_g["req_id"].tolist()[i] for i in result["audit"]["leaders"]["V1"]],
                    "V2": [df_g["req_id"].tolist()[i] for i in result["audit"]["leaders"]["V2"]],
                    "V3": [df_g["req_id"].tolist()[i] for i in result["audit"]["leaders"]["V3"]],
                    "V4": [df_g["req_id"].tolist()[i] for i in result["audit"]["leaders"]["V4"]],
                }
            },
            "params": {
                "n": int(result["n"]),
                "k": params["k"],
                "alpha_used": params["alpha_small_n"] if result["n"] < 5 else params["alpha"],
                "lambda_v2": params["lambda_v2"],
                "gamma_shrink": params["gamma_shrink"],
                "w_minmax": [params["w_min"], params["w_max"]],
                "eps_span": params["eps_span"],
                "corr_thresh": params["corr_thresh"],
                "corr_damp": params["corr_damp"],
                "legal_cap": params["legal_cap"],
                "measurable_override": params["measurable_override"],
                "redundancy_sim_thresh": params["redundancy_sim_thresh"],
                "redundancy_discount": params["redundancy_discount"]
            }
        }

    pd.DataFrame(out_rows).to_csv(args.out_scores, index=False)
    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n=== Volatility scoring complete ===")
    print(f"Saved scores: {args.out_scores}")
    print(f"Saved meta:   {args.out_meta}")

if __name__ == "__main__":
    main()
