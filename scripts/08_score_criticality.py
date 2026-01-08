#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Criticality scoring (C1, C2, C3) — PER-SYSTEM/CLASS with embeddings
===================================================================

Input:
  - resolved_requirements.csv with columns:
      req_id, text, system, class[, source]

Output:
  - results/criticality/criticality_scored.csv
  - results/criticality/weights_by_system.csv
  - results/criticality/diagnostics/{per_system.json, validation_summary.json}

Modes:
  • --embed_mode model (default): compute embeddings with SentenceTransformer(model)
  • --embed_mode files          : merge precomputed embeddings from --emb_dir

Components (per (system,class)):
  - C1: cosine similarity to centroid direction
  - C2: mean cosine similarity to k nearest neighbors
  - C3: cohesion drop S(G) - S(G\{i}) (clamped >= 0); neutral if n<4

Scaling & guards (per (system,class), per component):
  - Winsorize @ 5/95 → min–max to [0,1]
  - SPAN_EPS guard: if span<eps, set component to 0.5 (neutral)
  - Tiny groups: if n<=3 ⇒ C2=C1; C3=0.5

Weights (per system):
  - Entropy Weight Method (EWM) on M01=[C1,C2,C3] + shrink toward uniform
  - Hard bounds enforced: wmin ≤ w_i ≤ wmax, and sum(w)=1 (bounded-simplex projection)
"""

import argparse, ast, json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Optional dependency only used in embed_mode=model
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    k: int = 5
    winsor_p: float = 0.05
    span_eps: float = 0.05
    eps: float = 1e-12
    # EWM params
    wmin: float = 0.20
    wmax: float = 0.50
    kappa: float = 1.5

ID_CANDIDATES  = ["req_id","id","reqID","requirement_id","RequirementID","ReqID","RID"]
SYSTEM_CANDS   = ["system","System","sys"]
CLASS_CANDS    = ["class","Class","quality","Quality","qa","QA"]
TEXT_CANDS     = ["text","Text","requirement","Requirement","sentence","Sentence"]
EMBED_CANDS    = ["embedding","vector","Embedding","Vector"]

# -----------------------------
# Helpers
# -----------------------------
def _find_first(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None

def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sysc = _find_first(df, SYSTEM_CANDS)
    if sysc and sysc != "system":
        df = df.rename(columns={sysc: "system"})
    clsc = _find_first(df, CLASS_CANDS)
    if clsc and clsc != "class":
        df = df.rename(columns={clsc: "class"})
    txtc = _find_first(df, TEXT_CANDS)
    if txtc and txtc != "text":
        df = df.rename(columns={txtc: "text"})
    idc = _find_first(df, ID_CANDIDATES)
    if idc is None:
        df["req_id"] = np.arange(len(df))
    elif idc != "req_id":
        df = df.rename(columns={idc: "req_id"})
    required = {"req_id","text","system","class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df

def parse_embedding(val) -> np.ndarray:
    if isinstance(val, (list, tuple, np.ndarray)):
        return np.asarray(val, dtype=np.float32)
    if isinstance(val, (bytes, bytearray)):
        val = val.decode("utf-8", errors="ignore")
    if isinstance(val, str):
        s = val.strip()
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                return np.asarray(obj, dtype=np.float32)
        except Exception:
            pass
        parts = [p for p in (s.split(",") if "," in s else s.split()) if p.strip()]
        return np.asarray([float(p) for p in parts], dtype=np.float32)
    raise ValueError(f"Unsupported embedding format: {type(val)}")

def _detect_embedding_vector(df: pd.DataFrame) -> Optional[List[str]]:
    cols = list(df.columns)
    pat = re.compile(r"^(emb|vec|e|v)[_\-]?(\d+)$", re.IGNORECASE)
    hits = []
    for c in cols:
        m = pat.match(c.strip())
        if m:
            hits.append((int(m.group(2)), c))
    if not hits:
        pat2 = re.compile(r"^(emb|vec)(\d+)$", re.IGNORECASE)
        for c in cols:
            m = pat2.match(c.strip())
            if m:
                hits.append((int(m.group(2)), c))
    if not hits:
        return None
    hits.sort()
    return [c for _, c in hits]

def winsorize_1d(x: np.ndarray, p: float) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = np.quantile(x, [p, 1.0 - p])
    return np.clip(x, lo, hi)

def minmax_01(x: np.ndarray, span_eps: float) -> Tuple[np.ndarray, float]:
    mn = float(np.min(x))
    mx = float(np.max(x))
    span = mx - mn
    if span < span_eps:
        return np.full_like(x, 0.5, dtype=np.float64), span
    return (x - mn) / max(span, span_eps), span

def cosine_similarity_matrix(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
    Xn = X / norms
    return np.clip(Xn @ Xn.T, -1.0, 1.0)

def unit_normalize_rows(X: np.ndarray, eps: float) -> Tuple[np.ndarray, float]:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
    Xn = X / norms
    raw = np.linalg.norm(X, axis=1)
    frac = float(np.mean((raw < 0.999) | (raw > 1.001)))
    return Xn.astype(np.float32), frac

# -----------------------------
# Embeddings
# -----------------------------
def get_embeddings_model(texts: List[str], model_name: str) -> np.ndarray:
    if not _HAS_ST:
        raise RuntimeError("sentence-transformers not installed. Install it or use --embed_mode files.")
    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, batch_size=16, convert_to_numpy=True, show_progress_bar=True)
    return vecs.astype(np.float32)

def get_embeddings_from_files(df: pd.DataFrame, emb_dir: Path) -> pd.DataFrame:
    if "embedding" in df.columns:
        return df

    files = list(emb_dir.rglob("*.csv")) + list(emb_dir.rglob("*.parquet")) + list(emb_dir.rglob("*.jsonl"))
    if not files:
        raise ValueError(f"No embedding files found under {emb_dir}")

    merged_any = False

    def _read_any(p: Path) -> Optional[pd.DataFrame]:
        try:
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            if p.suffix == ".jsonl":
                return pd.read_json(p, lines=True)
            if p.suffix in (".csv", ".tsv"):
                sep = "," if p.suffix == ".csv" else "\t"
                return pd.read_csv(p, sep=sep)
            return None
        except Exception:
            return None

    for p in files:
        emb = _read_any(p)
        if emb is None or emb.empty:
            continue

        key = "req_id" if "req_id" in emb.columns else ("text" if "text" in emb.columns else None)
        if key is None:
            continue

        emb_col = _find_first(emb, EMBED_CANDS)
        wide = _detect_embedding_vector(emb)

        if wide:
            emb["embedding"] = emb[wide].astype(float).values.tolist()
        elif emb_col and emb_col != "embedding":
            emb = emb.rename(columns={emb_col: "embedding"})

        if "embedding" not in emb.columns:
            continue

        emb = emb[[key, "embedding"]].dropna()

        try:
            df = df.merge(emb, on=key, how="left")
            merged_any = merged_any or df["embedding"].notna().any()
        except Exception:
            continue

    if not merged_any:
        raise ValueError("Failed to merge embeddings from files. Provide `req_id` or exact `text` in embeddings.")

    return df

# -----------------------------
# Components
# -----------------------------
def C1(Emb_u: np.ndarray, eps: float) -> np.ndarray:
    m = Emb_u.mean(axis=0)
    nrm = np.linalg.norm(m)
    if nrm < eps:
        return np.full(Emb_u.shape[0], 0.5, dtype=np.float64)
    mu = m / nrm
    sims = np.clip(Emb_u @ mu, -1.0, 1.0)
    return (sims + 1.0) / 2.0

def C2(Emb_u: np.ndarray, k: int, eps: float) -> np.ndarray:
    n = Emb_u.shape[0]
    if n <= 1:
        return np.full(n, 0.5, dtype=np.float64)
    kk = max(1, min(k, n - 1))
    S = cosine_similarity_matrix(Emb_u, eps=eps)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sims = np.delete(S[i], i)
        if sims.size == 0:
            out[i] = 0.5
        else:
            idx = np.argpartition(-sims, kk - 1)[:kk]
            out[i] = float(np.mean(sims[idx]))
    return (out + 1.0) / 2.0

def C3(Emb_u: np.ndarray, eps: float) -> np.ndarray:
    n = Emb_u.shape[0]
    if n < 4:
        return np.full(n, 0.5, dtype=np.float64)
    S = cosine_similarity_matrix(Emb_u, eps=eps)
    iu = np.triu_indices(n, k=1)
    sum_all = float(np.sum(S[iu]))
    denom_all = n * (n - 1) / 2.0
    coh_all = sum_all / max(denom_all, eps)
    drops = np.zeros(n, dtype=np.float64)
    row_sums = S.sum(axis=1) - 1.0  # exclude diagonal
    for i in range(n):
        sum_leave = sum_all - row_sums[i]
        denom_leave = (n - 1) * (n - 2) / 2.0
        coh_leave = sum_leave / max(denom_leave, eps) if denom_leave > 0 else coh_all
        d = coh_all - coh_leave
        drops[i] = max(d, 0.0)
    return drops

# -----------------------------
# Weighting (EWM + bounded caps)
# -----------------------------
def ewm_weights(M01: np.ndarray, eps: float) -> np.ndarray:
    """
    Standard entropy weight method computed on M01 (already in [0,1]).
    Columns are criteria; rows are alternatives.
    """
    n, m = M01.shape
    P = M01 / (np.sum(M01, axis=0, keepdims=True) + eps)
    k = 1.0 / np.log(max(n, 2))
    E = -k * np.sum(P * np.log(P + eps), axis=0)
    D = 1.0 - E
    if np.all(D <= eps):
        return np.ones(m) / m
    return D / (np.sum(D) + eps)

def project_to_bounded_simplex(w: np.ndarray, wmin: float, wmax: float, eps: float = 1e-12) -> np.ndarray:
    """
    Enforce BOTH:
      sum(w)=1  and  wmin <= w_i <= wmax
    via iterative redistribution (water-filling).
    """
    w = np.asarray(w, dtype=np.float64)

    m = w.size
    if m * wmin - 1.0 > 1e-9 or 1.0 - m * wmax > 1e-9:
        raise ValueError(f"Infeasible bounds: m*wmin={m*wmin} m*wmax={m*wmax}")

    w = np.clip(w, wmin, wmax)

    for _ in range(1000):
        s = w.sum()
        if abs(s - 1.0) <= 1e-12:
            break

        if s < 1.0:
            free = w < (wmax - 1e-12)
            if not np.any(free):
                break
            add = (1.0 - s) / free.sum()
            w[free] = np.minimum(wmax, w[free] + add)
        else:
            free = w > (wmin + 1e-12)
            if not np.any(free):
                break
            sub = (s - 1.0) / free.sum()
            w[free] = np.maximum(wmin, w[free] - sub)

    s = w.sum()
    if abs(s - 1.0) > 1e-8:
        w = w / (s + eps)

    return w

def shrink_cap(w: np.ndarray, n_eff: int, kappa: float, wmin: float, wmax: float, eps: float) -> np.ndarray:
    """
    Shrink weights toward uniform as sample size increases,
    then project onto bounded simplex to enforce true caps.
    """
    m = w.size
    u = np.ones(m) / m

    lam = min(1.0, kappa / max(np.sqrt(max(n_eff, 1)), 1.0))
    w = lam * u + (1.0 - lam) * w

    # Hard bounds + sum-to-1
    w = project_to_bounded_simplex(w, wmin=wmin, wmax=wmax, eps=eps)
    return w

# -----------------------------
# Orchestration
# -----------------------------
def compute_criticality(df: pd.DataFrame, cfg: CFG, outdir: Path, use_ewm: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Expect df has: req_id, text, system, class, embedding

    # Parse embeddings and check dims
    E_list = df["embedding"].tolist()
    # convert if strings/lists
    E_list = [parse_embedding(v) for v in E_list]
    dim_set = {len(v) for v in E_list}
    if len(dim_set) != 1:
        raise ValueError(f"Inconsistent embedding dims: {sorted(dim_set)}")

    X = np.vstack(E_list).astype(np.float32)
    X_u, frac_renorm = unit_normalize_rows(X, cfg.eps)

    df = df.copy()
    df["C1"] = np.nan
    df["C2"] = np.nan
    df["C3"] = np.nan
    df["w_C1"] = np.nan
    df["w_C2"] = np.nan
    df["w_C3"] = np.nan
    df["Criticality"] = np.nan

    diagnostics_sys: Dict[str, Dict] = {}
    weight_rows: List[Dict] = []

    for system, g_sys in df.groupby("system", sort=False):
        idx_sys = g_sys.index.values

        C1_sys = np.zeros(len(g_sys), dtype=np.float64)
        C2_sys = np.zeros(len(g_sys), dtype=np.float64)
        C3_sys = np.zeros(len(g_sys), dtype=np.float64)

        cls_diags = []

        for cls, g_cls in g_sys.groupby("class", sort=False):
            idx_cls = g_cls.index.values
            Xc = X_u[idx_cls]
            n = Xc.shape[0]
            tiny = n <= 3

            if n == 1:
                c1r = np.array([0.5], dtype=np.float64)
                c2r = np.array([0.5], dtype=np.float64)
                c3r = np.array([0.5], dtype=np.float64)
            else:
                c1r = C1(Xc, cfg.eps).astype(np.float64)
                c2r = C2(Xc, cfg.k, cfg.eps).astype(np.float64)
                c3r = C3(Xc, cfg.eps).astype(np.float64)
                if tiny:
                    c2r = c1r.copy()
                    c3r = np.full_like(c3r, 0.5, dtype=np.float64)

            def scale(z: np.ndarray) -> Tuple[np.ndarray, float]:
                wz = winsorize_1d(z, cfg.winsor_p)
                z01, span = minmax_01(wz, cfg.span_eps)
                return z01.astype(np.float64), float(span)

            c1s, s1 = scale(c1r)
            c2s, s2 = scale(c2r)
            c3s, s3 = scale(c3r)

            # place into system vectors using relative positions inside g_sys
            rel = g_sys.index.get_indexer(idx_cls)
            C1_sys[rel] = c1s
            C2_sys[rel] = c2s
            C3_sys[rel] = c3s

            cls_diags.append({
                "class": str(cls),
                "n": int(n),
                "tiny_group": bool(tiny),
                "C1_span": float(s1),
                "C1_span_flag": bool(s1 < cfg.span_eps),
                "C2_span": float(s2),
                "C2_span_flag": bool(s2 < cfg.span_eps),
                "C3_span": float(s3),
                "C3_span_flag": bool(s3 < cfg.span_eps),
            })

        M01 = np.stack([C1_sys, C2_sys, C3_sys], axis=1).astype(np.float64)

        if use_ewm:
            w = ewm_weights(M01, cfg.eps)
            w = shrink_cap(w, n_eff=len(g_sys), kappa=cfg.kappa, wmin=cfg.wmin, wmax=cfg.wmax, eps=cfg.eps)
        else:
            w = np.array([1/3, 1/3, 1/3], dtype=np.float64)

        crit = (M01 @ w).astype(np.float64)

        df.loc[idx_sys, "C1"] = C1_sys
        df.loc[idx_sys, "C2"] = C2_sys
        df.loc[idx_sys, "C3"] = C3_sys
        df.loc[idx_sys, "Criticality"] = crit
        df.loc[idx_sys, "w_C1"] = float(w[0])
        df.loc[idx_sys, "w_C2"] = float(w[1])
        df.loc[idx_sys, "w_C3"] = float(w[2])

        corr = pd.DataFrame(M01, columns=["C1","C2","C3"]).corr(method="pearson").to_dict()
        diagnostics_sys[str(system)] = {
            "n_reqs": int(len(g_sys)),
            "n_classes": int(g_sys["class"].nunique()),
            "weights": {"C1": float(w[0]), "C2": float(w[1]), "C3": float(w[2])},
            "component_corr": corr,
            "classes": cls_diags
        }

        weight_rows.append({
            "system": str(system),
            "w_C1": float(w[0]),
            "w_C2": float(w[1]),
            "w_C3": float(w[2]),
            "n_reqs": int(len(g_sys)),
            "n_classes": int(g_sys["class"].nunique())
        })

    # Save outputs
    outdir.mkdir(parents=True, exist_ok=True)

    scored_cols = ["system","class","req_id","C1","C2","C3","w_C1","w_C2","w_C3","Criticality"]
    scored_df = df[scored_cols].copy().sort_values(["system","class","req_id"])
    scored_df.to_csv(outdir / "criticality_scored.csv", index=False)

    weights_df = pd.DataFrame(weight_rows).sort_values("system")
    weights_df.to_csv(outdir / "weights_by_system.csv", index=False)

    # Diagnostics
    diag_dir = outdir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    val_summary = {
        "embedding_dim": int(X_u.shape[1]),
        "frac_embeddings_renormalized": float(frac_renorm),
        "winsor_p": float(cfg.winsor_p),
        "span_eps": float(cfg.span_eps),
        "ewm_enabled": bool(use_ewm),
        "ewm_wmin": float(cfg.wmin),
        "ewm_wmax": float(cfg.wmax),
        "ewm_kappa": float(cfg.kappa),
    }

    (diag_dir / "per_system.json").write_text(json.dumps(diagnostics_sys, indent=2), encoding="utf-8")
    (diag_dir / "validation_summary.json").write_text(json.dumps(val_summary, indent=2), encoding="utf-8")

    return scored_df, weights_df

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute Criticality (C1,C2,C3) per system/class")
    ap.add_argument("--input", default="results/redundancy/resolved_requirements.csv")
    ap.add_argument("--outdir", default="results/criticality")

    # Embedding options
    ap.add_argument("--embed_mode", choices=["model","files"], default="model",
                    help="model: compute embeddings with SentenceTransformer; files: merge from --emb_dir")
    ap.add_argument("--model", default="intfloat/e5-large",
                    help="SentenceTransformer model (used if --embed_mode model)")
    ap.add_argument("--emb_dir", default="embeddings/domain_class",
                    help="Directory with embeddings (used if --embed_mode files)")

    # Component params
    ap.add_argument("--k", type=int, default=5, help="k for kNN in C2")
    ap.add_argument("--winsor_p", type=float, default=0.05)
    ap.add_argument("--span_eps", type=float, default=0.05)

    # Weighting
    ap.add_argument("--use_ewm", action="store_true",
                    help="Use EWM + shrink + bounded caps instead of uniform weights")
    ap.add_argument("--wmin", type=float, default=0.20)
    ap.add_argument("--wmax", type=float, default=0.50)
    ap.add_argument("--kappa", type=float, default=1.5)

    args = ap.parse_args()

    cfg = CFG(
        k=args.k,
        winsor_p=args.winsor_p,
        span_eps=args.span_eps,
        wmin=args.wmin,
        wmax=args.wmax,
        kappa=args.kappa,
    )

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Missing input CSV: {input_path}")

    df = pd.read_csv(input_path)
    df = coerce_schema(df)

    # Get embeddings
    if args.embed_mode == "model":
        vecs = get_embeddings_model(df["text"].astype(str).tolist(), args.model)
        df["embedding"] = [v for v in vecs]
    else:
        df = get_embeddings_from_files(df, Path(args.emb_dir))
        if "embedding" not in df.columns or df["embedding"].isna().all():
            raise SystemExit("Embedding merge failed: no embeddings found after reading from --emb_dir")

    outdir = Path(args.outdir)
    compute_criticality(df, cfg, outdir, use_ewm=args.use_ewm)

    print("[criticality] wrote:", outdir / "criticality_scored.csv")
    print("[criticality] wrote:", outdir / "weights_by_system.csv")
    print("[criticality] diagnostics:", outdir / "diagnostics")

if __name__ == "__main__":
    main()
