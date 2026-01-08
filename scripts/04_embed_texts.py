#!/usr/bin/env python3
# utils/embedding_generator.py
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
from typing import List
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEF_MODEL = "intfloat/e5-large"   # passage encoder
DEF_BATCH = 32

def set_seeds(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model(model_name: str = DEF_MODEL) -> SentenceTransformer:
    print(f" Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
        print("⚡ Using GPU")
    else:
        print("🖥️  Using CPU")
    return model

def _prefix_passage(texts: List[str]) -> List[str]:
    # E5 expects prefixes: use "passage: " for documents
    return [f"passage: {t}" for t in texts]

def _l2_normalize(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return arr / norms

def generate_embeddings(texts: List[str], model: SentenceTransformer, batch_size: int = DEF_BATCH) -> np.ndarray:
    if len(texts) == 0:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    # E5 passage prefix
    enc_inputs = _prefix_passage(texts)
    embs = model.encode(
        enc_inputs,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,   # we'll normalize ourselves
        show_progress_bar=False
    )
    embs = _l2_normalize(embs).astype(np.float32)
    return embs

def should_skip_csv(name: str) -> bool:
    # skip helper files such as _manifest.csv
    return not name.endswith(".csv") or name.startswith("_")

def process_dir(input_dir: str, output_dir: str, model: SentenceTransformer, batch_size: int = DEF_BATCH) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(input_dir) if not should_skip_csv(f)])
    print(f" Found {len(files)} CSV files to process in {input_dir}")

    manifest_rows = []
    for fname in tqdm(files, desc="🔍 Embedding CSVs"):
        in_path = os.path.join(input_dir, fname)
        base = fname[:-4]  # strip .csv
        out_npy = os.path.join(output_dir, f"{base}.npy")
        out_txt = os.path.join(output_dir, f"{base}_texts.csv")

        df = pd.read_csv(in_path)
        df.columns = df.columns.str.lower().str.strip()

        if "text" not in df.columns:
            print(f"⚠️  Skipping {fname}: missing 'text' column.")
            continue

        texts = df["text"].fillna("").astype(str).str.strip().tolist()
        req_ids = df["req_id"].tolist() if "req_id" in df.columns else list(range(len(df)))

        n = len(texts)
        if n == 0:
            print(f"⚠️  Skipping {fname}: zero rows.")
            continue

        # Encode
        embs = generate_embeddings(texts, model, batch_size=batch_size)
        if embs.shape[0] != n:
            raise RuntimeError(f"Row mismatch for {fname}: texts={n}, embs={embs.shape[0]}")

        # Save artifacts
        np.save(out_npy, embs.astype(np.float32))
        pd.DataFrame({"req_id": req_ids, "text": texts}).to_csv(out_txt, index=False)

        manifest_rows.append({
            "input_csv": in_path,
            "output_npy": out_npy,
            "texts_csv": out_txt,
            "rows": n,
            "dim": embs.shape[1],
            "dtype": "float32"
        })

    manifest = pd.DataFrame(manifest_rows)
    if not manifest.empty:
        man_path = os.path.join(output_dir, "_manifest.csv")
        manifest.to_csv(man_path, index=False)
        print(f" Embedding manifest written → {man_path}")
    print(" Embedding generation complete.")
    return manifest

def main():
    parser = argparse.ArgumentParser(description="Generate sentence embeddings per (system, class) CSV.")
    parser.add_argument("--input_dir", default="data/split_by_class-system", help="Directory with {system}_{class}.csv files")
    parser.add_argument("--output_dir", default="embeddings/domain_class", help="Directory to write .npy embeddings")
    parser.add_argument("--model", default=DEF_MODEL, help="SentenceTransformer model name")
    parser.add_argument("--batch_size", type=int, default=DEF_BATCH, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seeds(args.seed)
    model = load_model(args.model)
    process_dir(args.input_dir, args.output_dir, model, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
