from pathlib import Path
import pandas as pd
import yaml


def profile_scope(df: pd.DataFrame, scope_name: str) -> dict:
    lengths_chars = df["text"].str.len()
    lengths_tokens = df["text"].str.split().apply(len)

    return {
        "scope": scope_name,
        "n_requirements": int(len(df)),
        "n_classes": int(df["class"].nunique()),
        "char_min": int(lengths_chars.min()),
        "char_median": int(lengths_chars.median()),
        "char_max": int(lengths_chars.max()),
        "token_min": int(lengths_tokens.min()),
        "token_median": int(lengths_tokens.median()),
        "token_max": int(lengths_tokens.max()),
    }


def main():
    cfg = yaml.safe_load(Path("config/config.yaml").read_text())
    df = pd.read_csv(cfg["paths"]["data_csv"])

    results_dir = Path(cfg["paths"]["results_dir"]) / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Global profile ----
    global_profile = profile_scope(df, "global")

    # ---- Primary system profile ----
    primary_system = "ecommerce"
    df_primary = df[df["system"] == primary_system]

    primary_profile = profile_scope(df_primary, primary_system)

    # ---- Save summary table ----
    summary_df = pd.DataFrame([global_profile, primary_profile])
    summary_path = results_dir / "dataset_profile_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # ---- Class distribution for primary system ----
    class_dist = (
        df_primary["class"]
        .value_counts()
        .rename_axis("class")
        .reset_index(name="count")
    )
    class_dist_path = results_dir / "ecommerce_class_distribution.csv"
    class_dist.to_csv(class_dist_path, index=False)

    # ---- Markdown summary ----
    md_lines = []
    md_lines.append("# Step 2: Dataset Profiling Summary\n")

    for row in [global_profile, primary_profile]:
        md_lines.append(f"## Scope: {row['scope']}")
        md_lines.append(f"- Number of requirements: **{row['n_requirements']}**")
        md_lines.append(f"- Number of classes: **{row['n_classes']}**")
        md_lines.append(
            f"- Character length (min / median / max): "
            f"{row['char_min']} / {row['char_median']} / {row['char_max']}"
        )
        md_lines.append(
            f"- Token length (min / median / max): "
            f"{row['token_min']} / {row['token_median']} / {row['token_max']}"
        )
        md_lines.append("")

    md_path = Path(cfg["paths"]["results_dir"]) / "logs" / "dataset_profile_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[OK] Wrote: {summary_path}")
    print(f"[OK] Wrote: {class_dist_path}")
    print(f"[OK] Wrote: {md_path}")


if __name__ == "__main__":
    main()
