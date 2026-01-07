from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml


def main():
    cfg = yaml.safe_load(Path("config/config.yaml").read_text())
    df = pd.read_csv(cfg["paths"]["data_csv"])

    # Evidence-driven primary system (from Step 1)
    primary_system = "ecommerce"
    n_primary = int((df["system"] == primary_system).sum())
    n_total = int(len(df))

    # Evidence-driven diversity check (from Step 2)
    classes_global = sorted(df["class"].unique().tolist())
    classes_primary = sorted(df[df["system"] == primary_system]["class"].unique().tolist())
    covers_all_classes = (classes_primary == classes_global)

    # Sanity requirement: primary system must be large enough to support robustness tests
    # (We keep this threshold explicit and conservative; can be revised later)
    min_primary_n_for_robustness = 50
    primary_size_ok = n_primary >= min_primary_n_for_robustness

    # Summarize secondary systems (for bounded sanity checks)
    system_counts = df["system"].value_counts().to_dict()
    secondary_systems = [s for s in system_counts.keys() if s != primary_system]

    results_tables = Path(cfg["paths"]["results_dir"]) / "tables"
    results_tables.mkdir(parents=True, exist_ok=True)

    case_study_dir = Path("case_study")
    case_study_dir.mkdir(parents=True, exist_ok=True)

    # Table artifact for traceability
    design_rows = [
        {"field": "generated_at_utc", "value": datetime.utcnow().isoformat() + "Z"},
        {"field": "dataset_path", "value": cfg["paths"]["data_csv"]},
        {"field": "dataset_total_requirements", "value": n_total},
        {"field": "unit_of_analysis", "value": "requirement"},
        {"field": "primary_case_system", "value": primary_system},
        {"field": "primary_case_n", "value": n_primary},
        {"field": "primary_covers_all_classes", "value": covers_all_classes},
        {"field": "primary_size_ok_for_robustness_tests", "value": primary_size_ok},
        {"field": "min_primary_n_for_robustness", "value": min_primary_n_for_robustness},
        {"field": "secondary_systems_for_sanity_checks", "value": ", ".join(secondary_systems)},
        {"field": "validation_scope", "value": "construct validity (operational), internal consistency, robustness"},
        {"field": "out_of_scope", "value": "business value ground truth; stakeholder satisfaction; cost/benefit benchmarking"},
    ]

    out_csv = results_tables / "case_study_design_summary.csv"
    pd.DataFrame(design_rows).to_csv(out_csv, index=False)

    # Human-readable design doc (short + precise)
    md = []
    md.append("# Case Study Design (Step 3)\n")
    md.append("## Objective\n")
    md.append("Evaluate PRISM’s prioritization behavior on a realistic requirements dataset without assuming business-value ground truth.\n")
    md.append("## Dataset basis\n")
    md.append(f"- Frozen input: `{cfg['paths']['data_csv']}`\n")
    md.append(f"- Total requirements: **{n_total}**\n")
    md.append("## Unit of analysis\n")
    md.append("- Individual requirement (one row = one requirement)\n")
    md.append("## Primary case-study system\n")
    md.append(f"- System: **{primary_system}**\n")
    md.append(f"- Size: **{n_primary}** requirements\n")
    md.append(f"- Class coverage: **{len(classes_primary)}** classes (covers all global classes: **{covers_all_classes}**)\n")
    md.append("## Secondary systems (bounded role)\n")
    md.append("- Used only for sanity checks and small-sample behavior (no cross-system generalization claims)\n")
    for s in secondary_systems:
        md.append(f"  - {s}: {system_counts[s]}")
    md.append("\n## Validation focus (to be executed in later steps)\n")
    md.append("- Construct validity (operational): metric behaviors and expected directional checks\n")
    md.append("- Internal consistency: non-degenerate ranking behavior, coherence across outputs\n")
    md.append("- Robustness: stability under resampling/ablation/parameter perturbations\n")
    md.append("\n## Out of scope\n")
    md.append("- Business value or “true importance” ground truth\n")
    md.append("- Comparison to other prioritization methods (unless explicitly added)\n")
    md.append("- Cost/effort optimization\n")

    out_md = case_study_dir / "case_design.md"
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Wrote: {out_md}")

    # Hard stop if the design assumptions fail
    if not covers_all_classes:
        raise ValueError("Primary system does not cover all classes; revise case study scope.")
    if not primary_size_ok:
        raise ValueError("Primary system too small for robustness tests; revise robustness plan.")


if __name__ == "__main__":
    main()
