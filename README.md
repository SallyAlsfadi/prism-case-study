# PRISM Case Study Replication Package

This repository is a **case-study-only** replication package for running and validating PRISM on a frozen dataset.

## Scope
- Freeze and verify dataset integrity
- Profile dataset structure
- Execute PRISM (via your existing PRISM scripts)
- Run validation checks (internal consistency, robustness, sensitivity)

## Out of scope
- This repository is **not** the PRISM framework implementation.
- It does **not** claim business-value ground truth.
- It does **not** benchmark against other prioritization methods unless explicitly added later.

## Quick start
1) Create a virtual environment and install dependencies:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`

2) Add the frozen dataset:
   - Copy `requirements.csv` into `data/requirements.csv`

3) Run Step 1 (data integrity):
   - `python steps/step01_data_integrity/check_data.py --config config/config.yaml`

Outputs will be written to `results/logs/`.
