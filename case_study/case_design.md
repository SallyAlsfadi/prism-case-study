# Case Study Design (Step 3)

## Objective

Evaluate PRISM’s prioritization behavior on a realistic requirements dataset without assuming business-value ground truth.

## Dataset basis

- Frozen input: `data/requirements.csv`

- Total requirements: **303**

## Unit of analysis

- Individual requirement (one row = one requirement)

## Primary case-study system

- System: **ecommerce**

- Size: **228** requirements

- Class coverage: **12** classes (covers all global classes: **True**)

## Secondary systems (bounded role)

- Used only for sanity checks and small-sample behavior (no cross-system generalization claims)

  - banking: 26
  - education: 25
  - field_service: 19
  - healthcare: 5

## Validation focus (to be executed in later steps)

- Construct validity (operational): metric behaviors and expected directional checks

- Internal consistency: non-degenerate ranking behavior, coherence across outputs

- Robustness: stability under resampling/ablation/parameter perturbations

## Out of scope

- Business value or “true importance” ground truth

- Comparison to other prioritization methods

- Cost/effort optimization
