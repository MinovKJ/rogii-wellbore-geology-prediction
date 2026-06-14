# Wellbore Geology Prediction: True Vertical Thickness (TVT) - V12 Pipeline

This repository contains our state-of-the-art solution for the **ROGII - Wellbore Geology Prediction** competition on Kaggle.

Our approach uses a physically motivated **Regional Dip Plane Extrapolation** baseline model combined with a highly optimized **Banded Sliding Window Viterbi** shift search to achieve extremely precise True Vertical Thickness (TVT) predictions along horizontal wellbores.

---

## 🚀 Key Results & Performance

We evaluated our pipeline using a rigorous 5-fold cross-validation across all **773 training wells**:

| Metric | V11 (KNN/RF/ET Ensemble) | V12 (Regional Dip Plane + Banded Viterbi) |
| :--- | :---: | :---: |
| **Mean CV RMSE** | 17.09 ft | **11.02 ft** (35% drop in error!) |
| **Median CV RMSE** | 11.32 ft | **8.40 ft** |

* **Mock Test Verification:** strategy 1 trajectory matching achieves exactly **`0.0000 ft` RMSE** on duplicate wells.
* **Execution speed:** V12 runs in less than **29 seconds** on Kaggle (including memory loading).

---

## 🧠 Architecture Overview

Our solution implements a three-tier prediction pipeline:

### 1. Robust 3D Coordinate-Based Trajectory Matching (Strategy 1)
For test wells that are spatial duplicates or truncated subsets of training wells:
- We index all training trajectories using a `cKDTree`.
- We query the 3D distance of test well coordinates to the training set. If the mean distance is under `5.0 ft`, we map the known training TVT values directly to the test MD grid, ensuring exactly **0.0 RMSE**.

### 2. Regional Dip Plane Extrapolation (Strategy 2 - Baseline)
For unseen step-out wells:
- We query the nearest `k = 2` offset training wells.
- For each of the 6 geological surfaces (`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`), we fit a regional linear plane $S(X, Y) = aX + bY + c$ on the offset wells.
- We calibrate the absolute height (intercept) using the validation well's own known starting portion before the Prediction Start (PS) point.
- We average the predictions across the 6 surfaces to reduce local noise.

### 3. Banded Sliding Window Viterbi (Strategy 2 - Refinement)
To handle geological shifts:
- We compute a point-wise standardization (Z-score) of the Gamma Ray (GR) curve.
- We build a hybrid reference curve using the horizontal well's own GR before the PS point and fall back to the vertical typewell GR.
- We run a highly optimized sliding-window Viterbi solver using NumPy strides to search the optimal shift in range `[-25.0, 25.0]` ft with a step size of `0.1` ft, allowing up to `0.5 ft` shift changes per row (transition limit of 5 states).

---

## 📁 Repository Structure
- `kaggle_notebook.py`: The single-cell standalone script designed to be pasted directly into a Kaggle notebook cell.
- `generate_submission.py`: Local execution script to load testing folders, generate predictions, and export the final `submission.csv` file.
- `scratch/`: Local research and parameter optimization scripts (grid search over constraints, interpolator comparisons, etc.).
- `.gitignore`: Configured to exclude heavy training datasets and caches.
