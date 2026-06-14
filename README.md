# Wellbore Geology Prediction: True Vertical Thickness (TVT)

This repository contains an optimized analytical pipeline for the ROGII Wellbore Geology Prediction challenge. The objective is to predict True Vertical Thickness (TVT) along horizontal wellbores using spatial data and Gamma Ray (GR) measurements.

The implementation utilizes a hybrid approach: **Regional Dip Plane Extrapolation** to estimate the structural baseline of the geological layers, followed by a **Banded Sliding Window Viterbi** search to align local Gamma Ray curves with vertical typewells.

---

## Performance Summary

The pipeline was validated using a 5-fold cross-validation scheme across all 773 training wells. The transition from non-linear machine learning ensembles to regional dip plane baselines resulted in a significant reduction in prediction error:

| Metric | Machine Learning Ensemble (V11) | Regional Dip Plane + Banded Viterbi (V12) |
| :--- | :---: | :---: |
| **Mean Cross-Validation RMSE** | 17.09 ft | **11.02 ft** |
| **Median Cross-Validation RMSE** | 11.32 ft | **8.40 ft** |

* **Local Verification:** Trajectory coordinate matching achieves exactly `0.0000 ft` RMSE on duplicate test wells.
* **Inference Time:** The entire pipeline completes execution in less than 29 seconds on Kaggle.

---

## Technical Architecture

The prediction pipeline consists of three sequential strategies:

### 1. 3D Trajectory Coordinate Matching (Strategy 1)
For test wells that represent duplicate wellbores or direct subsets of training trajectories:
- Wellbore geometries are indexed using a spatial KD-Tree (`scipy.spatial.cKDTree`).
- The pipeline checks if the test trajectory lies within a 5.0 ft distance threshold of any training well.
- If a match is found, the known training TVT sequence is copied directly, guaranteeing an exact matching lookup (0.0 RMSE).

### 2. Regional Dip Plane Extrapolation (Strategy 2 - Baseline)
For step-out wells where geometry is unseen:
- The pipeline identifies the $k = 2$ nearest neighbor training wells based on wellbore centroids.
- For each of the 6 geological formations (`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`), a linear plane $S(X,Y) = aX + bY + c$ is fitted to estimate the regional dip slopes ($a$ and $b$).
- The vertical offset $C_{surf}$ is calibrated using the known vertical thickness before the Prediction Start (PS) point:
  $$C_{surf} = \text{mean}(\text{TVT}_{known} + Z_{known} - S_{neighbor}(X_{known}, Y_{known}))$$
- The baseline TVT is computed as:
  $$\text{TVT}_{base} = S_{neighbor}(X, Y) - Z + C_{surf}$$
- The 6 independent surface predictions are averaged to minimize local variance.

### 3. Banded Sliding Window Viterbi (Strategy 2 - Refinement)
To handle structural changes:
- Horizontal wellbore GR and typewell GR measurements are standardized (Z-score).
- A hybrid reference log is generated using the horizontal well's own GR before the PS point, falling back to the vertical typewell GR outside this range.
- A banded Viterbi dynamic programming solver is executed using NumPy's `sliding_window_view`. 
- The search space covers a shift range of $[-25.0, 25.0]$ ft with a step size of $0.1$ ft (501 states). The transition step limit is restricted to 5 states ($0.5$ ft change per row), optimized to prevent non-physical structural discontinuities.

---

## Project Structure
- `kaggle_notebook.py`: Standalone, single-cell script ready for deployment on Kaggle.
- `generate_submission.py`: Local submission generation script.
- `scratch/`: Research scripts containing hyperparameter tuning, interpolator benchmarks, and validation logs.
- `.gitignore`: Configured to ignore large training files, PowerPoint documents, and Python environment artifacts.
