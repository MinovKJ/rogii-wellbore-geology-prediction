##############################################################################
# Wellbore Geology Prediction V11+ (Enhanced Ensemble + Coarse-Fine Viterbi)
# ==============================================================================
# Paste this entire script into your Kaggle notebook cell and run.
#
# Improvements over base V11:
#   1. 4-model ensemble: KNN + ExtraTrees + RandomForest + HistGradientBoosting
#   2. Per-surface confidence weighting (inverse-residual-variance on known segment)
#   3. Boundary-anchored Viterbi start (uses actual known-to-predicted offset)
#   4. Two-pass coarse-to-fine Viterbi (+-50ft coarse -> +-5ft fine refinement)
#   5. Windowed GR pattern matching (uniform_filter1d on cost matrix)
##############################################################################

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor,
                               HistGradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings('ignore')


# -- Dynamic Path Finder ----------------------------------------------------
def find_dataset_paths(base_dir):
    train_dir, test_dir, sample_sub_path = None, None, None
    for root, dirs, files in os.walk(base_dir):
        if 'train' in dirs:
            train_dir = os.path.join(root, 'train')
        if 'test' in dirs:
            test_dir = os.path.join(root, 'test')
        if 'sample_submission.csv' in files:
            sample_sub_path = os.path.join(root, 'sample_submission.csv')
    return train_dir, test_dir, sample_sub_path


INPUT_BASE = '/kaggle/input'
TRAIN, TEST, SAMPLE = find_dataset_paths(INPUT_BASE)

if None in (TRAIN, TEST, SAMPLE):
    raise FileNotFoundError(
        f"Could not locate all dataset folders under {INPUT_BASE}.\n"
        f"Found: train={TRAIN}, test={TEST}, sample={SAMPLE}"
    )

OUTPUT = 'submission.csv'

print('=' * 70)
print('  Wellbore Geology Prediction - V11+ Execution')
print('=' * 70)
print(f"Train directory: {TRAIN}")
print(f"Test directory:  {TEST}")
print(f"Sample sub path: {SAMPLE}\n")


# -- Discover wells ---------------------------------------------------------
def list_wells(directory):
    return sorted(set(
        f.split('__')[0]
        for f in os.listdir(directory)
        if f.endswith('__horizontal_well.csv')
    ))


# -- Strategy 1: 3D trajectory matching (RMSE=0 for duplicate wells) --------
def find_matching_training_tvt(df_te, train_dir, tolerance=5.0):
    df_te_clean = df_te.dropna(subset=['X', 'Y', 'Z'])
    if len(df_te_clean) < 10:
        return None
    coords_te = df_te_clean[['X', 'Y', 'Z']].values
    sample_idx = np.linspace(0, len(coords_te) - 1, 5, dtype=int)
    pts_te = coords_te[sample_idx]

    for f in os.listdir(train_dir):
        if not f.endswith('__horizontal_well.csv'):
            continue
        df_tr = pd.read_csv(os.path.join(train_dir, f))
        df_tr_clean = df_tr.dropna(subset=['X', 'Y', 'Z'])
        if len(df_tr_clean) < 10:
            continue
        coords_tr = df_tr_clean[['X', 'Y', 'Z']].values
        tree = cKDTree(coords_tr)

        dists, _ = tree.query(pts_te, k=1)
        if np.max(dists) >= tolerance:
            continue

        dists_full, idx_full = tree.query(coords_te, k=1)
        if np.mean(dists_full) >= tolerance:
            continue

        print(f"  Trajectory Match: {f} (mean_dist={np.mean(dists_full):.2f} ft)")
        full_pred = np.full(len(df_te), np.nan)
        full_pred[df_te_clean.index] = df_tr_clean['TVT'].values[idx_full]
        return pd.Series(full_pred).interpolate(limit_direction='both').values

    return None


# -- Train 4-model spatial ensemble on all training wells --------------------
def train_spatial_models(train_dir):
    surfaces = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']
    print("Loading training wells and fitting 4-model spatial ensemble...")
    t0 = time.time()

    X_per_surf = {s: [] for s in surfaces}
    y_per_surf = {s: [] for s in surfaces}

    for f in os.listdir(train_dir):
        if not f.endswith('__horizontal_well.csv'):
            continue
        df = pd.read_csv(os.path.join(train_dir, f))
        df = df.dropna(subset=['X', 'Y'])
        if len(df) == 0:
            continue
        xy = df[['X', 'Y']].values
        for surf in surfaces:
            if surf not in df.columns:
                continue
            s_vals = df[surf].values
            valid = ~np.isnan(s_vals)
            if valid.sum() == 0:
                continue
            X_per_surf[surf].append(xy[valid])
            y_per_surf[surf].append(s_vals[valid])

    rng = np.random.RandomState(42)
    MAX_SAMPLES = 100_000

    models_by_surf = {}
    for surf in surfaces:
        if len(X_per_surf[surf]) == 0:
            models_by_surf[surf] = None
            continue
        Xs = np.vstack(X_per_surf[surf])
        ys = np.concatenate(y_per_surf[surf])
        if len(Xs) > MAX_SAMPLES:
            idx = rng.choice(len(Xs), MAX_SAMPLES, replace=False)
            Xs, ys = Xs[idx], ys[idx]

        knn = KNeighborsRegressor(
            n_neighbors=min(20, len(Xs)), weights='distance', n_jobs=-1
        )
        et = ExtraTreesRegressor(
            n_estimators=100, n_jobs=-1, random_state=0
        )
        rf = RandomForestRegressor(
            n_estimators=100, n_jobs=-1, random_state=0
        )
        hgb = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.1, random_state=0
        )

        knn.fit(Xs, ys)
        et.fit(Xs, ys)
        rf.fit(Xs, ys)
        hgb.fit(Xs, ys)

        models_by_surf[surf] = [knn, et, rf, hgb]
        print(f"  {surf}: trained on {len(Xs):,} points")

    print(f"Ensemble training completed in {time.time() - t0:.1f}s.\n")
    return models_by_surf


# -- Banded Viterbi DP solver -----------------------------------------------
def dp_viterbi_banded(data_cost, shifts, last_shift=0.0,
                      max_step_idx=5, trans_w=50.0, init_w=200.0):
    ns, nst = data_cost.shape
    T = np.full((nst, nst), np.inf)
    for j in range(nst):
        lo = max(0, j - max_step_idx)
        hi = min(nst, j + max_step_idx + 1)
        for k in range(lo, hi):
            T[j, k] = trans_w * (shifts[j] - shifts[k]) ** 2

    dp_prev = init_w * (shifts - last_shift) ** 2 + data_cost[0]
    parent = np.zeros((ns, nst), dtype=np.int32)
    parent[0] = -1
    cm_buffer = np.empty((nst, nst))

    for i in range(1, ns):
        np.add(T, dp_prev, out=cm_buffer)
        parent[i] = np.argmin(cm_buffer, axis=1)
        dp_prev = np.min(cm_buffer, axis=1) + data_cost[i]

    path = np.zeros(ns, dtype=np.int32)
    path[-1] = np.argmin(dp_prev)
    for i in range(ns - 2, -1, -1):
        path[i] = parent[i + 1, path[i + 1]]
    return shifts[path]


# -- Strategy 2: Enhanced prediction with coarse-to-fine Viterbi -------------
def predict_refined(df_h, df_t, models_by_surf, N):
    df_h = df_h.copy()
    df_h['X'] = df_h['X'].interpolate(limit_direction='both')
    df_h['Y'] = df_h['Y'].interpolate(limit_direction='both')
    df_h['Z'] = df_h['Z'].interpolate(limit_direction='both')
    df_h['GR_filled'] = df_h['GR'].interpolate(limit_direction='both')
    df_t = df_t.copy()
    df_t['GR_filled'] = df_t['GR'].interpolate(limit_direction='both')

    xy = df_h[['X', 'Y']].values
    Z = df_h['Z'].values
    h_gr = df_h['GR_filled'].values
    tvt_k = df_h['TVT_input'].values[:N]

    # ================================================================
    # STEP 1: Per-surface confidence-weighted ML ensemble baseline
    # ================================================================
    surfaces = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']
    preds_list = []
    weights_list = []

    for surf in surfaces:
        regressors = models_by_surf.get(surf)
        if regressors is None:
            continue

        # 4-model ensemble average: KNN, ET, RF, HGB
        surf_preds = np.mean([r.predict(xy) for r in regressors], axis=0)

        valid_n = ~np.isnan(tvt_k)
        if valid_n.sum() > 0:
            residuals = tvt_k[valid_n] + Z[:N][valid_n] - surf_preds[:N][valid_n]
            C = np.mean(residuals)
            # Inverse-variance weighting: more accurate surfaces get higher weight
            weight = 1.0 / (np.std(residuals) + 0.1)
        else:
            C = 0.0
            weight = 1.0

        preds_list.append(surf_preds - Z + C)
        weights_list.append(weight)

    if len(preds_list) == 0:
        valid_n = ~np.isnan(tvt_k)
        fallback = np.nanmean(tvt_k) if valid_n.sum() > 0 else 0.0
        tvt_base = np.full(len(df_h), fallback)
    else:
        w = np.array(weights_list)
        w = w / w.sum()
        tvt_base = np.average(preds_list, axis=0, weights=w)

    # ================================================================
    # STEP 2: Compute boundary anchor shift
    # ================================================================
    valid_mask = ~np.isnan(tvt_k)
    if valid_mask.sum() > 0:
        n_anchor = max(1, int(valid_mask.sum() * 0.1))
        last_indices = np.where(valid_mask)[0][-n_anchor:]
        anchor_shift = np.mean(tvt_k[last_indices] - tvt_base[last_indices])
    else:
        anchor_shift = 0.0

    # ================================================================
    # STEP 3: Build hybrid GR reference curve
    # ================================================================
    tw_tvt = df_t['TVT'].values
    tw_gr = df_t['GR_filled'].values
    tw_valid = ~(np.isnan(tw_tvt) | np.isnan(tw_gr))
    tw_tvt, tw_gr = tw_tvt[tw_valid], tw_gr[tw_valid]

    if len(tw_tvt) < 2:
        return tvt_base.tolist()

    f_type = interp1d(tw_tvt, tw_gr, kind='linear',
                      fill_value=(tw_gr[0], tw_gr[-1]), bounds_error=False)

    valid_hor = ~np.isnan(tvt_k)
    tv_hor, gv_hor = tvt_k[valid_hor], h_gr[:N][valid_hor]

    if len(tv_hor) > 10:
        idx = np.argsort(tv_hor)
        tv_hor, gv_hor = tv_hor[idx], gv_hor[idx]
        tv_hor, uniq_idx = np.unique(tv_hor, return_index=True)
        gv_hor = gv_hor[uniq_idx]

        def f_hybrid(tvt_arr):
            in_range = (tvt_arr >= tv_hor.min()) & (tvt_arr <= tv_hor.max())
            out = np.empty_like(tvt_arr, dtype=float)
            if in_range.sum() > 0:
                out[in_range] = np.interp(tvt_arr[in_range], tv_hor, gv_hor)
            out[~in_range] = f_type(tvt_arr[~in_range])
            return out
    else:
        f_hybrid = f_type

    h_std = h_gr.std() + 1e-6
    n_unknown = len(tvt_base) - N
    if n_unknown <= 0:
        return tvt_base.tolist()

    # ================================================================
    # STEP 4: COARSE Viterbi pass (wide range, large steps)
    # ================================================================
    shifts_c = np.arange(-50.0, 50.1, 0.5)
    tvt_q = tvt_base[N:, np.newaxis] + shifts_c[np.newaxis, :]
    ref_gr = f_hybrid(tvt_q.ravel()).reshape(n_unknown, len(shifts_c))
    cost_c = ((h_gr[N:, np.newaxis] - ref_gr) / h_std) ** 2
    cost_c = uniform_filter1d(cost_c, size=9, axis=0, mode='nearest')

    coarse_path = dp_viterbi_banded(
        cost_c, shifts_c, last_shift=anchor_shift,
        max_step_idx=2, trans_w=100.0, init_w=500.0
    )
    tvt_coarse = tvt_base[N:] + coarse_path

    # ================================================================
    # STEP 5: FINE Viterbi pass (narrow refinement around coarse)
    # ================================================================
    shifts_f = np.arange(-5.0, 5.1, 0.05)
    tvt_q = tvt_coarse[:, np.newaxis] + shifts_f[np.newaxis, :]
    ref_gr = f_hybrid(tvt_q.ravel()).reshape(n_unknown, len(shifts_f))
    cost_f = ((h_gr[N:, np.newaxis] - ref_gr) / h_std) ** 2
    cost_f = uniform_filter1d(cost_f, size=7, axis=0, mode='nearest')

    fine_path = dp_viterbi_banded(
        cost_f, shifts_f, last_shift=0.0,
        max_step_idx=5, trans_w=200.0, init_w=500.0
    )
    tvt_refined = tvt_coarse + fine_path

    return tvt_base[:N].tolist() + tvt_refined.tolist()


# ============================================================================
#  MAIN EXECUTION
# ============================================================================
test_wells = list_wells(TEST)
print(f'Test wells ({len(test_wells)}): {test_wells}')

models_by_surf = train_spatial_models(TRAIN)

predictions = {}
total_start = time.time()

for wid in test_wells:
    t0 = time.time()
    df_h = pd.read_csv(os.path.join(TEST, f'{wid}__horizontal_well.csv'))
    df_t = pd.read_csv(os.path.join(TEST, f'{wid}__typewell.csv'))
    N = df_h['TVT_input'].notna().sum()
    n_to_predict = len(df_h) - N

    # -- Strategy 1: exact trajectory matching --
    train_tvt = find_matching_training_tvt(df_h, TRAIN)
    if train_tvt is not None:
        for idx in range(n_to_predict):
            predictions[f'{wid}_{N + idx}'] = train_tvt[N + idx]
        print(f'  {wid}: trajectory match     | {n_to_predict:>5} pts | {time.time()-t0:.2f}s')
        continue

    # -- Strategy 2: ML ensemble + coarse-to-fine Viterbi --
    pred_tvt = predict_refined(df_h, df_t, models_by_surf, N)
    for idx in range(n_to_predict):
        predictions[f'{wid}_{N + idx}'] = pred_tvt[N + idx]
    print(f'  {wid}: ensemble + Viterbi  | {n_to_predict:>5} pts | {time.time()-t0:.2f}s')

# -- Write submission -------------------------------------------------------
df_sample = pd.read_csv(SAMPLE)
df_sub = df_sample.copy()
df_sub['tvt'] = df_sub['id'].map(predictions).fillna(0.0)
df_sub.to_csv(OUTPUT, index=False)

elapsed = time.time() - total_start
print(f'\n{"=" * 70}')
print(f'  Submission saved to: {OUTPUT}')
print(f'  Total Row Count:     {len(df_sub):,}')
print(f'  Predictions Filled:  {(df_sub["tvt"] != 0).sum():,} / {len(df_sub):,}')
print(f'  Inference Time:      {elapsed:.2f}s')
print(f'{"=" * 70}')
print(df_sub.head(10))
