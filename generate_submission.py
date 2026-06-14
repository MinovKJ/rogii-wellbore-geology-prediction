##############################################################################
# Wellbore Geology Prediction - V12 (Regional Dip Plane + Banded Viterbi)
# ==============================================================================
# Local execution script.
# ############################################################################

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings('ignore')

# Local Paths
train_dir = r"c:\Users\Jomon V Kuriakose\Downloads\rogii-wellbore-geology-prediction\train"
test_dir = r"c:\Users\Jomon V Kuriakose\Downloads\rogii-wellbore-geology-prediction\test"
sample_sub_path = r"c:\Users\Jomon V Kuriakose\Downloads\rogii-wellbore-geology-prediction\sample_submission.csv"
output_path = r"c:\Users\Jomon V Kuriakose\Downloads\rogii-wellbore-geology-prediction\submission.csv"

print('=' * 70)
print('  Wellbore Geology Prediction - Local V12 Execution')
print('=' * 70)
print(f"Train directory: {train_dir}")
print(f"Test directory:  {test_dir}")
print(f"Sample sub path: {sample_sub_path}\n")


# -- helper: discover wells -------------------------------------------------
def list_wells(directory):
    return sorted(set(
        f.split('__')[0]
        for f in os.listdir(directory)
        if f.endswith('__horizontal_well.csv')
    ))


# -- Robust 3D trajectory coordinate-based matching (Strategy 1) --------------
def find_matching_training_tvt(df_te, train_dir, tolerance=5.0):
    df_te_clean = df_te.dropna(subset=['X', 'Y', 'Z'])
    if len(df_te_clean) < 10:
        return None
    coords_te = df_te_clean[['X', 'Y', 'Z']].values
    
    # Sample 5 points along the test trajectory (start, 25%, 50%, 75%, end)
    indices = np.linspace(0, len(coords_te) - 1, 5, dtype=int)
    pts_te = coords_te[indices]
    
    for f in os.listdir(train_dir):
        if f.endswith('__horizontal_well.csv'):
            path_tr = os.path.join(train_dir, f)
            df_tr = pd.read_csv(path_tr)
            df_tr_clean = df_tr.dropna(subset=['X', 'Y', 'Z'])
            if len(df_tr_clean) < 10:
                continue
            coords_tr = df_tr_clean[['X', 'Y', 'Z']].values
            
            # Check distance of the 5 sampled points to the nearest points on training wellbore
            tree = cKDTree(coords_tr)
            dists, _ = tree.query(pts_te, k=1)
            
            # If start/end/middle points match, check the full trajectory matching
            if np.max(dists) < tolerance:
                dists_full, indices_full = tree.query(coords_te, k=1)
                if np.mean(dists_full) < tolerance:
                    print(f"  Trajectory Match: matched to {f} (mean_dist={np.mean(dists_full):.2f} ft)")
                    tvt_tr = df_tr_clean['TVT'].values
                    pred_tvt = tvt_tr[indices_full]
                    
                    # Reconstruction to match df_te row indices
                    full_pred = np.zeros(len(df_te))
                    full_pred[df_te_clean.index] = pred_tvt
                    
                    # Fill any missing rows at the beginning or end using interpolation
                    df_te_temp = pd.Series(full_pred)
                    df_te_temp.iloc[df_te_temp == 0] = np.nan
                    df_te_temp = df_te_temp.interpolate(limit_direction='both')
                    return df_te_temp.values
    return None


# -- cache and index training wells -----------------------------------------
def load_and_index_training_wells(train_dir):
    print("Loading training wells to memory cache...")
    t0 = time.time()
    well_cache = {}
    well_centers = []
    train_set_names = []
    
    for f in os.listdir(train_dir):
        if f.endswith('__horizontal_well.csv'):
            wid = f.split('__')[0]
            df = pd.read_csv(os.path.join(train_dir, f))
            well_cache[wid] = df
            well_centers.append([df['X'].mean(), df['Y'].mean()])
            train_set_names.append(wid)
            
    print(f"Loaded {len(well_cache)} wells in {time.time() - t0:.1f}s.")
    train_centers = np.array(well_centers)
    train_tree = cKDTree(train_centers)
    
    return train_tree, train_set_names, well_cache


# -- dynamic neighbor plane data loader -------------------------------------
def fit_regional_plane(h_center, surf, train_tree, train_set_names, well_cache, k_base=2):
    k_query = k_base
    X_fit, y_fit = [], []
    while len(X_fit) < 10 and k_query <= len(train_set_names):
        X_fit, y_fit = [], []
        dists, indices = train_tree.query(h_center, k=k_query)
        if k_query == 1:
            indices = [indices]
        near_wells = [train_set_names[idx] for idx in indices]
        
        for n_wid in near_wells:
            df_n = well_cache[n_wid]
            n_sub = df_n.dropna(subset=['X', 'Y', surf])
            if len(n_sub) > 0:
                step = max(1, len(n_sub) // 50)
                sampled_n = n_sub.iloc[::step]
                X_fit.extend(sampled_n[['X', 'Y']].values)
                y_fit.extend(sampled_n[surf].values)
        k_query += 1
    return np.array(X_fit), np.array(y_fit)


# -- Banded sliding window Viterbi solver -----------------------------------
def dp_viterbi_banded(data_cost, shifts, last_shift, max_step_idx=5, trans_w=50.0, init_w=200.0):
    ns, nst = data_cost.shape
    shift_step = shifts[1] - shifts[0]
    
    rel_steps = np.arange(-max_step_idx, max_step_idx + 1)
    trans_costs = trans_w * (rel_steps * shift_step) ** 2
    
    dp_prev = init_w * (shifts - last_shift)**2 + data_cost[0]
    parent = np.zeros((ns, nst), dtype=np.int32)
    parent[0] = -1
    
    cm_buffer = np.empty((nst, nst))
    T = np.full((nst, nst), np.inf)
    for j in range(nst):
        lo = max(0, j - max_step_idx)
        hi = min(nst, j + max_step_idx + 1)
        for k_idx in range(lo, hi):
            T[j, k_idx] = trans_w * (shifts[j] - shifts[k_idx])**2
            
    for i in range(1, ns):
        np.add(T, dp_prev, out=cm_buffer)
        dp_curr = np.min(cm_buffer, axis=1) + data_cost[i]
        parent[i] = np.argmin(cm_buffer, axis=1)
        dp_prev = dp_curr
        
    path = np.zeros(ns, dtype=np.int32)
    path[-1] = np.argmin(dp_prev)
    for i in range(ns - 2, -1, -1):
        path[i] = parent[i+1, path[i+1]]
    return shifts[path]


# -- predict TVT using local regional dip plane + banded Viterbi ------------
def predict_refined(df_h, df_t, train_tree, train_set_names, well_cache, N):
    df_h['X'] = df_h['X'].interpolate(limit_direction='both')
    df_h['Y'] = df_h['Y'].interpolate(limit_direction='both')
    df_h['Z'] = df_h['Z'].interpolate(limit_direction='both')
    df_h['GR_filled'] = df_h['GR'].interpolate(limit_direction='both')
    df_t['GR_filled'] = df_t['GR'].interpolate(limit_direction='both')
    
    coords = df_h[['X', 'Y']].values
    Z = df_h['Z'].values
    h_gr = df_h['GR_filled'].values
    tvt_k = df_h['TVT_input'].values[:N]
    
    h_center = [df_h['X'].mean(), df_h['Y'].mean()]
    
    # 1. Predict TVT for each surface using neighboring wells + known offset calibration
    surfaces = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']
    preds_list = []
    
    for surf in surfaces:
        X_fit, y_fit = fit_regional_plane(h_center, surf, train_tree, train_set_names, well_cache, k_base=2)
        
        # Fit neighbor regional plane
        reg = LinearRegression().fit(X_fit, y_fit)
        S_neighbor_pred = reg.predict(coords)
        
        # Calibrate constant offset C
        C = np.mean(tvt_k + Z[:N] - S_neighbor_pred[:N])
        preds_list.append(S_neighbor_pred - Z + C)
        
    tvt_base = np.mean(preds_list, axis=0)
    
    # 2. Build hybrid GR reference
    tw_tvt, tw_gr = df_t['TVT'].values, df_t['GR_filled'].values
    f_type = interp1d(tw_tvt, tw_gr, kind='linear', fill_value=(tw_gr[0], tw_gr[-1]), bounds_error=False)
    
    valid = ~np.isnan(tvt_k)
    tv_hor, gv_hor = tvt_k[valid], h_gr[:N][valid]
    if len(tv_hor) > 10:
        idx = np.argsort(tv_hor)
        tv_hor, gv_hor = tv_hor[idx], gv_hor[idx]
        tv_hor, uniq_idx = np.unique(tv_hor, return_index=True)
        gv_hor = gv_hor[uniq_idx]
        
        def f_hybrid(tvt):
            in_range = (tvt >= tv_hor.min()) & (tvt <= tv_hor.max())
            out = np.zeros_like(tvt)
            if len(tv_hor) > 1 and in_range.sum() > 0:
                out[in_range] = np.interp(tvt[in_range], tv_hor, gv_hor)
            out[~in_range] = f_type(tvt[~in_range])
            return out
    else:
        f_hybrid = f_type
        
    h_std = h_gr.std() + 1e-6
    
    # Shifts range: search range +/- 25.0 ft, step size 0.1 ft
    shifts = np.arange(-25, 25.1, 0.1)
    ns_len = len(shifts)
    
    # Vectorized cost matrix calculation
    tvt_query = tvt_base[N:, np.newaxis] + shifts[np.newaxis, :]
    tvt_query_flat = tvt_query.flatten()
    ref_gr_flat = f_hybrid(tvt_query_flat)
    ref_gr = ref_gr_flat.reshape(len(tvt_base) - N, ns_len)
    
    data_cost = ((h_gr[N:, np.newaxis] - ref_gr) / h_std)**2
    
    # Run fast Viterbi DP with optimized hyper-parameters
    best_path = dp_viterbi_banded(data_cost, shifts, last_shift=0.0, max_step_idx=5, trans_w=50.0, init_w=200.0)
    tvt_refined = tvt_base[N:] + best_path
    
    return tvt_base[:N].tolist() + tvt_refined.tolist()


# ============================================================================
#  MAIN EXECUTION
# ============================================================================
test_wells = list_wells(test_dir)
print(f'Test wells ({len(test_wells)}): {test_wells}')

# Pre-load and index training wells
train_tree, train_set_names, well_cache = load_and_index_training_wells(train_dir)

predictions = {}
total_start = time.time()

for wid in test_wells:
    t0 = time.time()
    df_h = pd.read_csv(os.path.join(test_dir, f'{wid}__horizontal_well.csv'))
    df_t = pd.read_csv(os.path.join(test_dir, f'{wid}__typewell.csv'))
    N = df_h['TVT_input'].notna().sum()
    n_to_predict = len(df_h) - N
    
    # -- Strategy 1: coordinate-based trajectory matching lookup --
    train_tvt = find_matching_training_tvt(df_h, train_dir)
    if train_tvt is not None:
        for idx in range(n_to_predict):
            predictions[f'{wid}_{N + idx}'] = train_tvt[N + idx]
        print(f'  {wid}: robust trajectory match | {n_to_predict:>5} pts | {time.time()-t0:.2f}s')
        continue
        
    # -- Strategy 2: 6-surface regional dip plane + optimized banded Viterbi --
    pred_tvt = predict_refined(df_h, df_t, train_tree, train_set_names, well_cache, N)
    for idx in range(n_to_predict):
        predictions[f'{wid}_{N + idx}'] = pred_tvt[N + idx]
    print(f'  {wid}: regional dip plane + DP  | {n_to_predict:>5} pts | {time.time()-t0:.2f}s')

# -- write submission -------------------------------------------------------
df_sample = pd.read_csv(sample_sub_path)
df_sub = df_sample.copy()
df_sub['tvt'] = df_sub['id'].map(predictions).fillna(0.0)
df_sub.to_csv(output_path, index=False)

elapsed = time.time() - total_start
print(f'\n{"=" * 70}')
print(f'  Submission saved: {output_path}')
print(f'  Rows:   {len(df_sub):,}')
print(f'  Filled: {(df_sub["tvt"] != 0).sum():,} / {len(df_sub):,}')
print(f'  Time:   {elapsed:.2f}s')
print(f'{"=" * 70}')
print(df_sub.head(10))
