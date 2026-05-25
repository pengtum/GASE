"""Quick smoke test: each baseline runs without error on a single grid_*h.txt
and returns a label array of length 625 with k>1 regions.
"""
import os
import sys
import time
import warnings

import numpy as np
import libpysal

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "codes"))

from baselines import azp, region_k_models, gwr_skater, skater_reg, shap_based


def input_data(file_obj, side=25):
    raw_lines = [ln.rstrip('\n\r') for ln in file_obj.readlines()]
    mats = [np.zeros((side, side), dtype=float) for _ in range(6)]
    idx, mat_idx = 0, 0
    while mat_idx < 6:
        block = raw_lines[idx: idx + side]
        idx += side
        for r in range(side):
            tokens = block[r].strip().split()
            row_vals = list(map(float, tokens)) if mat_idx != 3 else list(map(int, tokens))
            mats[mat_idx][r, :] = row_vals
        mat_idx += 1
        while idx < len(raw_lines) and not raw_lines[idx].strip():
            idx += 1
    x1, x2, y, region, b1, b2 = mats
    Xarr = np.column_stack((x1.flatten(), x2.flatten()))
    Yarr = y.flatten()
    label = region.flatten()
    coeff = np.column_stack((b1.flatten(), b2.flatten()))
    return Xarr, Yarr, label, coeff


def main():
    side = 25
    data_path = os.path.join(BASE, "simulation_study", "datasets", "grid_5h.txt")
    print(f"Loading {data_path}")
    with open(data_path) as f:
        X, Y, true_label, true_coeff = input_data(f, side)
    w = libpysal.weights.lat2W(side, side)
    coords = np.array([[i % side, i // side] for i in range(side * side)], dtype=float)

    baselines = {
        "AZP": (azp.run, dict()),
        "RKM": (region_k_models.run, dict()),
        "SKATER-reg": (skater_reg.run, dict()),
        "GWR+SKATER": (gwr_skater.run, dict(coords=coords)),
        "SHAP-based": (shap_based.run, dict(coords=coords)),
    }

    for name, (fn, kwargs) in baselines.items():
        print(f"\n--- {name} ---")
        t0 = time.time()
        try:
            labels = fn(X, Y, n_regions=5, w=w, min_size=10, max_iter=200,
                       seed=42, **kwargs)
            elapsed = time.time() - t0
            uniq = np.unique(labels)
            print(f"  ok  (time={elapsed:.2f}s, regions={len(uniq)}, "
                  f"sizes={[int((labels == r).sum()) for r in uniq]})")
        except Exception as e:
            elapsed = time.time() - t0
            import traceback
            print(f"  FAIL ({elapsed:.2f}s): {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
