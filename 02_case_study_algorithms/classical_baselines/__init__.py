"""Runnable baseline implementations for spatial regime / regionalization.

Each module exposes:
    run(X, Y, n_regions, w, min_size=10, max_iter=300, init_stoc_step=True,
        verbose=False, coords=None) -> labels (1-D np.ndarray, length n_samples)
"""
