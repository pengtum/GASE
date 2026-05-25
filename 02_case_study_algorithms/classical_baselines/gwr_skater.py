"""GWR + SKATER: estimate local regression coefficients with GWR, then cluster
those coefficients into spatially-contiguous regions with SKATER.

Coefficients are clustered as feature attributes; SKATER ensures contiguity.
"""
import numpy as np
import warnings
import geopandas as gpd
from sklearn import metrics, preprocessing
from mgwr import gwr
from mgwr.sel_bw import Sel_BW
from spopt.region import Skater

from . import _common as C


def run(X, Y, n_regions, w, min_size=10, max_iter=300,
        init_stoc_step=True, verbose=False, coords=None, seed=None):
    """`coords` is required (n,2). If None, infer grid coordinates from sqrt(n)."""
    n = X.shape[0]
    nvar = X.shape[1] - 1  # exclude intercept

    if coords is None:
        side = int(np.sqrt(n))
        if side * side != n:
            raise ValueError("coords must be supplied when n is not a perfect square")
        coords = np.array([[i % side, i // side] for i in range(n)], dtype=float)
    coords = np.asarray(coords, dtype=float)

    # X without intercept for GWR
    Xprm = X[:, 1:].reshape((n, nvar))
    Yprm = Y.reshape((n, 1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bw = Sel_BW(coords, Yprm, Xprm, fixed=False, kernel='bisquare').search(criterion='AICc')
        model = gwr.GWR(coords, Yprm, Xprm, bw=bw, fixed=False, kernel='bisquare')
        results = model.fit()
    coeff = results.params  # shape (n, nvar+1)

    coeff_std = preprocessing.StandardScaler().fit_transform(coeff)
    fields = ['intercept'] + [f'slope{v}' for v in range(nvar)]
    df = gpd.GeoDataFrame(coeff_std, columns=fields, dtype=float)

    spconfig = dict(dissimilarity=metrics.pairwise.manhattan_distances,
                    affinity=None, reduction=np.sum, center=np.mean)
    skater = Skater(df, w, attrs_name=fields, n_clusters=n_regions,
                    floor=min_size, trace=False, spanning_forest_kwds=spconfig)
    skater.solve()
    return C.relabel_sequential(np.asarray(skater.labels_, dtype=int))
