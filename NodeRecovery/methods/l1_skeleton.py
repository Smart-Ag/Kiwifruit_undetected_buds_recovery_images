"""
L1-medial curve skeleton (Huang et al., SIGGRAPH Asia 2013).
============================================================
Contracts a set of sample points toward the local L1-median of nearby cloud
points, with an inter-sample repulsion term that spreads samples along the
structure. Unlike global-axis methods this follows arbitrary curvature, so it
traces a curved branch correctly.

Public API:
    skeleton_pts            = l1_medial_skeleton(points, ...)
    ordered, arc, tangents  = order_skeleton(skeleton_pts, n_stations)
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import (minimum_spanning_tree, shortest_path,
                                  connected_components)


def l1_medial_skeleton(points, n_samples=160, h=0.04, mu=0.35,
                       n_iter=40, seed=0):
    """
    Args:
        points    : (N,3) cloud points
        n_samples : number of skeleton sample points
        h         : neighbourhood radius (m) -- the support scale
        mu        : repulsion weight (0..0.5)
        n_iter    : contraction iterations
    Returns:
        (M,3) skeleton points
    """
    rng = np.random.default_rng(seed)
    P = points
    # Initialise samples as a random subset of the cloud
    idx = rng.choice(len(P), size=min(n_samples, len(P)), replace=False)
    X = P[idx].copy()

    h2 = (h / 2.0) ** 2
    eps = 1e-9

    for _ in range(n_iter):
        ptree = cKDTree(P)
        xtree = cKDTree(X)
        newX = X.copy()

        for i in range(len(X)):
            xi = X[i]

            # --- attraction term: weighted L1-median of nearby cloud points ---
            pj_idx = ptree.query_ball_point(xi, h)
            if len(pj_idx) < 3:
                continue
            pj = P[pj_idx]
            d = np.linalg.norm(pj - xi, axis=1) + eps
            theta = np.exp(-(d ** 2) / h2)
            alpha = theta / d
            attract = (alpha[:, None] * pj).sum(0) / (alpha.sum() + eps)

            # --- repulsion term: push away from other samples ---
            xj_idx = xtree.query_ball_point(xi, h)
            xj_idx = [j for j in xj_idx if j != i]
            repulse = np.zeros(3)
            if len(xj_idx) >= 1:
                xj = X[xj_idx]
                dx = xi - xj
                dd = np.linalg.norm(dx, axis=1) + eps
                thetax = np.exp(-(dd ** 2) / h2)
                beta = thetax / (dd ** 2)
                repulse = (beta[:, None] * dx).sum(0) / (beta.sum() + eps)

            newX[i] = attract + mu * repulse

        X = newX

    return X


def order_skeleton(skel_pts, n_stations, k=6):
    """
    Order skeleton points into a single arc-length-parametrised polyline by
    building a kNN graph, taking its MST, finding the two graph-diameter
    endpoints, and tracing the path between them. Then resample to n_stations.

    Returns:
        stations  : (n_stations,3) resampled centerline points
        arc       : (n_stations,) arc length at each station
        tangents  : (n_stations,3) unit tangents
    """
    M = len(skel_pts)
    tree = cKDTree(skel_pts)
    dists, nbrs = tree.query(skel_pts, k=min(k + 1, M))

    # kNN graph (symmetric), edge weight = euclidean distance
    rows, cols, vals = [], [], []
    for i in range(M):
        for jd, j in zip(dists[i, 1:], nbrs[i, 1:]):
            rows.append(i); cols.append(j); vals.append(jd)
    G = csr_matrix((vals, (rows, cols)), shape=(M, M))
    G = G.maximum(G.T)                       # symmetrise

    # Force connectivity: the kNN graph can split into components across a
    # data gap. Bridge components by their nearest inter-component point pair.
    n_comp, comp = connected_components(G, directed=False)
    while n_comp > 1:
        # link component 0 to the globally nearest point in any other component
        in0 = np.where(comp == 0)[0]
        out = np.where(comp != 0)[0]
        sub = cKDTree(skel_pts[out])
        d, j = sub.query(skel_pts[in0])
        a_local = int(np.argmin(d))
        i0, i1 = in0[a_local], out[j[a_local]]
        bridge = csr_matrix(([d[a_local], d[a_local]], ([i0, i1], [i1, i0])),
                            shape=(M, M))
        G = G.maximum(bridge)
        n_comp, comp = connected_components(G, directed=False)

    mst = minimum_spanning_tree(G)
    mst = mst.maximum(mst.T)

    # Graph diameter via double shortest-path sweep
    d0 = shortest_path(mst, indices=0)
    a = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    da, preds = shortest_path(mst, indices=a, return_predecessors=True)
    b = int(np.argmax(np.where(np.isfinite(da), da, -1)))

    # Trace path a -> b
    path = []
    cur = b
    while cur != a and cur >= 0:
        path.append(cur)
        cur = preds[cur]
    path.append(a)
    path = path[::-1]
    ordered = skel_pts[path]

    # Smooth the ordered polyline to remove skeleton wiggle (better tangents)
    if len(ordered) >= 5:
        win = 5
        k = np.ones(win) / win
        sm = np.column_stack([
            np.convolve(ordered[:, d], k, mode="same") for d in range(3)
        ])
        # keep endpoints unsmoothed (convolve distorts them)
        sm[0], sm[-1] = ordered[0], ordered[-1]
        sm[1], sm[-2] = ordered[1], ordered[-2]
        ordered = sm

    # Arc-length resample
    seg = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    cum = np.insert(np.cumsum(seg), 0, 0.0)
    total = cum[-1]
    s_targets = np.linspace(0, total, n_stations)

    stations = np.empty((n_stations, 3))
    tangents = np.empty((n_stations, 3))
    for i, s in enumerate(s_targets):
        j = np.searchsorted(cum, s)
        j = np.clip(j, 1, len(ordered) - 1)
        f = (s - cum[j - 1]) / (cum[j] - cum[j - 1] + 1e-12)
        stations[i] = ordered[j - 1] + f * (ordered[j] - ordered[j - 1])
        t = ordered[j] - ordered[j - 1]
        tangents[i] = t / (np.linalg.norm(t) + 1e-12)

    return stations, s_targets, tangents
