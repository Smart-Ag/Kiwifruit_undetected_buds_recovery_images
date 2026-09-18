"""
Local skeleton-radial cleaning of a raw branch point cloud.
===========================================================
Removes off-tube protrusions (reconstruction spikes that break the branch
continuation) while preserving the tube and modest bud bumps.

Method
------
1. Fit an L1-medial skeleton (follows the curved branch axis robustly; the
   off-axis spike is a local minority so it does not pull the axis).
2. Resample to an arc-length centerline with tangents (order_skeleton).
3. For each cloud point, assign it to the nearest centerline station and
   measure its *perpendicular* radial distance to the local axis.
4. Estimate the local tube radius from a rolling window of stations
   (median + K*MAD, robust to the spike's own points). Drop points whose
   radial distance exceeds the local threshold.

Because the threshold is *local*, it adapts to the branch tapering from a
thick budded base to a thin tip, and a single global cut is avoided.

Usage:
    python clean_branch_radial.py pc_out/branch4_raw.ply
    python clean_branch_radial.py pc_out/branch4_raw.ply --K 4 --out pc_out/branch4_clean.ply
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from l1_skeleton import l1_medial_skeleton, order_skeleton


def clean(path, out=None, n_samples=180, h=0.045, n_stations=260,
          win=10, K=4.0, floor=0.004):
    path = Path(path)
    if out is None:
        out = path.with_name(path.stem.replace("_raw", "") + "_clean.ply")
    out = Path(out)

    pcd = o3d.io.read_point_cloud(str(path))
    P = np.asarray(pcd.points)
    C = np.asarray(pcd.colors) if pcd.has_colors() else None
    print(f"[load] {path.name}: {len(P)} points")

    # 1-2. skeleton + arc-length centerline
    print("[skel] fitting L1-medial skeleton ...")
    skel = l1_medial_skeleton(P, n_samples=n_samples, h=h)
    stations, arc, tang = order_skeleton(skel, n_stations=n_stations)
    print(f"[skel] centerline: {len(stations)} stations, arc length {arc[-1]:.3f}")

    # 3. assign each point to nearest station, perpendicular radial distance
    stree = cKDTree(stations)
    _, sidx = stree.query(P)
    v = P - stations[sidx]
    t = tang[sidx]
    along = np.einsum("ij,ij->i", v, t)
    perp = v - along[:, None] * t
    r = np.linalg.norm(perp, axis=1)

    # 4. local tube radius per station from a rolling window of stations,
    #    pooling the radial values of points assigned to those stations.
    order = np.argsort(sidx)
    sidx_s = sidx[order]
    r_s = r[order]
    # start index of each station block in the sorted arrays
    starts = np.searchsorted(sidx_s, np.arange(n_stations))
    ends = np.searchsorted(sidx_s, np.arange(n_stations) + 1)

    med = np.full(n_stations, np.nan)
    mad = np.full(n_stations, np.nan)
    for s in range(n_stations):
        lo = max(0, s - win)
        hi = min(n_stations, s + win + 1)
        pooled = r_s[starts[lo]:ends[hi - 1]]
        pooled = pooled[~np.isnan(pooled)]
        if len(pooled) >= 5:
            m = np.median(pooled)
            med[s] = m
            mad[s] = np.median(np.abs(pooled - m))
    # fill empty stations by interpolation
    good = ~np.isnan(med)
    xs = np.arange(n_stations)
    med = np.interp(xs, xs[good], med[good])
    mad = np.interp(xs, xs[good], mad[good])

    thr_station = med + K * 1.4826 * mad
    thr_station = np.maximum(thr_station, med + floor)  # avoid zero-spread bins
    thr = thr_station[sidx]

    removed = r > thr
    kept = ~removed
    print(f"[trim] K={K}: removed {removed.sum()} / {len(P)} points "
          f"({100*removed.sum()/len(P):.1f}%)")
    if removed.any():
        rp = P[removed]
        print(f"[trim] removed bbox x[{rp[:,0].min():.3f},{rp[:,0].max():.3f}] "
              f"z[{rp[:,2].min():.3f},{rp[:,2].max():.3f}]")

    # save cleaned cloud
    clean_pcd = o3d.geometry.PointCloud()
    clean_pcd.points = o3d.utility.Vector3dVector(P[kept])
    if C is not None:
        clean_pcd.colors = o3d.utility.Vector3dVector(C[kept])
    o3d.io.write_point_cloud(str(out), clean_pcd)
    print(f"[save] {out}  ({kept.sum()} points)")

    # visualization: kept (grey) vs removed (red) + centerline, 3 projections
    fig, ax = plt.subplots(1, 3, figsize=(21, 6))
    views = [(0, 2, "X-Z (side)"), (0, 1, "X-Y (top)"), (1, 2, "Y-Z (cross)")]
    for a, (i, j, ttl) in zip(ax, views):
        a.scatter(P[kept, i], P[kept, j], s=1, c="0.6", label="kept")
        a.scatter(P[removed, i], P[removed, j], s=3, c="red", label="removed")
        a.plot(stations[:, i], stations[:, j], "-", c="dodgerblue", lw=1.2,
               label="centerline")
        a.set_title(ttl); a.set_aspect("equal")
        a.set_xlabel("xyz"[i]); a.set_ylabel("xyz"[j])
    ax[0].legend(markerscale=4, loc="best")
    fig.suptitle(f"{path.name} -> {out.name}   removed {removed.sum()} pts "
                 f"(K={K})")
    fig.tight_layout()
    viz = out.with_name(out.stem + "_cleaning.png")
    fig.savefig(viz, dpi=110)
    print(f"[viz ] {viz}")
    return out, viz


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default=None)
    ap.add_argument("--K", type=float, default=4.0,
                    help="outlier strictness: lower removes more")
    ap.add_argument("--win", type=int, default=10,
                    help="rolling window half-width in stations")
    args = ap.parse_args()
    clean(args.input, out=args.out, K=args.K, win=args.win)
