"""
Centerline-anchored depth (Z) denoising for a ZED-reconstructed branch cloud.
=============================================================================
The cloud is a ~2.5-D depth shell: the image plane (X,Y) is crisp, but the
depth axis Z is stereo noise -- 1 mm quantized and ~3x the lateral spread --
which smears the branch surface in the side (X-Z) view.

We keep the genuine 3-D curviness (the slow Z trend along the branch) and strip
only the high-frequency Z jitter:

  1. Fit a smooth arc-length centerline (L1 skeleton, reliable because X-Y is
     clean) and give every point a surface coordinate (s = arc length,
     v = lateral offset across the branch).
  2. Replace each point's Z by a BILATERAL weighted mean of nearby points:
       w = exp(-ds^2/2 sig_s^2) * exp(-dv^2/2 sig_v^2) * exp(-dz^2/2 sig_r^2)
     The (s,v) terms keep smoothing local so curvature is preserved; the Z
     range term (sig_r) stops the folded base (front+back layers) from blending.
  X and Y are left exactly as-is.

Saves the denoised cloud (colors preserved) + before/after visualization.

Usage:
    python denoise_depth.py
    python denoise_depth.py --sig-s 0.012 --sig-v 0.006 --sig-r 0.008 --iters 2
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


def _smooth_polyline(poly, win):
    """Moving-average smoothing of an ordered polyline; endpoints preserved."""
    if win < 3:
        return poly
    k = np.ones(win) / win
    sm = np.column_stack([np.convolve(poly[:, d], k, mode="same") for d in range(3)])
    # convolution distorts the ends -> keep the original endpoints/near-ends
    half = win // 2
    sm[:half] = poly[:half]
    sm[-half:] = poly[-half:]
    return sm


def surface_coords(P, n_samples=180, h=0.045, n_stations=320, skel_smooth=0):
    """
    Per-point (arc-length s, lateral offset v) from a smooth centerline.

    Stiffness knobs (resist cloud waviness):
        h           : L1 support radius -- larger = stiffer skeleton (main lever)
        n_samples   : fewer skeleton samples = fewer DOF to wiggle
        skel_smooth : extra moving-average window on the ordered centerline
    """
    skel = l1_medial_skeleton(P, n_samples=n_samples, h=h)
    stations, arc, tang = order_skeleton(skel, n_stations=n_stations)
    if skel_smooth and skel_smooth >= 3:
        stations = _smooth_polyline(stations, skel_smooth)
        # recompute arc length and tangents from the stiffened centerline
        seg = np.linalg.norm(np.diff(stations, axis=0), axis=1)
        arc = np.insert(np.cumsum(seg), 0, 0.0)
        tang = np.gradient(stations, axis=0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
    tree = cKDTree(stations)
    _, idx = tree.query(P)
    s = arc[idx]
    # in-plane (XY) normal at each station: rotate the XY tangent by 90 deg
    txy = tang[idx, :2]
    txy /= np.linalg.norm(txy, axis=1, keepdims=True) + 1e-12
    nxy = np.column_stack([-txy[:, 1], txy[:, 0]])
    v = np.einsum("ij,ij->i", P[:, :2] - stations[idx, :2], nxy)
    return s, v, stations, arc, skel


def denoise(path="case/in/clean.ply", out=None,
            sig_s=0.012, sig_v=0.006, sig_r=0.008, iters=2,
            h=0.045, n_samples=180, skel_smooth=0):
    path = Path(path)
    if out is None:
        out = path.with_name(path.stem.replace("_clean", "") + "_smooth.ply")
    out = Path(out)

    pcd = o3d.io.read_point_cloud(str(path))
    P = np.asarray(pcd.points).copy()
    C = np.asarray(pcd.colors) if pcd.has_colors() else None
    print(f"[load] {path.name}: {len(P)} points")

    s, v, stations, arc, skel = surface_coords(
        P, n_samples=n_samples, h=h, skel_smooth=skel_smooth)
    print(f"[skel] h={h} n_samples={n_samples} skel_smooth={skel_smooth}")
    print(f"[skel] {len(skel)} raw skeleton pts, {len(stations)} ordered stations, "
          f"arc length {arc[-1]:.3f}")

    # KDTree in scaled (s, v) surface space; radius covers ~3 sigma
    scale = np.array([1.0 / sig_s, 1.0 / sig_v])
    SV = np.column_stack([s, v]) * scale
    tree = cKDTree(SV)
    rad = 3.0
    z = P[:, 2].copy()
    z0 = z.copy()
    for it in range(iters):
        znew = z.copy()
        nbrs = tree.query_ball_point(SV, r=rad)
        for i, nb in enumerate(nbrs):
            if len(nb) < 2:
                continue
            nb = np.asarray(nb)
            d2 = ((SV[nb] - SV[i]) ** 2).sum(1)          # already scaled
            dz = z[nb] - z[i]
            w = np.exp(-0.5 * d2) * np.exp(-0.5 * (dz / sig_r) ** 2)
            znew[i] = (w * z[nb]).sum() / w.sum()
        z = znew
        print(f"[pass {it+1}] mean absolute depth change = {np.mean(np.abs(z - z0)):.4f}")
    P[:, 2] = z

    # report noise reduction: per-slab Z std before vs after
    x = P[:, 0]
    def slab_std(zz):
        out = []
        for lo in np.arange(x.min(), x.max(), 0.05):
            m = (x >= lo) & (x < lo + 0.05)
            if m.sum() >= 20:
                out.append(zz[m].std())
        return np.median(out)
    print(f"[noise] median per-slab Z std: {slab_std(z0):.4f} -> {slab_std(z):.4f}")

    sm = o3d.geometry.PointCloud()
    sm.points = o3d.utility.Vector3dVector(P)
    if C is not None:
        sm.colors = o3d.utility.Vector3dVector(C)
    o3d.io.write_point_cloud(str(out), sm)
    print(f"[save] {out}")

    # save skeleton (ordered centerline stations) as a red point cloud
    skel_out = out.with_name(out.stem.replace("_smooth", "") + "_skeleton.ply")
    skp = o3d.geometry.PointCloud()
    skp.points = o3d.utility.Vector3dVector(stations)
    skp.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (len(stations), 1)))
    o3d.io.write_point_cloud(str(skel_out), skp)
    print(f"[save] {skel_out}  ({len(stations)} centerline points)")

    # visualization: smoothed cloud with the skeleton drawn over it, 3 views + zoom
    def draw(a, i, j, title, zoom=None):
        pts = P
        if zoom is not None:
            m = (P[:, 0] > zoom[0]) & (P[:, 0] < zoom[1]); pts = P[m]
            sk = stations[(stations[:, 0] > zoom[0]) & (stations[:, 0] < zoom[1])]
        else:
            sk = stations
        a.scatter(pts[:, i], pts[:, j], s=2, c="0.7", label="cloud")
        a.plot(sk[:, i], sk[:, j], "-", c="crimson", lw=1.4, zorder=3)
        a.scatter(sk[:, i], sk[:, j], s=6, c="crimson", zorder=4, label="skeleton")
        a.set_title(title); a.set_aspect("equal")
        a.set_xlabel("xyz"[i]); a.set_ylabel("xyz"[j])

    fig, ax = plt.subplots(2, 2, figsize=(18, 11))
    draw(ax[0, 0], 0, 2, "X-Z (side) + skeleton")
    draw(ax[0, 1], 0, 1, "X-Y (top) + skeleton")
    draw(ax[1, 0], 1, 2, "Y-Z (cross) + skeleton")
    draw(ax[1, 1], 0, 2, "X-Z zoom + skeleton", zoom=(-0.65, -0.25))
    ax[0, 0].legend(markerscale=2, loc="best")
    fig.suptitle(f"{out.name} with {len(stations)}-pt skeleton ({skel_out.name})",
                 fontsize=13)
    fig.tight_layout()
    viz = out.with_name(out.stem + "_skeleton.png")
    fig.savefig(viz, dpi=110)
    print(f"[viz ] {viz}")
    return out, viz


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="case/in/clean.ply")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sig-s", type=float, default=0.018)
    ap.add_argument("--sig-v", type=float, default=0.008)
    ap.add_argument("--sig-r", type=float, default=0.012)
    ap.add_argument("--iters", type=int, default=3)
    # skeleton stiffness knobs
    ap.add_argument("--h", type=float, default=0.12, # 0.045
                    help="L1 support radius; larger = stiffer skeleton (main lever)")
    ap.add_argument("--n-samples", type=int, default=180,
                    help="number of L1 skeleton samples; fewer = stiffer")
    ap.add_argument("--skel-smooth", type=int, default=5, # 0
                    help="extra moving-average window on the ordered centerline")
    args = ap.parse_args()
    denoise(args.input, args.out, sig_s=args.sig_s, sig_v=args.sig_v,
            sig_r=args.sig_r, iters=args.iters, h=args.h,
            n_samples=args.n_samples, skel_smooth=args.skel_smooth)
