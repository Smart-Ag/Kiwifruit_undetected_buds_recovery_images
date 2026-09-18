"""
Detect potential dormant buds as lateral bulges in the branch (X-Y top view).
=============================================================================
A bud makes the branch locally wider. Seen from the top (X-Y) the smooth branch
is a gently tapering ribbon; a bud is a short one-sided protrusion of that
ribbon's edge. So we:

  1. Parameterise every surface point by arc length s along the skeleton and a
     signed lateral offset v in the X-Y plane (v>0 / v<0 = the two edges).
  2. Per arc bin, take the upper edge U(s)=high-percentile(v) and lower edge
     L(s)=low-percentile(v) -- the ribbon outline.
  3. Detrend each edge against a wide rolling-median baseline (the true taper).
     The positive residual = how far the edge bulges past the taper.
  4. Peak-detect the bulges on each side; merge nearby ones. Each peak is a
     potential bud, placed on the bulging surface.

Gaps (bins with too few points) are skipped, not flagged.

Outputs: green bud-marker spheres PLY + an annotated X-Y / profile PNG.
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def arc_lateral(P, sk):
    """Per-point arc length s and signed X-Y lateral offset v from the skeleton."""
    seglen = np.linalg.norm(np.diff(sk, axis=0), axis=1)
    cum = np.insert(np.cumsum(seglen), 0, 0.0)
    tan = np.gradient(sk, axis=0)
    txy = tan[:, :2]; txy /= np.linalg.norm(txy, axis=1, keepdims=True) + 1e-12
    nxy = np.column_stack([-txy[:, 1], txy[:, 0]])          # X-Y normal per station
    _, idx = cKDTree(sk).query(P)
    s = cum[idx]
    v = np.einsum("ij,ij->i", P[:, :2] - sk[idx, :2], nxy[idx])
    return s, v, cum, nxy


def roll_med(a, W):
    n = len(a)
    return np.array([np.nanmedian(a[max(0, i - W):min(n, i + W + 1)]) for i in range(n)])


def find_bulges(P, sk, nb=200, env_pct=92, base_win=18, min_pts=8,
                prominence=0.0022, min_sep=0.04):
    """Pure bulge detector (no file IO). Returns (records, profile).

    records : list of dicts {arc, side, prominence, center(3,), surface(3,)}
              -- arc is polyline arc length from station 0; center is the
              skeleton-projected point; surface is the point on the bulging edge.
    profile : (sc, resU, resL, ok) for plotting.
    """
    s, v, cum, nxy = arc_lateral(P, sk)
    total = cum[-1]
    edges = np.linspace(0, total, nb + 1)
    sc = (edges[:-1] + edges[1:]) / 2
    bid = np.clip(np.digitize(s, edges) - 1, 0, nb - 1)
    U = np.full(nb, np.nan); L = np.full(nb, np.nan); cnt = np.zeros(nb)
    for i in range(nb):
        m = bid == i; cnt[i] = m.sum()
        if m.sum() >= min_pts:
            U[i] = np.percentile(v[m], env_pct)
            L[i] = np.percentile(v[m], 100 - env_pct)
    ok = cnt >= min_pts
    xs = np.arange(nb)
    Ui = np.interp(xs, xs[ok], U[ok]); Li = np.interp(xs, xs[ok], L[ok])
    baseU = roll_med(Ui, base_win); baseL = roll_med(Li, base_win)
    resU = Ui - baseU; resL = baseL - Li

    sep_bins = max(1, int(min_sep / (total / nb)))
    candidates = []
    for res, side in [(resU, +1), (resL, -1)]:
        res_m = np.where(ok, res, 0.0)
        pk, props = find_peaks(res_m, prominence=prominence, distance=sep_bins)
        for p, pr in zip(pk, props["prominences"]):
            if ok[p]:
                candidates.append((sc[p], side, pr, p))
    candidates.sort()
    merged = []
    for c in candidates:
        if merged and abs(c[0] - merged[-1][0]) < min_sep:
            if c[2] > merged[-1][2]:
                merged[-1] = c
        else:
            merged.append(c)

    def center_at(arc):
        k = int(np.clip(np.searchsorted(cum, arc) - 1, 0, len(sk) - 2))
        f = (arc - cum[k]) / (cum[k + 1] - cum[k] + 1e-12)
        return sk[k] + f * (sk[k + 1] - sk[k]), nxy[k]

    records = []
    for arc, side, pr, p in merged:
        c, nrm = center_at(arc)
        env = U[p] if side > 0 else L[p]
        surf = c.copy(); surf[:2] = c[:2] + nrm * env
        records.append({"arc": float(arc), "side": int(side),
                        "prominence": float(pr), "center": c, "surface": surf})
    return records, (sc, resU, resL, ok)


def detect(smooth="pc_out/branch2_smooth.ply", skel="pc_out/branch2_skeleton.ply",
           out=None, nb=200, env_pct=92, base_win=18, min_pts=8,
           prominence=0.0022, min_sep=0.04):
    smooth = Path(smooth); skel = Path(skel)
    if out is None:
        out = smooth.with_name(smooth.stem.replace("_smooth", "") + "_buds_detected.ply")
    out = Path(out)

    P = np.asarray(o3d.io.read_point_cloud(str(smooth)).points)
    sk = np.asarray(o3d.io.read_point_cloud(str(skel)).points)
    s, v, cum, nxy = arc_lateral(P, sk)
    total = cum[-1]
    print(f"[load] {len(P)} surface pts, arc length {total:.3f} m")

    edges = np.linspace(0, total, nb + 1)
    sc = (edges[:-1] + edges[1:]) / 2
    bid = np.clip(np.digitize(s, edges) - 1, 0, nb - 1)
    U = np.full(nb, np.nan); L = np.full(nb, np.nan); cnt = np.zeros(nb)
    for i in range(nb):
        m = bid == i; cnt[i] = m.sum()
        if m.sum() >= min_pts:
            U[i] = np.percentile(v[m], env_pct)
            L[i] = np.percentile(v[m], 100 - env_pct)
    ok = cnt >= min_pts
    # interpolate small holes only for the baseline (continuity), keep `ok` mask
    xs = np.arange(nb)
    Ui = np.interp(xs, xs[ok], U[ok]); Li = np.interp(xs, xs[ok], L[ok])
    baseU = roll_med(Ui, base_win); baseL = roll_med(Li, base_win)
    resU = Ui - baseU           # bulge outward on +side
    resL = baseL - Li           # bulge outward on -side

    sep_bins = max(1, int(min_sep / (total / nb)))
    candidates = []   # (arc, side, prominence, bin)
    for res, side in [(resU, +1), (resL, -1)]:
        res_m = np.where(ok, res, 0.0)
        pk, props = find_peaks(res_m, prominence=prominence, distance=sep_bins)
        for p, pr in zip(pk, props["prominences"]):
            if ok[p]:
                candidates.append((sc[p], side, pr, p))

    # merge peaks from the two sides that fall within min_sep (same bud, both edges)
    candidates.sort()
    merged = []
    for c in candidates:
        if merged and abs(c[0] - merged[-1][0]) < min_sep:
            if c[2] > merged[-1][2]:
                merged[-1] = c
        else:
            merged.append(c)
    print(f"[detect] {len(merged)} bulge candidates (prominence>{prominence*1000:.1f}mm)")

    # place each marker on the bulging surface: centerline(s) + side*env in X-Y normal
    def center_at(arc):
        k = int(np.clip(np.searchsorted(cum, arc) - 1, 0, len(sk) - 2))
        f = (arc - cum[k]) / (cum[k + 1] - cum[k] + 1e-12)
        return sk[k] + f * (sk[k + 1] - sk[k]), nxy[k]

    marks = []
    for arc, side, pr, p in merged:
        c, nrm = center_at(arc)
        env = (U[p] if side > 0 else L[p])
        mk = c.copy(); mk[:2] = c[:2] + nrm * env
        marks.append((arc, side, pr, mk, c))
        print(f"  bud? s={arc:.3f} side={'+ ' if side>0 else '- '} "
              f"bulge={pr*1000:.1f}mm  xyz=({mk[0]:+.3f},{mk[1]:+.3f},{mk[2]:.3f})")

    # ---- save green markers as spheres ----
    mesh = o3d.geometry.TriangleMesh()
    for _, _, _, mk, _ in marks:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
        sph.translate(mk); sph.paint_uniform_color([0.0, 0.9, 0.0])
        mesh += sph
    if len(mesh.vertices):
        o3d.io.write_triangle_mesh(str(out), mesh)
    print(f"[save] {out}  ({len(marks)} green markers)")

    # ---- plot: X-Y top with markers + lateral residual profile ----
    fig, ax = plt.subplots(2, 1, figsize=(17, 9))
    ax[0].scatter(P[:, 0], P[:, 1], s=2, c="0.7")
    ax[0].plot(sk[:, 0], sk[:, 1], "-", c="crimson", lw=1)
    for _, _, _, mk, c in marks:
        ax[0].scatter(mk[0], mk[1], s=160, c="lime", edgecolor="darkgreen",
                      zorder=5, marker="o")
    ax[0].set_aspect("equal"); ax[0].set_title("X-Y (top) — detected bulges (green)")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("y")

    ax[1].plot(sc, resU * 1000, c="steelblue", label="+side bulge")
    ax[1].plot(sc, resL * 1000, c="indianred", label="-side bulge")
    ax[1].axhline(prominence * 1000, ls=":", c="0.5", label="prominence thr")
    for arc, side, pr, mk, c in marks:
        ax[1].axvline(arc, c="green", lw=1, alpha=0.6)
    ax[1].set_xlabel("arc length s (m)"); ax[1].set_ylabel("bulge past taper (mm)")
    ax[1].set_title("lateral edge bulge vs arc length"); ax[1].legend()
    fig.suptitle(f"{smooth.name}: {len(marks)} potential dormant buds", fontsize=13)
    fig.tight_layout()
    viz = out.with_name(out.stem + ".png")
    fig.savefig(viz, dpi=110)
    print(f"[viz ] {viz}")
    return out, viz


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smooth", default="pc_out/branch1_smooth.ply")
    ap.add_argument("--skel", default="pc_out/branch1_skeleton.ply")
    ap.add_argument("--prominence", type=float, default=0.0022,
                    help="min bulge height past the taper to flag (m)")
    ap.add_argument("--min-sep", type=float, default=0.04,
                    help="min arc spacing between buds (m)")
    args = ap.parse_args()
    detect(args.smooth, args.skel, prominence=args.prominence, min_sep=args.min_sep)
