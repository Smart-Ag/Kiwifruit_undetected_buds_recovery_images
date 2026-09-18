## Point to cloud with search radius

import os
import cv2
import numpy as np

# -------------------- USER INPUTS --------------------
RGB_PATH   = r"case/rgb.png"
DEPTH_PATH = r"case/depth.png"

POINTS_2D = []

INTRINSICS = {
    "fx": 693.553467,
    "fy": 693.492859,
    "cx": 643.257935,
    "cy": 358.344116
}

UINT16_DEPTH_IS_MM = True
UINT16_SCALE_M     = 0.001

Z_MIN = 0.55
Z_MAX = 1.25

MAX_SEARCH_RADIUS_PX = 3          # spatial window
DEPTH_RADIUS_M      = 0.002       # ±3 mm depth constraint
MAX_ANGLE_DEG       = 20.0        # angular constraint

OUT_DIR = "pc_output"
# ----------------------------------------------------


def imread_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read {path}")
    return img


def load_depth_meters(path):
    d = imread_any(path)
    if d.dtype == np.uint16 and UINT16_DEPTH_IS_MM:
        d = d.astype(np.float32) * UINT16_SCALE_M
    else:
        d = d.astype(np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    return d


def backproject(u, v, z, fx, fy, cx, cy):
    X = (u - cx) / fx * z
    Y = (v - cy) / fy * z
    return np.array([X, Y, z], dtype=np.float32)


def find_stable_depth(u, v, depth):
    """Find a depth that is spatially close AND depth-consistent."""
    h, w = depth.shape
    samples = []

    for dv in range(-MAX_SEARCH_RADIUS_PX, MAX_SEARCH_RADIUS_PX + 1):
        for du in range(-MAX_SEARCH_RADIUS_PX, MAX_SEARCH_RADIUS_PX + 1):
            uu = u + du
            vv = v + dv
            if uu < 0 or uu >= w or vv < 0 or vv >= h:
                continue

            z = depth[vv, uu]
            if np.isfinite(z) and Z_MIN < z < Z_MAX:
                samples.append((uu, vv, z))

    if not samples:
        return None

    # robust depth selection
    zs = np.array([s[2] for s in samples])
    z_med = np.median(zs)

    # keep only depths close to median (±3mm)
    filtered = [
        (uu, vv, z) for (uu, vv, z) in samples
        if abs(z - z_med) <= DEPTH_RADIUS_M
    ]

    if not filtered:
        return None

    # choose the spatially closest valid pixel
    filtered.sort(key=lambda p: (p[0] - u) ** 2 + (p[1] - v) ** 2)
    return filtered[0]


def angle_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def write_ply_xyzrgb(path, pts, color_rgb):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z) in pts:
            r, g, b = color_rgb
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    depth = load_depth_meters(DEPTH_PATH)

    fx, fy, cx, cy = (
        INTRINSICS["fx"],
        INTRINSICS["fy"],
        INTRINSICS["cx"],
        INTRINSICS["cy"],
    )

    points_3d = []

    # ---- project points robustly ----
    for i, (u, v) in enumerate(POINTS_2D):
        res = find_stable_depth(u, v, depth)

        if res is None:
            print(f"[WARN] No stable depth near ({u},{v})")
            continue

        uu, vv, z = res
        pt = backproject(uu, vv, z, fx, fy, cx, cy)
        points_3d.append(pt)

        out_path = os.path.join(OUT_DIR, f"point_{i:03d}.ply")
        write_ply_xyzrgb(out_path, [pt], (255, 0, 0))
        print(f"Point {i}: ({u},{v}) -> ({uu},{vv}) z={z:.4f} m")

    # ---- pairwise validation (diameter sanity) ----
    if len(points_3d) == 2:
        p0, p1 = points_3d
        vec = p1 - p0

        # depth consistency
        dz = abs(p0[2] - p1[2])

        # angle relative to horizontal plane
        horizontal = np.array([vec[0], vec[1], 0.0])
        ang = angle_between(vec, horizontal)

        if dz > DEPTH_RADIUS_M:
            print(f"[REJECTED] Depth mismatch: dz={dz:.4f} m")
            return

        if ang > MAX_ANGLE_DEG:
            print(f"[REJECTED] Angle too steep: {ang:.1f} degrees")
            return

        print(f"[OK] Points valid | dz={dz:.4f} m | angle={ang:.1f} degrees")

    # ---- save combined ----
    if points_3d:
        combined_path = os.path.join(OUT_DIR, "all_points.ply")
        write_ply_xyzrgb(combined_path, points_3d, (255, 0, 0))
        print(f"Saved combined cloud: {combined_path}")


if __name__ == "__main__":
    main()
