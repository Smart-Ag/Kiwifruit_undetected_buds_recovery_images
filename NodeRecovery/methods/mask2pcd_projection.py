import os
import cv2
import numpy as np

# --- Default Constants (used if not provided to the function) ---
DEFAULT_INTRINSICS = {
    "fx": 693.553467, "fy": 693.492859,
    "cx": 643.257935, "cy": 358.344116
}


def write_ply_xyzrgb(path, xyz, rgb):
    """Saves points and colors to a PLY file."""
    assert xyz.shape[0] == rgb.shape[0]
    n = xyz.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def project_mask_to_pcd(rgb, depth, mask, output_path,
                        intrinsics=None, z_range=(0.45, 1.5), decimate=1):
    """
    Modular function to project a masked region into a 3D Point Cloud.

    Args:
        rgb: BGR image (numpy array)
        depth: Depth image in meters (numpy array)
        mask: Binary mask (numpy array, 0 and 255)
        output_path: Where to save the .ply file
        intrinsics: Dict with fx, fy, cx, cy
        z_range: Tuple of (z_min, z_max) for clipping
        decimate: Integer (1 = all points, 2 = half, etc.)
    """
    if intrinsics is None:
        intrinsics = DEFAULT_INTRINSICS

    H, W = rgb.shape[:2]
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    z_min, z_max = z_range

    # Ensure depth and mask match RGB size
    if depth.shape != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
    if mask.shape != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # Filter for valid depth and mask foreground
    depth_valid = np.isfinite(depth) & (depth > z_min) & (depth < z_max)
    valid = (mask > 0) & depth_valid

    ys, xs = np.where(valid)
    if decimate > 1 and ys.size > 0:
        sel = np.arange(ys.size)[::decimate]
        ys, xs = ys[sel], xs[sel]

    if ys.size == 0:
        print(f"Warning: No valid points found for {output_path}")
        return None, None

    # Projection Math
    Z = depth[ys, xs].astype(np.float32)
    X = (xs.astype(np.float32) - cx) / fx * Z
    Y = (ys.astype(np.float32) - cy) / fy * Z
    pts = np.stack([X, Y, Z], axis=1)

    # Color extraction (BGR to RGB)
    cols_rgb = rgb[ys, xs, ::-1]

    # Save to file
    write_ply_xyzrgb(output_path, pts, cols_rgb)

    return pts, cols_rgb


# --- Legacy Helper Functions ---
def load_depth_meters(path, is_mm=True, scale=0.001):
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None: return None
    if d.dtype == np.uint16 and is_mm:
        return d.astype(np.float32) * scale
    return d.astype(np.float32)


# --- Standalone Testing ---
if __name__ == "__main__":
    # Example of how to call the new function
    RGB_P = r"case/rgb.png"
    DEP_P = r"case/depth.png"
    MSK_P = r"case/fpv_mask.png"

    rgb_img = cv2.imread(RGB_P)
    depth_img = load_depth_meters(DEP_P)
    mask_img = cv2.imread(MSK_P, 0)

    if rgb_img is not None and depth_img is not None:
        project_mask_to_pcd(rgb_img, depth_img, mask_img, "case/in/fpv_raw.ply")
