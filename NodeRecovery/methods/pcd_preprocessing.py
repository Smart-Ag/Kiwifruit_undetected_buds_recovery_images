# Cleaning + Cluster Filtering + Downsampling
import open3d as o3d
import numpy as np
from pathlib import Path


# --------------------------------------------------
# Load PLY
# --------------------------------------------------
def load_point_cloud(ply_path):
    pcd = o3d.io.read_point_cloud(str(ply_path))
    return pcd


# --------------------------------------------------
# Cleaning steps
# --------------------------------------------------
def remove_statistical_outliers(pcd, nb_neighbors=30, std_ratio=2.0):
    pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    print(f"Statistical outlier removal: {len(ind)} points kept")
    return pcd


def remove_flying_pixels(pcd, radius=0.05, min_neighbors=5):
    pcd, ind = pcd.remove_radius_outlier(
        nb_points=min_neighbors,
        radius=radius
    )
    print(f"Flying pixel removal: {len(ind)} points kept")
    return pcd


def remove_small_clusters(pcd, eps=0.03, min_cluster_size=200,
                          max_centroid_distance=9999, auto_threshold=True):
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=5, print_progress=True)
    )

    if labels.size == 0:
        return pcd

    points = np.asarray(pcd.points)
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)

    if auto_threshold:
        sorted_counts = np.sort(counts)
        gaps = np.diff(sorted_counts)
        cut_idx = np.argmax(gaps)
        auto_min = sorted_counts[cut_idx] + 1

        # Safety floor: never delete clusters above min_cluster_size
        # even if they fall below the auto-detected threshold
        min_cluster_size = min(auto_min, min_cluster_size)
        print(f"Auto threshold: detected gap at {sorted_counts[cut_idx]} -> "
              f"{sorted_counts[cut_idx + 1]} points, setting min_cluster_size={min_cluster_size}")


    large_cluster_mask = np.zeros(len(points), dtype=bool)
    for lbl, cnt in zip(unique_labels, counts):
        if cnt >= min_cluster_size:
            large_cluster_mask |= (labels == lbl)

    if not np.any(large_cluster_mask):
        print("Warning: no clusters met the size threshold.")
        return pcd

    main_centroid = points[large_cluster_mask].mean(axis=0)

    valid_mask = np.zeros(len(points), dtype=bool)
    kept_clusters = 0
    removed_clusters = 0

    for lbl, cnt in zip(unique_labels, counts):
        cluster_mask = labels == lbl
        cluster_centroid = points[cluster_mask].mean(axis=0)
        dist_to_main = np.linalg.norm(cluster_centroid - main_centroid)

        if cnt >= min_cluster_size and dist_to_main <= max_centroid_distance:
            valid_mask |= cluster_mask
            kept_clusters += 1
        else:
            removed_clusters += 1
            print(f"  Removed cluster {lbl}: {cnt} pts, "
                  f"centroid dist={dist_to_main:.4f}")

    filtered_points = points[valid_mask]
    filtered_colors = np.asarray(pcd.colors)[valid_mask] if pcd.has_colors() else None

    clean_pcd = o3d.geometry.PointCloud()
    clean_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    if filtered_colors is not None:
        clean_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    print(f"Cluster filtering: kept {kept_clusters} clusters, "
          f"removed {removed_clusters} clusters "
          f"({len(filtered_points)} points remaining)")

    return clean_pcd

# --------------------------------------------------
# NEW: Remove post cleaned isolated clusters
# --------------------------------------------------
def remove_extra_clusters(pcd, eps=0.03, min_cluster_s=80):
    labels = np.array(
        pcd.cluster_dbscan(
            eps=eps,
            min_points=5,
            print_progress=True
        )
    )

    if labels.size == 0:
        return pcd

    valid_mask = np.zeros_like(labels, dtype=bool)

    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)

    kept_clusters = 0
    for lbl, cnt in zip(unique_labels, counts):
        if cnt >= min_cluster_s:
            valid_mask |= (labels == lbl)
            kept_clusters += 1

    filtered_points = np.asarray(pcd.points)[valid_mask]

    if pcd.has_colors():
        filtered_colors = np.asarray(pcd.colors)[valid_mask]
    else:
        filtered_colors = None

    clean_pcd = o3d.geometry.PointCloud()
    clean_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    if filtered_colors is not None:
        clean_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    print(f"Cluster filtering: kept {kept_clusters} clusters "
          f"({len(filtered_points)} points)")

    return clean_pcd

# --------------------------------------------------
# Voxel downsampling
# --------------------------------------------------
def voxel_downsample(pcd, voxel_size=0.002):
    down = pcd.voxel_down_sample(voxel_size)
    print(f"Voxel downsampling ({voxel_size*1000:.1f} mm): "
          f"{len(down.points)} points")
    return down


# --------------------------------------------------
# Full pipeline
# --------------------------------------------------
def process_point_cloud(
    input_ply,
    output_ply,
    sor_neighbors=30,
    sor_std=2.0,
    flying_radius=0.05,
    flying_min_neighbors=5,
    voxel_size=0.002,
    cluster_eps_factor=3.0,
    min_cluster_size=50,
    min_cluster_s=50
):
    input_ply = Path(input_ply)
    output_ply = Path(output_ply)

    print(f"\nLoading: {input_ply}")
    pcd = load_point_cloud(input_ply)
    print(f"Initial points: {len(pcd.points)}")

    # # Cleaning
    # pcd = remove_statistical_outliers(pcd, sor_neighbors, sor_std)
    # pcd = remove_flying_pixels(pcd, flying_radius, flying_min_neighbors)

    # # Cluster cleanup
    # pcd = remove_small_clusters(
    #     pcd,
    #     eps=voxel_size * cluster_eps_factor,
    #     min_cluster_size=min_cluster_size
    # )

    # Downsampling
    pcd = voxel_downsample(pcd, voxel_size)
    # pcd = remove_statistical_outliers(pcd, sor_neighbors, sor_std)
    # pcd = remove_flying_pixels(pcd, flying_radius, flying_min_neighbors)

    # # Final Cluster cleanup
    # pcd = remove_extra_clusters(
    #     pcd,
    #     eps=voxel_size * cluster_eps_factor,
    #     min_cluster_s=min_cluster_size
    # )


    # Save
    o3d.io.write_point_cloud(str(output_ply), pcd)
    print(f"\nFinal output saved to: {output_ply}")
    print(f"Final point count: {len(pcd.points)}")


# --------------------------------------------------
# Example usage
# --------------------------------------------------
if __name__ == "__main__":
    process_point_cloud(
        input_ply=r"pc_out/fv_raw.ply",
        output_ply=r"pc_out\clean_pcd.ply",
        sor_neighbors=10,
        sor_std=1.5,
        flying_radius=0.55,
        flying_min_neighbors=40,
        voxel_size=0.003,        # 12 mm 0.006
        cluster_eps_factor=3.0,  # eps = voxel_size * 3
        min_cluster_size=40, # 200 --> 20
        min_cluster_s=20 # 80 --> 40
    )
