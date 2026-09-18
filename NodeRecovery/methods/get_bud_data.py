# Extract bud information from the reconstructed branch - without diameter
import numpy as np
import open3d as o3d
import os
import csv
import networkx as nx
from scipy.spatial import cKDTree
from glob import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- PATHS --------------------
SKELETON_PLY_PATH  = r"case/in/skeleton.ply"
# Diameter is the silhouette WIDTH in the X-Y (top) view: the flattened branch
# reads as a clean ribbon from the top, so a line cut perpendicular to the local
# skeleton direction spans the branch = diameter. Robust to the sparse cloud (no
# circle fit needed). Measured on the flattened (depth-smoothed) surface.
DIAM_PLY_PATH      = r"case/in/smooth.ply"
FPV_PLY_PATH       = r"case/in/fpv_raw.ply"
BUDS_DIR           = r"case/buds"     # each bud ply contains ONE point

OUT_PLY_PATH       = r"res_out/recon_analysis.ply"
OUT_CSV_PATH       = r"res_out/recon_bud_measurements.csv"
OUT_BUDS_MESH_PATH = r"res_out/recon_buds_visualization.ply"
OUT_DIAM_IMG_PATH  = r"res_out/recon_diameter_lines.png"   # X-Y view of the cut lines
# -----------------------------------------------

# ---------------- GRAPH SETTINGS ---------------
NEIGHBOR_RADIUS = 0.6   # <<< tune to skeleton point density
# -----------------------------------------------

# ---------------- VIS SETTINGS -----------------
BUD_RADIUS = 0.001      # <<< make buds bigger/smaller here
# -----------------------------------------------

# -------------- DIAMETER SETTINGS ---------------
# The branch is fattened at a bud, so diameter is measured a short distance
# APICALLY (basal->apex) past each bud. Diameter = X-Y silhouette width of a thin
# arc-slab, projected onto the line perpendicular to the local skeleton tangent.
BUD_DIAM_OFFSET = 0.015   # m, measure this far past the bud toward the apex
SLAB_HALF       = 0.005   # m, half-thickness of the perpendicular cut slab
DIAM_TRIM_PCT   = 2.0     # %, trim this tail off each side (reject stray points)
# -----------------------------------------------


def load_point_cloud_xyz(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        raise RuntimeError(f"Empty point cloud: {ply_path}")
    return np.asarray(pcd.points)


# -------------------------------------------------
# Skeleton extremes (basal / apex)
# -------------------------------------------------
def find_skeleton_extremes(skeleton_pts):
    center = np.mean(skeleton_pts, axis=0)
    centered = skeleton_pts - center

    cov = np.cov(centered.T)
    _, eigenvectors = np.linalg.eigh(cov)
    axis = eigenvectors[:, -1]

    projections = centered @ axis
    return skeleton_pts[np.argmin(projections)], skeleton_pts[np.argmax(projections)]


# -------------------------------------------------
# Graph construction
# -------------------------------------------------
def build_graph(points, radius):
    tree = cKDTree(points)
    G = nx.Graph()
    for i in range(len(points)):
        G.add_node(i, pos=points[i])

    neighbors = tree.query_ball_tree(tree, r=radius)
    for i, neigh_list in enumerate(neighbors):
        for j in neigh_list:
            if i >= j:
                continue
            dist = np.linalg.norm(points[i] - points[j])
            G.add_edge(i, j, weight=dist)

    print(f"Graph nodes: {G.number_of_nodes()}  |  Graph edges: {G.number_of_edges()}")

    # Connectivity check
    n_components = nx.number_connected_components(G)
    if n_components > 1:
        print(f"WARNING: Graph has {n_components} disconnected components. "
              f"Consider increasing NEIGHBOR_RADIUS.")
    else:
        print("Graph is fully connected.")

    return G


# -------------------------------------------------
# Projection utilities
# -------------------------------------------------
def project_point_to_segment(point, seg_start, seg_end):
    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)

    if seg_len_sq == 0:
        return seg_start, 0.0

    t = np.dot(point - seg_start, seg_vec) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)

    return seg_start + t * seg_vec, t


def find_closest_skeleton_segment(bud_point, skeleton_points):
    """
    Returns the segment index, projected point on segment, and t parameter.
    Searches all segments (defined by nearest-neighbor pairs via cKDTree ordering)
    but here we iterate over consecutive index pairs — caller should pass an
    ordered skeleton or the raw points; the graph handles correctness of distances.
    """
    min_dist = float("inf")
    best_idx = 0
    best_proj = skeleton_points[0]
    best_t = 0.0

    for i in range(len(skeleton_points) - 1):
        proj, t = project_point_to_segment(
            bud_point,
            skeleton_points[i],
            skeleton_points[i + 1]
        )
        d = np.linalg.norm(bud_point - proj)

        if d < min_dist:
            min_dist = d
            best_idx = i
            best_proj = proj
            best_t = t

    return best_idx, best_proj, best_t


# -------------------------------------------------
# Geodesic arc length via graph
# -------------------------------------------------
def geodesic_arc_length(G, skeleton_pts, basal_node, seg_idx, proj_pt, t):
    """
    Temporarily inserts proj_pt as a new node into G, connected to the two
    endpoints of its host segment, then computes Dijkstra distance from
    basal_node to proj_pt.  The temporary node is removed afterwards.

    Parameters
    ----------
    G            : nx.Graph built from skeleton_pts
    skeleton_pts : np.ndarray of skeleton points (same indexing as G nodes)
    basal_node   : int  — node index of the basal point in G
    seg_idx      : int  — index of seg_start in skeleton_pts  (seg_end = seg_idx+1)
    proj_pt      : np.ndarray shape (3,) — the projected point on the segment
    t            : float in [0,1] — interpolation parameter along the segment

    Returns
    -------
    float — geodesic distance from basal to proj_pt
    """
    seg_start_node = seg_idx
    seg_end_node   = seg_idx + 1

    seg_len = np.linalg.norm(skeleton_pts[seg_end_node] - skeleton_pts[seg_start_node])
    dist_to_start = t * seg_len
    dist_to_end   = (1.0 - t) * seg_len

    # Use a temporary node index that won't clash with existing ones
    tmp_node = len(skeleton_pts)

    G.add_node(tmp_node, pos=proj_pt)
    G.add_edge(seg_start_node, tmp_node, weight=dist_to_start)
    G.add_edge(seg_end_node,   tmp_node, weight=dist_to_end)

    try:
        arc = nx.dijkstra_path_length(G, basal_node, tmp_node, weight="weight")
    except nx.NetworkXNoPath:
        print(f"WARNING: No path found from basal node to projected bud point. "
              f"Returning nan. Check NEIGHBOR_RADIUS.")
        arc = np.nan
    finally:
        G.remove_node(tmp_node)   # always clean up

    return arc


# -------------------------------------------------
# Arc-length parameterisation of the ordered skeleton polyline
# -------------------------------------------------
def polyline_arc(skeleton_pts):
    """Cumulative arc length at each ordered station (cum[0] = 0)."""
    seg = np.linalg.norm(np.diff(skeleton_pts, axis=0), axis=1)
    return np.insert(np.cumsum(seg), 0, 0.0), seg


def point_tangent_at_arc(skeleton_pts, cum, target_arc):
    """Interpolated centerline point and unit tangent at a given arc length."""
    total = cum[-1]
    target_arc = float(np.clip(target_arc, 0.0, total))
    k = int(np.searchsorted(cum, target_arc, side="right") - 1)
    k = max(0, min(k, len(skeleton_pts) - 2))
    seg_len = cum[k + 1] - cum[k]
    f = 0.0 if seg_len <= 1e-12 else (target_arc - cum[k]) / seg_len
    pt = skeleton_pts[k] + f * (skeleton_pts[k + 1] - skeleton_pts[k])
    tan = skeleton_pts[k + 1] - skeleton_pts[k]
    tan = tan / (np.linalg.norm(tan) + 1e-12)
    return pt, tan


# -------------------------------------------------
# X-Y silhouette-width diameter (perpendicular line cut)
# -------------------------------------------------
def measure_diameter(center, tangent, surface_pts):
    """Diameter = width of the branch in the X-Y top view at (center, tangent).

    Takes a thin slab perpendicular to the local skeleton tangent (so the cut
    stays square to a curving branch), projects the slab onto the in-plane (X-Y)
    normal, and measures the trimmed extent of that projection.

    Returns (diameter_m, p_lo, p_hi, n_slab_points). p_lo/p_hi are the 3-D end
    points of the cut line (at the centerline's z). NaN diameter if too sparse.
    """
    txy = tangent[:2].astype(float)
    nrm = np.linalg.norm(txy)
    txy = np.array([1.0, 0.0]) if nrm < 1e-9 else txy / nrm
    nxy = np.array([-txy[1], txy[0]])                    # X-Y perpendicular

    rel_xy = surface_pts[:, :2] - center[:2]
    along = rel_xy @ txy
    slab = surface_pts[np.abs(along) <= SLAB_HALF]
    if len(slab) < 5:
        return np.nan, None, None, len(slab)

    lat = (slab[:, :2] - center[:2]) @ nxy               # signed perpendicular pos
    lo = np.percentile(lat, DIAM_TRIM_PCT)
    hi = np.percentile(lat, 100.0 - DIAM_TRIM_PCT)
    diam = float(hi - lo)
    p_lo = np.array([center[0] + nxy[0] * lo, center[1] + nxy[1] * lo, center[2]])
    p_hi = np.array([center[0] + nxy[0] * hi, center[1] + nxy[1] * hi, center[2]])
    return diam, p_lo, p_hi, len(slab)


def main():
    # -------------------------------------------------
    # Load clouds
    # -------------------------------------------------
    skeleton_pts = load_point_cloud_xyz(SKELETON_PLY_PATH)
    fpv_pts      = load_point_cloud_xyz(FPV_PLY_PATH)

    # Flattened branch surface for the X-Y silhouette-width diameter.
    surface_pts = load_point_cloud_xyz(DIAM_PLY_PATH)
    print(f"Surface for diameter: {len(surface_pts)} points "
          f"(from {os.path.basename(DIAM_PLY_PATH)})")

    # -------------------------------------------------
    # Basal / Apex  (coordinate-based, not index-based yet)
    # -------------------------------------------------
    ep1, ep2 = find_skeleton_extremes(skeleton_pts)
    fpv_center = np.mean(fpv_pts, axis=0)

    basal_coord, apex_coord = (
        (ep1, ep2) if np.linalg.norm(ep1 - fpv_center) < np.linalg.norm(ep2 - fpv_center)
        else (ep2, ep1)
    )

    # Find the skeleton node indices closest to basal/apex coordinates
    tree = cKDTree(skeleton_pts)
    basal_node = int(tree.query(basal_coord)[1])
    apex_node  = int(tree.query(apex_coord)[1])

    print(f"Basal node index: {basal_node}  coord: {skeleton_pts[basal_node]}")
    print(f"Apex  node index: {apex_node}   coord: {skeleton_pts[apex_node]}")

    # Arc-length along the ordered skeleton polyline + basal->apex direction.
    # Used to step BUD_DIAM_OFFSET apically past each bud for the diameter slice.
    cum, _ = polyline_arc(skeleton_pts)
    apex_dir = 1.0 if cum[apex_node] >= cum[basal_node] else -1.0
    print(f"Polyline length: {cum[-1]:.4f} m  |  basal->apex direction: "
          f"{'+arc' if apex_dir > 0 else '-arc'}")

    # -------------------------------------------------
    # Build graph
    # -------------------------------------------------
    G = build_graph(skeleton_pts, radius=NEIGHBOR_RADIUS)

    # Sanity check: total skeleton geodesic length (basal → apex)
    try:
        total_length = nx.dijkstra_path_length(G, basal_node, apex_node, weight="weight")
        print(f"Total skeleton geodesic length (basal to apex): {total_length:.6f} m")
    except nx.NetworkXNoPath:
        print("WARNING: No path between basal and apex; graph may be disconnected.")

    # -------------------------------------------------
    # Process buds
    # -------------------------------------------------
    bud_results = []
    bud_meshes  = []

    bud_files = sorted(glob(os.path.join(BUDS_DIR, "*.ply")))

    for bud_path in bud_files:
        bud_pts = load_point_cloud_xyz(bud_path)

        if bud_pts.shape[0] != 1:
            print(f"Skipping {os.path.basename(bud_path)} (expected 1 point)")
            continue

        bud_xyz = bud_pts[0]

        # Project bud onto nearest skeleton segment
        seg_idx, proj_pt, t = find_closest_skeleton_segment(bud_xyz, skeleton_pts)

        # Geodesic arc length from basal to proj_pt through the graph
        bud_arc = geodesic_arc_length(G, skeleton_pts, basal_node, seg_idx, proj_pt, t)

        # ----- diameter measured 0.015 m APICALLY (basal->apex) past the bud -----
        # Step along the ordered polyline so the cut clears the bud's bulge; the
        # local tangent keeps the cut perpendicular even where the branch curves.
        bud_poly_arc = cum[seg_idx] + t * (cum[seg_idx + 1] - cum[seg_idx])
        meas_arc = bud_poly_arc + apex_dir * BUD_DIAM_OFFSET
        meas_pt, meas_tan = point_tangent_at_arc(skeleton_pts, cum, meas_arc)
        diam, p_lo, p_hi, n_slab = measure_diameter(meas_pt, meas_tan, surface_pts)
        if np.isnan(diam):
            print(f"  {os.path.basename(bud_path)}: too sparse at "
                  f"+{BUD_DIAM_OFFSET*1000:.0f}mm ({n_slab} slab pts)")

        # Visible sphere placed at the projected point on the skeleton
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=BUD_RADIUS)
        sphere.translate(proj_pt)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # red
        bud_meshes.append(sphere)

        bud_results.append({
            "file":        os.path.basename(bud_path),
            "anchor":      proj_pt,
            "arc_dist":    bud_arc,
            "diam_anchor": meas_pt,
            "diameter":    diam,
            "n_slab":      n_slab,
            "line":        None if np.isnan(diam) else (p_lo, p_hi),
        })

    # -------------------------------------------------
    # Sort buds by arc distance & compute internode lengths
    # -------------------------------------------------
    bud_results.sort(key=lambda x: (np.inf if np.isnan(x["arc_dist"]) else x["arc_dist"]))

    # -------------------------------------------------
    # Insert synthetic BASAL (bud0) and APEX (last bud) entries
    # -------------------------------------------------
    try:
        total_length = nx.dijkstra_path_length(G, basal_node, apex_node, weight="weight")
    except nx.NetworkXNoPath:
        total_length = np.nan

    basal_entry = {
        "file":        "basal",
        "anchor":      skeleton_pts[basal_node],
        "arc_dist":    0.0,
        "diam_anchor": skeleton_pts[basal_node],
        "diameter":    np.nan,
        "n_slab":      0,
        "line":        None,
    }
    apex_entry = {
        "file":        "apex",
        "anchor":      skeleton_pts[apex_node],
        "arc_dist":    total_length,
        "diam_anchor": skeleton_pts[apex_node],
        "diameter":    np.nan,
        "n_slab":      0,
        "line":        None,
    }

    all_entries = [basal_entry] + bud_results + [apex_entry]

    # Compute internode lengths across the full list
    prev_arc = None
    for entry in all_entries:
        if prev_arc is None or np.isnan(entry["arc_dist"]) or np.isnan(prev_arc):
            entry["internode_length"] = np.nan
        else:
            entry["internode_length"] = entry["arc_dist"] - prev_arc
        prev_arc = entry["arc_dist"]

    # Relative position along the branch (0 = base, 1 = tip)
    for entry in all_entries:
        arc = entry["arc_dist"]
        entry["rel_position"] = (arc / total_length) if (
            not np.isnan(arc) and total_length and not np.isnan(total_length)) else np.nan

    # -------------------------------------------------
    # Export CSV
    # -------------------------------------------------
    os.makedirs(os.path.dirname(OUT_CSV_PATH), exist_ok=True)

    with open(OUT_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bud_id",
            "bud_file",
            "arc_length_from_basal_m",
            "internode_length_m",
            "rel_position",
            "x", "y", "z",
            "diameter_mm",
            "diam_n_points",
            "diam_meas_x", "diam_meas_y", "diam_meas_z",
        ])

        for i, entry in enumerate(all_entries):
            diam = entry.get("diameter", np.nan)
            meas = entry.get("diam_anchor", entry["anchor"])
            writer.writerow([
                i,
                entry["file"],
                "" if np.isnan(entry["arc_dist"])         else f"{entry['arc_dist']:.6f}",
                "" if np.isnan(entry["internode_length"]) else f"{entry['internode_length']:.6f}",
                "" if np.isnan(entry.get("rel_position", np.nan)) else f"{entry['rel_position']:.4f}",
                f"{entry['anchor'][0]:.6f}",
                f"{entry['anchor'][1]:.6f}",
                f"{entry['anchor'][2]:.6f}",
                "" if np.isnan(diam) else f"{diam*1000:.2f}",
                entry.get("n_slab", 0),
                f"{meas[0]:.6f}", f"{meas[1]:.6f}", f"{meas[2]:.6f}",
            ])

    print(f"Saved CSV: {OUT_CSV_PATH}")
    print(f"Total entries in CSV: {len(all_entries)}  "
          f"(1 basal + {len(bud_results)} detected buds + 1 apex)")

    # Add basal (blue) and apex (green) spheres to visualization
    for coord, color in [(skeleton_pts[basal_node], [0.0, 0.0, 1.0]),
                         (skeleton_pts[apex_node],  [0.0, 1.0, 0.0])]:
        sp = o3d.geometry.TriangleMesh.create_sphere(radius=BUD_RADIUS)
        sp.translate(coord)
        sp.paint_uniform_color(color)
        bud_meshes.append(sp)

    # -------------------------------------------------
    # Skeleton visualization  (basal=blue, apex=green, rest=grey)
    # -------------------------------------------------
    colors = np.tile([0.6, 0.6, 0.6], (len(skeleton_pts), 1))
    colors[basal_node] = [0.0, 0.0, 1.0]   # blue
    colors[apex_node]  = [0.0, 1.0, 0.0]   # green

    skel_pcd = o3d.geometry.PointCloud()
    skel_pcd.points = o3d.utility.Vector3dVector(skeleton_pts)
    skel_pcd.colors = o3d.utility.Vector3dVector(colors)

    # -------------------------------------------------
    # Save bud spheres as a single mesh
    # -------------------------------------------------
    if len(bud_meshes) > 0:
        buds_mesh = bud_meshes[0]
        for m in bud_meshes[1:]:
            buds_mesh += m

        os.makedirs(os.path.dirname(OUT_BUDS_MESH_PATH), exist_ok=True)
        o3d.io.write_triangle_mesh(OUT_BUDS_MESH_PATH, buds_mesh)
        print(f"Saved bud visualization mesh: {OUT_BUDS_MESH_PATH}")

    os.makedirs(os.path.dirname(OUT_PLY_PATH), exist_ok=True)
    o3d.io.write_point_cloud(OUT_PLY_PATH, skel_pcd)
    print(f"Saved skeleton visualization: {OUT_PLY_PATH}")

    # -------------------------------------------------
    # X-Y (top) image showing each diameter cut line
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.scatter(surface_pts[:, 0], surface_pts[:, 1], s=2, c="0.75", zorder=1)
    ax.plot(skeleton_pts[:, 0], skeleton_pts[:, 1], "-", c="crimson", lw=1,
            zorder=2, label="skeleton")
    for entry in all_entries:
        line = entry.get("line")
        if line is None:
            continue
        p_lo, p_hi = line
        ax.plot([p_lo[0], p_hi[0]], [p_lo[1], p_hi[1]], "-", c="blue", lw=2,
                zorder=4)
        ax.scatter(entry["anchor"][0], entry["anchor"][1], s=40, c="red",
                   zorder=5)                                   # bud position
        mid = (p_lo + p_hi) / 2
        ax.annotate(f"{entry['diameter']*1000:.1f}", (mid[0], mid[1]),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=8, color="navy")
    ax.scatter([], [], c="red", label="bud (projected)")
    ax.plot([], [], c="blue", lw=2, label=f"diameter cut (+{BUD_DIAM_OFFSET*1000:.0f}mm, mm)")
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Diameter = X-Y silhouette width, {BUD_DIAM_OFFSET*1000:.0f} mm apical of each bud")
    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_DIAM_IMG_PATH), exist_ok=True)
    fig.savefig(OUT_DIAM_IMG_PATH, dpi=120)
    print(f"Saved diameter-line image: {OUT_DIAM_IMG_PATH}")


if __name__ == "__main__":
    main()
