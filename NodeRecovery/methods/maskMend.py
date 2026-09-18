# maskmend for branches --> works on skeleton level

import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import splprep, splev
from collections import deque
import os

# ---------------- CONFIG ----------------
class config:
    EDGE_MARGIN = 10
    SHOOT_START_DISTANCE = 20
    CONE_SEARCH_DISTANCE = 80
    CONE_SEARCH_ANGLE = 30  # degrees
    DILATION_SQUARE_SIZE = 5


# ---------------- HELPERS ----------------
def find_mask_endpoints(skeleton):
    kernel = np.array([[1, 1, 1],
                       [1,10, 1],
                       [1, 1, 1]])
    convolved = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
    return np.argwhere(convolved == 11)


def find_extreme_points(coords):
    if len(coords) < 2:
        return None, None
    dist_matrix = squareform(pdist(coords))
    idx1, idx2 = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    return coords[idx1], coords[idx2]


def find_point_at_distance(skeleton, start_point, distance):
    queue = deque([(start_point, 0)])
    visited = set()

    while queue:
        current, dist = queue.popleft()
        if tuple(current) in visited:
            continue
        visited.add(tuple(current))

        if dist >= distance:
            return current

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = current[1] + dx, current[0] + dy
                if (0 <= nx < skeleton.shape[1] and
                    0 <= ny < skeleton.shape[0] and
                    skeleton[ny, nx] > 0):
                    queue.append(([ny, nx], dist + np.hypot(dx, dy)))
    return None


def draw_spline(img, p0, p1, p2, thickness=4, num_points=10):
    x = [p0[0], p1[0], p2[0]]
    y = [p0[1], p1[1], p2[1]]

    try:
        tck, _ = splprep([x, y], s=0)
        u = np.linspace(0, 1, num_points)
        x_f, y_f = splev(u, tck)

        for i in range(len(x_f) - 1):
            cv2.line(
                img,
                (int(x_f[i]), int(y_f[i])),
                (int(x_f[i + 1]), int(y_f[i + 1])),
                255, thickness)
    except Exception:
        cv2.line(img, p0, p2, 255, thickness)


# ---------------- CORE PIPELINE ----------------
def process_components(binary_mask_shoot):
    image_h, image_w = binary_mask_shoot.shape
    num_labels, labels = cv2.connectedComponents(binary_mask_shoot)

    skeletons = []
    endpoints_list = []

    for label in range(1, num_labels):
        mask = (labels == label).astype(np.uint8)
        skeleton = skeletonize(mask > 0).astype(np.uint8)
        endpoints = find_mask_endpoints(skeleton)

        skeletons.append(skeleton)
        endpoints_list.append(endpoints)

    all_extreme_points = []
    point_to_skeleton = {}

    for idx, endpoints in enumerate(endpoints_list):
        if len(endpoints) < 2:
            continue

        p1, p2 = find_extreme_points(endpoints)
        for ep in [p1, p2]:
            y, x = ep
            if (x < config.EDGE_MARGIN or x > image_w - config.EDGE_MARGIN or
                y < config.EDGE_MARGIN or y > image_h - config.EDGE_MARGIN):
                continue

            all_extreme_points.append(ep)
            point_to_skeleton[tuple(ep)] = idx

    return all_extreme_points, point_to_skeleton, skeletons


def mend_binary_mask(binary_mask_shoot, output_path="mended_debug.png"):
    all_extreme_points, point_to_skeleton, skeletons = process_components(binary_mask_shoot)

    result_img = binary_mask_shoot.copy()
    image_h, image_w = binary_mask_shoot.shape

    for idx, ep in enumerate(all_extreme_points):
        y, x = ep
        skeleton = skeletons[point_to_skeleton[tuple(ep)]]

        base = find_point_at_distance(skeleton, ep, distance=40)
        if base is None:
            continue

        ep_xy = np.array(ep[::-1])      # (x, y)
        base_xy = np.array(base[::-1])

        direction = ep_xy - base_xy
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue

        unit_vec = direction / norm

        for j, candidate in enumerate(all_extreme_points):
            if j == idx:
                continue

            cand_xy = np.array(candidate[::-1])
            vec_to_cand = cand_xy - base_xy
            dist = np.linalg.norm(vec_to_cand)

            if not (5 < dist <= config.CONE_SEARCH_DISTANCE):
                continue

            angle = np.degrees(np.arccos(
                np.clip(np.dot(unit_vec, vec_to_cand / dist), -1.0, 1.0)
            ))

            if angle <= config.CONE_SEARCH_ANGLE:
                control_xy = ep_xy + unit_vec * 20
                draw_spline(
                    result_img,
                    tuple(ep_xy.astype(int)),
                    tuple(control_xy.astype(int)),
                    tuple(cand_xy.astype(int))
                )
                break

    # cv2.imwrite(output_path, result_img)
    print(f"[INFO] Saved mended visualization to: {output_path}")
    return result_img


# ---------------- MAIN ----------------
if __name__ == "__main__":
    INPUT_MASK = "case/fc_mask.png"
    OUTPUT_IMG = "case/fc_mask_mended.png"

    binary = cv2.imread(INPUT_MASK, cv2.IMREAD_GRAYSCALE)
    if binary is None:
        raise FileNotFoundError(INPUT_MASK)

    binary = (binary > 0).astype(np.uint8) * 255
    img_skeleton = skeletonize((binary))
    mend_binary_mask(binary, OUTPUT_IMG)
