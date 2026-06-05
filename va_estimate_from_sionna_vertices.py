"""
Estimate virtual anchors from Sionna path interaction vertices.

This is the simulation-side method: use Sionna RT path metadata to collect
first-order specular reflection points, fit wall planes with RANSAC, and mirror
the BS across each plane to obtain VA coordinates.

The script can either:

1. Read a previously saved NPZ containing ``vertices`` and ``interactions``; or
2. Re-run a small Sionna RT job for a subset of UE positions and extract
   ``paths.vertices`` directly.

The current ``ue_mimo_multipath_data.npz`` does not contain vertices, so the
default path is to re-run a lightweight first-order specular trace.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np


os.environ["HOME"] = os.getcwd()
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


CARRIER_FREQ = 28e9
BS_POS = np.array([40.0, -40.0, 15.0])


@dataclass
class PlaneEstimate:
    normal: np.ndarray
    d: float
    inlier_count: int
    mean_point: np.ndarray
    va_position: np.ndarray
    rmse_m: float


def write_default_xml(xml_path: str):
    xml = """<scene version="2.1.0">
    <default name="integrator" value="path"/>
    <integrator type="path"/>

    <bsdf type="radio-material" id="custom_concrete">
        <float name="relative_permittivity" value="5.0"/>
        <float name="conductivity" value="0.05"/>
        <float name="scattering_coefficient" value="0.7"/>
        <float name="thickness" value="0.1"/>
        <float name="xpd_coefficient" value="0.0"/>
    </bsdf>

    <bsdf type="radio-material" id="ground_mat">
        <float name="relative_permittivity" value="4.0"/>
        <float name="conductivity" value="0.1"/>
        <float name="scattering_coefficient" value="0.5"/>
        <float name="thickness" value="0.1"/>
        <float name="xpd_coefficient" value="0.0"/>
    </bsdf>

    <shape type="ply" id="b">
        <string name="filename" value="metis_scene.ply"/>
        <boolean name="face_normals" value="true"/>
        <ref id="custom_concrete" name="bsdf"/>
    </shape>
    <shape type="ply" id="g">
        <string name="filename" value="metis_ground.ply"/>
        <boolean name="face_normals" value="true"/>
        <ref id="ground_mat" name="bsdf"/>
    </shape>
    </scene>"""
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml)


def ensure_scene_files(xml_path: str):
    if not os.path.exists("metis_scene.ply") or not os.path.exists("metis_ground.ply"):
        import trimesh

        buildings = []
        for cx, cy, h in [(-40, 40, 25.0), (40, 40, 30.0), (-40, -40, 35.0)]:
            box = trimesh.creation.box(
                extents=[60.0, 60.0, h],
                transform=trimesh.transformations.translation_matrix([cx, cy, h / 2.0]),
            )
            buildings.append(box)
        trimesh.util.concatenate(buildings).export("metis_scene.ply")
        trimesh.creation.box(extents=[300.0, 300.0, 1.0]).export("metis_ground.ply")

    if not os.path.exists(xml_path):
        write_default_xml(xml_path)


def configure_sionna(use_gpu: bool):
    import mitsuba as mi

    variant = "cuda_ad_mono_polarized" if use_gpu else "llvm_ad_mono_polarized"
    mi.set_variant(variant)

    from sionna.rt import load_scene, PathSolver, PlanarArray, Receiver, Transmitter
    from sionna.rt.constants import InteractionType

    return load_scene, PathSolver, PlanarArray, Receiver, Transmitter, InteractionType


def tensor_to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def flatten_path_tensor(arr: np.ndarray, tail_dim: int | None = None):
    arr = np.asarray(arr)
    if tail_dim is None:
        return arr.reshape(-1)
    return arr.reshape(-1, tail_dim)


def extract_first_order_specular(paths, interaction_type):
    vertices = tensor_to_numpy(paths.vertices)
    interactions = tensor_to_numpy(paths.interactions)
    a_real, a_imag = paths.a
    gains = np.hypot(tensor_to_numpy(a_real), tensor_to_numpy(a_imag))

    # Depth is axis 0. We only want the first interaction point.
    first_vertices = flatten_path_tensor(vertices[0], tail_dim=3)
    first_interactions = flatten_path_tensor(interactions[0])
    gains_flat = flatten_path_tensor(gains)

    n = min(len(first_vertices), len(first_interactions), len(gains_flat))
    first_vertices = first_vertices[:n]
    first_interactions = first_interactions[:n]
    gains_flat = gains_flat[:n]

    mask = (
        (first_interactions == interaction_type.SPECULAR)
        & np.all(np.isfinite(first_vertices), axis=1)
        & (np.linalg.norm(first_vertices, axis=1) > 1e-6)
        & (gains_flat > 1e-18)
    )
    return first_vertices[mask], gains_flat[mask]


def load_vertices_from_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    if "vertices" not in data.files or "interactions" not in data.files:
        raise KeyError(
            f"{npz_path} does not contain 'vertices' and 'interactions'. "
            "Use the recompute mode or regenerate multipath data with path metadata."
        )

    vertices = np.asarray(data["vertices"])
    interactions = np.asarray(data["interactions"])
    gains = np.asarray(data["path_gains"]) if "path_gains" in data.files else np.ones(vertices.shape[-2])

    points = flatten_path_tensor(vertices[0], tail_dim=3)
    inter = flatten_path_tensor(interactions[0])
    gains = flatten_path_tensor(gains)
    n = min(len(points), len(inter), len(gains))
    points = points[:n]
    inter = inter[:n]
    gains = gains[:n]

    specular_value = 1
    mask = (
        (inter == specular_value)
        & np.all(np.isfinite(points), axis=1)
        & (np.linalg.norm(points, axis=1) > 1e-6)
        & (gains > 1e-18)
    )
    return points[mask], gains[mask]


def load_ue_positions(npz_path: str, max_frames: int):
    if os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        ue_positions = np.asarray(data["ue_positions"], dtype=float)
    else:
        ys = np.linspace(-20.0, 40.0, 122)
        ue_positions = np.column_stack((np.zeros_like(ys), ys, np.full_like(ys, 1.5)))

    if max_frames >= len(ue_positions):
        return ue_positions
    idx = np.linspace(0, len(ue_positions) - 1, max_frames).round().astype(int)
    return ue_positions[np.unique(idx)]


def recompute_vertices(args):
    ensure_scene_files(args.xml)
    (
        load_scene,
        PathSolver,
        PlanarArray,
        Receiver,
        Transmitter,
        InteractionType,
    ) = configure_sionna(args.gpu)

    scene = load_scene(args.xml, merge_shapes=False)
    scene.frequency = CARRIER_FREQ
    scene.tx_array = PlanarArray(
        num_rows=8,
        num_cols=8,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="V",
    )

    tx = Transmitter(
        name="BS0",
        position=BS_POS.tolist(),
        orientation=[np.pi * 0.75, 0.0, 0.0],
    )
    rx = Receiver(name="UE", position=[0.0, -20.0, 1.5], orientation=[0.0, 0.0, 0.0])
    scene.add(tx)
    scene.add(rx)

    solver = PathSolver()
    ue_positions = load_ue_positions(args.ue_npz, args.max_frames)

    all_points = []
    all_gains = []
    for frame_idx, ue_pos in enumerate(ue_positions):
        rx.position = ue_pos.tolist()
        paths = solver(
            scene,
            max_depth=1,
            samples_per_src=args.samples,
            max_num_paths_per_src=args.max_paths,
            synthetic_array=True,
            los=False,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=False,
            diffraction=False,
            seed=args.seed + frame_idx,
        )
        points, gains = extract_first_order_specular(paths, InteractionType)
        print(
            f"Frame {frame_idx + 1:02d}/{len(ue_positions)} "
            f"UE=[{ue_pos[0]:.1f}, {ue_pos[1]:.1f}, {ue_pos[2]:.1f}] "
            f"specular_vertices={len(points)}"
        )
        if len(points) > 0:
            all_points.append(points)
            all_gains.append(gains)

    if not all_points:
        raise RuntimeError("No first-order specular vertices found. Increase --samples.")

    return np.vstack(all_points), np.concatenate(all_gains)


def plane_from_points(p1, p2, p3):
    n = np.cross(p2 - p1, p3 - p1)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return None
    n = n / norm
    d = -float(np.dot(n, p1))
    return n, d


def fit_plane_least_squares(points):
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    n = vh[-1]
    n = n / np.linalg.norm(n)
    d = -float(np.dot(n, centroid))
    return n, d


def mirror_point_to_plane(point, normal, d):
    signed_distance = float(np.dot(normal, point) + d)
    return point - 2.0 * signed_distance * normal


def plane_rmse(points, normal, d):
    distances = points @ normal + d
    return float(np.sqrt(np.mean(distances**2)))


def ransac_planes(
    points: np.ndarray,
    max_planes: int,
    threshold_m: float,
    min_inliers: int,
    iterations: int,
    rng: np.random.Generator,
):
    remaining = points.copy()
    planes = []

    while len(remaining) >= min_inliers and len(planes) < max_planes:
        best_inliers = None
        best_model = None
        for _ in range(iterations):
            sample_idx = rng.choice(len(remaining), size=3, replace=False)
            model = plane_from_points(*remaining[sample_idx])
            if model is None:
                continue
            n, d = model
            distances = np.abs(remaining @ n + d)
            inliers = np.where(distances < threshold_m)[0]
            if best_inliers is None or len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_model = (n, d)

        if best_inliers is None or len(best_inliers) < min_inliers:
            break

        inlier_points = remaining[best_inliers]
        n, d = fit_plane_least_squares(inlier_points)

        # Keep vertical wall-like planes only.
        if abs(n[2]) < 0.35:
            mean_point = np.mean(inlier_points, axis=0)
            if np.dot(BS_POS - mean_point, n) < 0:
                n = -n
                d = -d
            va = mirror_point_to_plane(BS_POS, n, d)
            planes.append(
                PlaneEstimate(
                    normal=n,
                    d=d,
                    inlier_count=len(best_inliers),
                    mean_point=mean_point,
                    va_position=va,
                    rmse_m=plane_rmse(inlier_points, n, d),
                )
            )

        mask = np.ones(len(remaining), dtype=bool)
        mask[best_inliers] = False
        remaining = remaining[mask]

    return planes


def filter_and_sort_planes(planes, max_va_distance_m: float):
    filtered = []
    for plane in planes:
        if np.linalg.norm(plane.va_position - BS_POS) > max_va_distance_m:
            continue
        filtered.append(plane)
    filtered.sort(key=lambda p: p.inlier_count, reverse=True)
    return filtered


def axis_aligned_planes_from_vertices(points, threshold_m, min_inliers):
    """Fallback for single-line UE tracks.

    With fixed UE x/z, first-order reflection vertices on a vertical wall can
    collapse to a line, making generic 3D plane RANSAC underdetermined. In this
    box scene, the building walls are axis-aligned, so clusters of nearly
    constant x or y are valid wall-plane evidence.
    """
    planes = []
    for axis in (0, 1):
        coord = points[:, axis]
        bins = np.round(coord / threshold_m).astype(int)
        for bin_id in np.unique(bins):
            inliers = np.where(bins == bin_id)[0]
            if len(inliers) < min_inliers:
                continue

            inlier_points = points[inliers]
            # Avoid promoting residual ground-reflection lines to walls.
            if np.median(inlier_points[:, 2]) < 1.0:
                continue

            value = float(np.median(inlier_points[:, axis]))
            distances = np.abs(coord - value)
            inliers = np.where(distances < threshold_m)[0]
            if len(inliers) < min_inliers:
                continue

            normal = np.zeros(3, dtype=float)
            normal[axis] = 1.0
            d = -value
            mean_point = np.mean(points[inliers], axis=0)
            if np.dot(BS_POS - mean_point, normal) < 0:
                normal = -normal
                d = -d

            va = mirror_point_to_plane(BS_POS, normal, d)
            planes.append(
                PlaneEstimate(
                    normal=normal,
                    d=d,
                    inlier_count=len(inliers),
                    mean_point=mean_point,
                    va_position=va,
                    rmse_m=plane_rmse(points[inliers], normal, d),
                )
            )
    return planes


def merge_duplicate_planes(planes, va_sep_m=2.0):
    merged = []
    for plane in sorted(planes, key=lambda p: p.inlier_count, reverse=True):
        if any(np.linalg.norm(plane.va_position - other.va_position) < va_sep_m for other in merged):
            continue
        merged.append(plane)
    return merged


def estimate(args):
    if args.vertices_npz:
        points, gains = load_vertices_from_npz(args.vertices_npz)
    else:
        points, gains = recompute_vertices(args)

    if len(points) == 0:
        raise RuntimeError("No usable reflection vertices found.")

    # Remove ground/roof outliers and keep strong-ish vertices for stable RANSAC.
    mask = (points[:, 2] > 0.5) & (points[:, 2] < args.max_z)
    points = points[mask]
    gains = gains[mask]
    if len(points) > args.max_points:
        order = np.argsort(gains)[-args.max_points :]
        points = points[order]

    rng = np.random.default_rng(args.seed)
    planes = ransac_planes(
        points,
        max_planes=args.max_planes,
        threshold_m=args.ransac_threshold,
        min_inliers=args.min_inliers,
        iterations=args.ransac_iterations,
        rng=rng,
    )
    planes.extend(
        axis_aligned_planes_from_vertices(
            points,
            threshold_m=args.axis_threshold,
            min_inliers=args.min_inliers,
        )
    )
    planes = merge_duplicate_planes(planes)
    return filter_and_sort_planes(planes, args.max_va_distance), len(points)


def print_planes(planes, num_points):
    print("\nEstimated VA coordinates from Sionna path vertices")
    print("=" * 72)
    print(f"Reflection vertices used for RANSAC: {num_points}")
    if not planes:
        print("No wall-like plane survived filtering.")
        return

    for idx, plane in enumerate(planes, start=1):
        va = plane.va_position
        n = plane.normal
        print(f"[{idx}] VA = [{va[0]:9.3f}, {va[1]:9.3f}, {va[2]:7.3f}]")
        print(
            f"    normal=[{n[0]: .4f}, {n[1]: .4f}, {n[2]: .4f}], "
            f"d={plane.d:.3f}, inliers={plane.inlier_count}, "
            f"plane_rmse={plane.rmse_m:.3f} m"
        )
    print("=" * 72)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate VA coordinates from Sionna first-order specular vertices."
    )
    parser.add_argument("--vertices-npz", default=None)
    parser.add_argument("--ue-npz", default="ue_mimo_multipath_data.npz")
    parser.add_argument("--xml", default="temp_sim.xml")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--max-paths", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-z", type=float, default=40.0)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--max-planes", type=int, default=6)
    parser.add_argument("--min-inliers", type=int, default=4)
    parser.add_argument("--ransac-threshold", type=float, default=0.35)
    parser.add_argument("--axis-threshold", type=float, default=0.45)
    parser.add_argument("--ransac-iterations", type=int, default=1500)
    parser.add_argument("--max-va-distance", type=float, default=180.0)
    return parser.parse_args()


if __name__ == "__main__":
    planes_, num_points_ = estimate(parse_args())
    print_planes(planes_, num_points_)
