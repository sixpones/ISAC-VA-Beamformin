"""
快速 CSI 路径轨迹法估计虚拟锚点 VA。

与 va_estimate_from_tau_csi.py 的区别：
1. 不做 x-y VA 网格搜索；
2. 先从每帧海量 Sionna 路径中选取较少的代表路径；
3. 用 BS 侧 64 维 h_complex 的归一化 CSI 相关性做跨帧路径跟踪；
4. 对每条 CSI 连续轨迹，用 tau 对应距离闭式最小二乘拟合 VA；
5. 再用 CSI 空间一致性与象限先验选择 +x/-x 镜像侧。

该方法仍使用 tau 做轨迹后的距离拟合，但避免了密集 VA 网格搜索，
适合作为更快的 CSI-track-first baseline。
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np

from va_estimate_from_tau_csi import (
    BS_POS,
    C,
    adjacent_csi_corr,
    evaluate_candidate_csi_spatial_consistency,
    quadrant_prior_score,
)


@dataclass
class FrameCandidates:
    ranges: np.ndarray
    gains: np.ndarray
    h_units: np.ndarray
    h_vectors: np.ndarray
    path_indices: np.ndarray


@dataclass
class CsiTrack:
    frames: list[int]
    ranges: list[float]
    gains: list[float]
    h_units: list[np.ndarray]
    h_vectors: list[np.ndarray]
    path_indices: list[int]
    last_frame: int


@dataclass
class FastVAEstimate:
    selected_va: np.ndarray
    delay_plus_va: np.ndarray
    delay_minus_va: np.ndarray
    selected_sign: int
    count: int
    range_rmse_m: float
    mean_gain_db: float
    adjacent_csi_corr: float
    mean_array_corr: float
    mean_beam_gain: float
    mean_rank1_ratio: float
    mean_phase_slope_error: float
    csi_spatial_score: float
    score: float
    track_id: int


def norm_rows(h: np.ndarray) -> np.ndarray:
    if h.size == 0:
        return np.array([], dtype=float)
    return np.linalg.norm(h, axis=1)


def unit_rows(h: np.ndarray, norms: np.ndarray) -> np.ndarray:
    if h.size == 0:
        return h
    return h / (norms[:, None] + 1e-30)


def select_representative_paths(
    ranges: np.ndarray,
    gains: np.ndarray,
    args,
) -> np.ndarray:
    """
    从每帧几千到几万条路径中选少量代表路径。

    选择策略：
    - 保留全局增益最强的 top_global_paths 条；
    - 按距离分箱，每个距离 bin 保留 top_per_range_bin 条强路径；
    - 最后如果仍超过 max_paths_per_frame，再按增益裁剪。

    这样比单纯取 top-K 更不容易漏掉较弱但距离轮廓清晰的反射路径。
    """
    n = len(ranges)
    if n == 0:
        return np.array([], dtype=int)

    selected: set[int] = set()
    if args.top_global_paths > 0:
        k = min(args.top_global_paths, n)
        top = np.argpartition(gains, -k)[-k:]
        selected.update(int(i) for i in top)

    if args.range_bin_m > 0.0 and args.top_per_range_bin > 0:
        bins = np.floor(ranges / args.range_bin_m).astype(int)
        order = np.argsort(gains)[::-1]
        bin_counts: dict[int, int] = {}
        for idx in order:
            b = int(bins[idx])
            used = bin_counts.get(b, 0)
            if used >= args.top_per_range_bin:
                continue
            selected.add(int(idx))
            bin_counts[b] = used + 1

    idx = np.asarray(sorted(selected), dtype=int)
    if len(idx) == 0:
        return idx

    if args.max_paths_per_frame > 0 and len(idx) > args.max_paths_per_frame:
        local_gains = gains[idx]
        k = args.max_paths_per_frame
        keep = np.argpartition(local_gains, -k)[-k:]
        idx = idx[keep]

    return idx[np.argsort(ranges[idx])]


def build_frame_candidates(npz_path: str, args) -> tuple[np.ndarray, list[FrameCandidates]]:
    data = np.load(npz_path, allow_pickle=True)
    ue_positions = np.asarray(data["ue_positions"], dtype=float)
    frames: list[FrameCandidates] = []

    for taus, h_complex in zip(data["taus"], data["h_complexs"]):
        ranges = np.asarray(taus, dtype=float).reshape(-1) * C
        h = np.asarray(h_complex)
        if h.ndim != 2 or h.shape[0] == 0:
            frames.append(
                FrameCandidates(
                    ranges=np.array([], dtype=float),
                    gains=np.array([], dtype=float),
                    h_units=np.empty((0, args.num_rows * args.num_cols), dtype=complex),
                    h_vectors=np.empty((0, args.num_rows * args.num_cols), dtype=complex),
                    path_indices=np.array([], dtype=int),
                )
            )
            continue

        gains = norm_rows(h)
        valid = np.isfinite(ranges) & np.isfinite(gains) & (ranges > 0.0) & (gains > 0.0)
        valid_idx = np.where(valid)[0]
        ranges_v = ranges[valid_idx]
        gains_v = gains[valid_idx]
        h_v = h[valid_idx]

        keep_local = select_representative_paths(ranges_v, gains_v, args)
        keep_global = valid_idx[keep_local]
        kept_h = h_v[keep_local]
        kept_gains = gains_v[keep_local]

        frames.append(
            FrameCandidates(
                ranges=ranges_v[keep_local],
                gains=kept_gains,
                h_units=unit_rows(kept_h, kept_gains),
                h_vectors=kept_h,
                path_indices=keep_global.astype(int),
            )
        )

    return ue_positions, frames


def append_track(track: CsiTrack, frame_idx: int, cand: FrameCandidates, path_idx: int):
    track.frames.append(frame_idx)
    track.ranges.append(float(cand.ranges[path_idx]))
    track.gains.append(float(cand.gains[path_idx]))
    track.h_units.append(cand.h_units[path_idx])
    track.h_vectors.append(cand.h_vectors[path_idx])
    track.path_indices.append(int(cand.path_indices[path_idx]))
    track.last_frame = frame_idx


def new_track(frame_idx: int, cand: FrameCandidates, path_idx: int) -> CsiTrack:
    track = CsiTrack([], [], [], [], [], [], frame_idx)
    append_track(track, frame_idx, cand, path_idx)
    return track


def build_csi_tracks(frame_candidates: list[FrameCandidates], args) -> list[CsiTrack]:
    """
    用相邻帧 CSI 相关性 + 距离连续性构造多条路径轨迹。
    """
    tracks: list[CsiTrack] = []

    for frame_idx, cand in enumerate(frame_candidates):
        if len(cand.ranges) == 0:
            continue

        active_ids = [
            i for i, tr in enumerate(tracks)
            if frame_idx - tr.last_frame <= args.max_gap + 1
        ]
        if not active_ids:
            for p in range(len(cand.ranges)):
                tracks.append(new_track(frame_idx, cand, p))
            continue

        active_h = np.vstack([tracks[i].h_units[-1] for i in active_ids])
        active_ranges = np.asarray([tracks[i].ranges[-1] for i in active_ids], dtype=float)
        active_gaps = np.asarray([frame_idx - tracks[i].last_frame for i in active_ids], dtype=float)

        corr = np.abs(active_h @ np.conj(cand.h_units.T))
        range_diff = np.abs(active_ranges[:, None] - cand.ranges[None, :])
        range_gate = args.range_gate_m * np.maximum(active_gaps[:, None], 1.0)
        valid = (corr >= args.min_csi_corr) & (range_diff <= range_gate)

        score = (
            args.corr_weight * corr
            - args.range_jump_weight * range_diff
            - args.gap_weight * active_gaps[:, None]
        )
        score[~valid] = -np.inf

        pair_rows, pair_cols = np.where(np.isfinite(score))
        pair_scores = score[pair_rows, pair_cols]
        order = np.argsort(pair_scores)[::-1]

        used_tracks: set[int] = set()
        used_paths: set[int] = set()
        for ord_idx in order:
            row = int(pair_rows[ord_idx])
            col = int(pair_cols[ord_idx])
            track_id = active_ids[row]
            if track_id in used_tracks or col in used_paths:
                continue
            append_track(tracks[track_id], frame_idx, cand, col)
            used_tracks.add(track_id)
            used_paths.add(col)

        for p in range(len(cand.ranges)):
            if p not in used_paths:
                tracks.append(new_track(frame_idx, cand, p))

    return tracks


def refine_va_from_track(
    ue_positions: np.ndarray,
    frames: np.ndarray,
    ranges_m: np.ndarray,
    z_va: float,
) -> tuple[float, float, float]:
    """
    用单条路径轨迹的距离序列闭式拟合 VA。

    当前数据 UE 基本沿 x=0 直线移动，因此返回 |x| 和 y；+/- x 之后再由
    CSI 空间一致性和象限先验选择。
    """
    y = ue_positions[frames, 1]
    dz = z_va - ue_positions[frames, 2]
    b = ranges_m**2 - y**2 - dz**2
    a = np.column_stack((-2.0 * y, np.ones_like(y)))
    y0, rho2_plus_y02 = np.linalg.lstsq(a, b, rcond=None)[0]
    x_abs2 = max(rho2_plus_y02 - y0**2, 0.0)
    x_abs = float(np.sqrt(x_abs2))
    pred = np.sqrt(x_abs2 + (y - y0) ** 2 + dz**2)
    rmse = float(np.sqrt(np.mean((pred - ranges_m) ** 2)))
    return x_abs, float(y0), rmse


def estimate_from_tracks(
    ue_positions: np.ndarray,
    tracks: list[CsiTrack],
    args,
) -> list[FastVAEstimate]:
    estimates: list[FastVAEstimate] = []

    for track_id, tr in enumerate(tracks):
        if len(tr.frames) < args.min_track_frames:
            continue

        frames = np.asarray(tr.frames, dtype=int)
        ranges = np.asarray(tr.ranges, dtype=float)
        gains = np.asarray(tr.gains, dtype=float)
        h_units = np.asarray(tr.h_units)
        h_vectors = np.asarray(tr.h_vectors)

        x_abs, y0, rmse = refine_va_from_track(ue_positions, frames, ranges, args.z_va)
        if rmse > args.max_range_rmse:
            continue
        if x_abs < args.min_x_abs:
            continue
        if x_abs > args.x_max_abs or y0 < args.y_min or y0 > args.y_max:
            continue

        va_plus = np.array([x_abs, y0, args.z_va], dtype=float)
        va_minus = np.array([-x_abs, y0, args.z_va], dtype=float)
        if np.linalg.norm(va_plus[:2] - BS_POS[:2]) < args.exclude_bs_radius:
            continue

        plus_metrics = evaluate_candidate_csi_spatial_consistency(
            va_plus, ue_positions, frames, h_vectors, BS_POS, args
        )
        minus_metrics = evaluate_candidate_csi_spatial_consistency(
            va_minus, ue_positions, frames, h_vectors, BS_POS, args
        )

        plus_total = plus_metrics[-1] + quadrant_prior_score(
            va_plus, BS_POS, mode=args.quadrant_prior_mode
        )
        minus_total = minus_metrics[-1] + quadrant_prior_score(
            va_minus, BS_POS, mode=args.quadrant_prior_mode
        )

        if minus_total > plus_total:
            selected_sign = -1
            selected_va = va_minus
            metrics = minus_metrics
        else:
            selected_sign = 1
            selected_va = va_plus
            metrics = plus_metrics

        (
            mean_array_corr,
            mean_beam_gain,
            mean_rank1_ratio,
            mean_phase_slope_error,
            csi_spatial_score,
        ) = metrics

        csi_corr = adjacent_csi_corr(h_units, frames)
        hit_ratio = len(frames) / len(ue_positions)
        mean_gain_db = float(20.0 * np.log10(np.mean(gains) + 1e-30))
        phase_err = np.nan_to_num(mean_phase_slope_error, nan=np.pi)

        score = (
            3.0 * hit_ratio
            - 1.2 * rmse
            + 3.0 * np.nan_to_num(csi_corr, nan=0.0)
            + 5.0 * np.nan_to_num(mean_array_corr, nan=0.0)
            + 2.0 * np.nan_to_num(mean_beam_gain, nan=0.0)
            + 1.0 * np.nan_to_num(mean_rank1_ratio, nan=0.0)
            - 1.0 * phase_err
        )

        estimates.append(
            FastVAEstimate(
                selected_va=selected_va,
                delay_plus_va=va_plus,
                delay_minus_va=va_minus,
                selected_sign=selected_sign,
                count=len(frames),
                range_rmse_m=rmse,
                mean_gain_db=mean_gain_db,
                adjacent_csi_corr=csi_corr,
                mean_array_corr=mean_array_corr,
                mean_beam_gain=mean_beam_gain,
                mean_rank1_ratio=mean_rank1_ratio,
                mean_phase_slope_error=mean_phase_slope_error,
                csi_spatial_score=csi_spatial_score,
                score=float(score),
                track_id=track_id,
            )
        )

    estimates.sort(key=lambda e: e.score, reverse=True)
    return deduplicate_estimates(estimates, args)


def deduplicate_estimates(estimates: list[FastVAEstimate], args) -> list[FastVAEstimate]:
    kept: list[FastVAEstimate] = []
    for est in estimates:
        if any(np.linalg.norm(est.selected_va[:2] - old.selected_va[:2]) < args.cluster_sep for old in kept):
            continue
        kept.append(est)
        if len(kept) >= args.max_outputs:
            break
    return kept


def print_estimates(estimates: list[FastVAEstimate], elapsed_s: float):
    print("\n快速 CSI 轨迹法 VA 估计结果")
    print("=" * 76)
    print(f"运行耗时：{elapsed_s:.3f} s")
    if not estimates:
        print("没有找到满足条件的 CSI 路径轨迹；可增大 --top-global-paths 或 --max-paths-per-frame。")
        print("=" * 76)
        return

    for idx, est in enumerate(estimates, start=1):
        v = est.selected_va
        vp = est.delay_plus_va
        vm = est.delay_minus_va
        print(f"[{idx}] track_id={est.track_id}，CSI track-first VA 候选")
        print(f"    delay 拟合 +x VA：[{vp[0]:9.3f}, {vp[1]:9.3f}, {vp[2]:7.3f}]")
        print(f"    delay 拟合 -x VA：[{vm[0]:9.3f}, {vm[1]:9.3f}, {vm[2]:7.3f}]")
        print(f"    最终选择 VA：  [{v[0]:9.3f}, {v[1]:9.3f}, {v[2]:7.3f}]")
        print(f"    selected_sign={est.selected_sign}，轨迹长度={est.count} 帧")
        print(f"    range_rmse={est.range_rmse_m:.3f} m，mean_gain={est.mean_gain_db:.2f} dB")
        print(f"    adjacent_CSI_corr={est.adjacent_csi_corr:.3f}")
        print(f"    mean_array_corr={est.mean_array_corr:.3f}，mean_beam_gain={est.mean_beam_gain:.3f}")
        print(f"    mean_rank1_ratio={est.mean_rank1_ratio:.3f}，phase_slope_error={est.mean_phase_slope_error:.3f}")
        print(f"    csi_spatial_score={est.csi_spatial_score:.3f}，final_score={est.score:.3f}")
    print("=" * 76)
    print(
        "说明：该脚本不做 VA 网格搜索，而是先用 h_complex 的 CSI 相关性快速形成路径轨迹，"
        "再由轨迹距离闭式拟合 VA。若漏掉弱反射径，可增大 --top-global-paths、"
        "--top-per-range-bin 或 --max-paths-per-frame。"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="快速 CSI 路径轨迹法估计虚拟锚点 VA，不做 VA 网格搜索。"
    )
    parser.add_argument("--npz", default="ue_mimo_multipath_data.npz")
    parser.add_argument("--z-va", type=float, default=float(BS_POS[2]))
    parser.add_argument("--top-global-paths", type=int, default=120)
    parser.add_argument("--range-bin-m", type=float, default=0.5)
    parser.add_argument("--top-per-range-bin", type=int, default=2)
    parser.add_argument("--max-paths-per-frame", type=int, default=900)
    parser.add_argument("--min-csi-corr", type=float, default=0.72)
    parser.add_argument("--range-gate-m", type=float, default=1.5)
    parser.add_argument("--max-gap", type=int, default=1)
    parser.add_argument("--corr-weight", type=float, default=1.0)
    parser.add_argument("--range-jump-weight", type=float, default=0.12)
    parser.add_argument("--gap-weight", type=float, default=0.08)
    parser.add_argument("--min-track-frames", type=int, default=25)
    parser.add_argument("--max-range-rmse", type=float, default=0.6)
    parser.add_argument("--min-x-abs", type=float, default=5.0)
    parser.add_argument("--x-max-abs", type=float, default=110.0)
    parser.add_argument("--y-min", type=float, default=-80.0)
    parser.add_argument("--y-max", type=float, default=80.0)
    parser.add_argument("--exclude-bs-radius", type=float, default=12.0)
    parser.add_argument("--cluster-sep", type=float, default=8.0)
    parser.add_argument("--max-outputs", type=int, default=8)
    parser.add_argument("--yaw-deg", type=float, default=135.0)
    parser.add_argument("--array-plane", choices=("xz", "yz"), default="xz")
    parser.add_argument("--num-rows", type=int, default=8)
    parser.add_argument("--num-cols", type=int, default=8)
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument(
        "--quadrant-prior-mode",
        choices=("off", "penalty", "hard"),
        default="penalty",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.perf_counter()
    ue_positions, frame_candidates = build_frame_candidates(args.npz, args)
    selected_counts = [len(c.ranges) for c in frame_candidates]
    print(f"已加载 {args.npz}，UE 帧数={len(ue_positions)}")
    print(
        f"每帧代表路径数：min={min(selected_counts)}, "
        f"mean={np.mean(selected_counts):.1f}, max={max(selected_counts)}"
    )
    tracks = build_csi_tracks(frame_candidates, args)
    long_tracks = sum(len(t.frames) >= args.min_track_frames for t in tracks)
    print(f"构造 CSI 路径轨迹 {len(tracks)} 条，其中长度 >= {args.min_track_frames} 的轨迹 {long_tracks} 条")
    estimates = estimate_from_tracks(ue_positions, tracks, args)
    elapsed = time.perf_counter() - start
    print_estimates(estimates, elapsed)


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))
    main()
