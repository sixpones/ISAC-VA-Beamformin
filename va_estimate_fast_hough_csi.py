"""
快速 Hough + CSI 空间重排序估计虚拟锚点 VA。

这个脚本是 va_estimate_from_tau_csi.py 的快速替代实验：

1. 不枚举完整二维 (x, y) VA 网格；
2. 利用当前 UE 基本沿 x=0 直线运动的结构，只枚举 y0；
3. 每条路径距离 r=c*tau 对每个 y0 可直接反推出 |x|；
4. 在 (|x|, y0) 累加器中找 delay-consistent peaks；
5. 对少量 peaks 再执行路径匹配、最小二乘精修、CSI 空间一致性重排序。

它仍使用 tau 做快速候选生成，但避免了 x-y 双重网格搜索；CSI 用于候选
+x/-x 选择与最终排序。
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np

from va_estimate_fast_csi_tracks import build_frame_candidates
from va_estimate_from_tau_csi import (
    BS_POS,
    adjacent_csi_corr,
    evaluate_candidate_csi_spatial_consistency,
    quadrant_prior_score,
)


@dataclass
class HoughCandidate:
    x_abs: float
    y: float
    votes: float


@dataclass
class HoughVAEstimate:
    selected_va: np.ndarray
    delay_plus_va: np.ndarray
    delay_minus_va: np.ndarray
    selected_sign: int
    votes: float
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


def flatten_observations(ue_positions, frame_candidates):
    frame_ids = []
    ue_y = []
    ue_z = []
    ranges = []
    gains = []
    for frame_idx, (ue, cand) in enumerate(zip(ue_positions, frame_candidates)):
        n = len(cand.ranges)
        if n == 0:
            continue
        frame_ids.append(np.full(n, frame_idx, dtype=int))
        ue_y.append(np.full(n, ue[1], dtype=float))
        ue_z.append(np.full(n, ue[2], dtype=float))
        ranges.append(cand.ranges)
        gains.append(cand.gains)

    return (
        np.concatenate(frame_ids),
        np.concatenate(ue_y),
        np.concatenate(ue_z),
        np.concatenate(ranges),
        np.concatenate(gains),
    )


def build_hough_candidates(ue_positions, frame_candidates, args):
    frame_ids, obs_y, obs_z, obs_ranges, obs_gains = flatten_observations(
        ue_positions, frame_candidates
    )
    y_grid = np.arange(args.y_min, args.y_max + 0.5 * args.y_step, args.y_step)
    x_bins = np.arange(args.x_min, args.x_max + args.x_bin_m, args.x_bin_m)
    acc = np.zeros((len(y_grid), len(x_bins)), dtype=float)
    dz = args.z_va - obs_z

    gain_weight = np.log10(obs_gains + 1e-30)
    gain_weight = gain_weight - np.nanmin(gain_weight)
    gain_weight = 1.0 + args.gain_vote_weight * gain_weight

    for yi, y0 in enumerate(y_grid):
        x2 = obs_ranges**2 - (obs_y - y0) ** 2 - dz**2
        valid = (x2 >= args.x_min**2) & (x2 <= args.x_max**2)
        if not np.any(valid):
            continue
        x_abs = np.sqrt(x2[valid])
        xb = np.rint((x_abs - args.x_min) / args.x_bin_m).astype(int)
        ok = (xb >= 0) & (xb < len(x_bins))
        np.add.at(acc[yi], xb[ok], gain_weight[valid][ok])

    peaks = []
    flat_order = np.argsort(acc.ravel())[::-1]
    for flat_idx in flat_order:
        votes = float(acc.ravel()[flat_idx])
        if votes < args.min_votes:
            break
        yi, xi = np.unravel_index(flat_idx, acc.shape)
        cand = HoughCandidate(
            x_abs=float(args.x_min + xi * args.x_bin_m),
            y=float(y_grid[yi]),
            votes=votes,
        )
        if any(
            np.linalg.norm([cand.x_abs - old.x_abs, cand.y - old.y]) < args.peak_sep_m
            for old in peaks
        ):
            continue
        peaks.append(cand)
        if len(peaks) >= args.preselect_peaks:
            break
    return peaks


def collect_matches_for_candidate(cand: HoughCandidate, ue_positions, frame_candidates, args):
    frames = []
    ranges = []
    gains = []
    h_units = []
    h_vectors = []

    va_abs = np.array([cand.x_abs, cand.y, args.z_va], dtype=float)
    for frame_idx, (ue, fc) in enumerate(zip(ue_positions, frame_candidates)):
        if len(fc.ranges) == 0:
            continue
        target_range = np.linalg.norm(ue - va_abs)
        idx = np.where(np.abs(fc.ranges - target_range) <= args.range_tol)[0]
        if len(idx) == 0:
            continue
        best = idx[np.argmax(fc.gains[idx])]
        frames.append(frame_idx)
        ranges.append(float(fc.ranges[best]))
        gains.append(float(fc.gains[best]))
        h_units.append(fc.h_units[best])
        h_vectors.append(fc.h_vectors[best])

    return (
        np.asarray(frames, dtype=int),
        np.asarray(ranges, dtype=float),
        np.asarray(gains, dtype=float),
        np.asarray(h_units),
        np.asarray(h_vectors),
    )


def refine_line_track(ue_positions, frames, ranges_m, z_va):
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


def estimate_candidates(peaks, ue_positions, frame_candidates, args):
    estimates = []

    for peak in peaks:
        frames, ranges, gains, h_units, h_vectors = collect_matches_for_candidate(
            peak, ue_positions, frame_candidates, args
        )
        if len(frames) < args.min_frames:
            continue

        x_abs, y0, rmse = refine_line_track(ue_positions, frames, ranges, args.z_va)
        if rmse > args.max_range_rmse:
            continue

        va_plus = np.array([x_abs, y0, args.z_va], dtype=float)
        va_minus = np.array([-x_abs, y0, args.z_va], dtype=float)

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
            2.0 * hit_ratio
            - 1.5 * rmse
            + args.vote_score_weight * np.log1p(peak.votes)
            + 3.0 * np.nan_to_num(csi_corr, nan=0.0)
            + 8.0 * np.nan_to_num(mean_array_corr, nan=0.0)
            + 4.0 * np.nan_to_num(mean_beam_gain, nan=0.0)
            + 2.0 * np.nan_to_num(mean_rank1_ratio, nan=0.0)
            - 2.0 * phase_err
        )

        estimates.append(
            HoughVAEstimate(
                selected_va=selected_va,
                delay_plus_va=va_plus,
                delay_minus_va=va_minus,
                selected_sign=selected_sign,
                votes=peak.votes,
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
            )
        )

    estimates.sort(key=lambda e: e.score, reverse=True)
    return deduplicate(estimates, args)


def deduplicate(estimates, args):
    kept = []
    for est in estimates:
        if any(np.linalg.norm(est.selected_va[:2] - old.selected_va[:2]) < args.cluster_sep for old in kept):
            continue
        kept.append(est)
        if len(kept) >= args.max_outputs:
            break
    return kept


def print_estimates(estimates, elapsed_s):
    print("\n快速 Hough + CSI 空间重排序 VA 估计结果")
    print("=" * 76)
    print(f"运行耗时：{elapsed_s:.3f} s")
    if not estimates:
        print("没有找到候选；可增大 --preselect-peaks、--range-tol 或 --max-range-rmse。")
        print("=" * 76)
        return

    for idx, est in enumerate(estimates, start=1):
        vp, vm, v = est.delay_plus_va, est.delay_minus_va, est.selected_va
        print(f"[{idx}] Hough-CSI VA 候选")
        print(f"    Hough/时延精修 +x VA：[{vp[0]:9.3f}, {vp[1]:9.3f}, {vp[2]:7.3f}]")
        print(f"    Hough/时延精修 -x VA：[{vm[0]:9.3f}, {vm[1]:9.3f}, {vm[2]:7.3f}]")
        print(f"    最终选择 VA：        [{v[0]:9.3f}, {v[1]:9.3f}, {v[2]:7.3f}]")
        print(f"    selected_sign={est.selected_sign}，Hough votes={est.votes:.1f}")
        print(f"    matched_frames={est.count}，range_rmse={est.range_rmse_m:.3f} m")
        print(f"    mean_gain={est.mean_gain_db:.2f} dB，adjacent_CSI_corr={est.adjacent_csi_corr:.3f}")
        print(f"    mean_array_corr={est.mean_array_corr:.3f}，mean_beam_gain={est.mean_beam_gain:.3f}")
        print(f"    mean_rank1_ratio={est.mean_rank1_ratio:.3f}，phase_slope_error={est.mean_phase_slope_error:.3f}")
        print(f"    csi_spatial_score={est.csi_spatial_score:.3f}，final_score={est.score:.3f}")
    print("=" * 76)
    print(
        "说明：该方法用一维 y-Hough 快速产生 delay-consistent VA peaks，"
        "再用 h_complex 的 CSI 空间一致性选择 +x/-x 并重排序。它比完整 x-y "
        "VA 网格搜索更快，但仍保留 tau 作为候选生成信息。"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="快速一维 Hough + CSI 空间重排序估计 VA。"
    )
    parser.add_argument("--npz", default="ue_mimo_multipath_data.npz")
    parser.add_argument("--z-va", type=float, default=float(BS_POS[2]))
    parser.add_argument("--x-min", type=float, default=5.0)
    parser.add_argument("--x-max", type=float, default=110.0)
    parser.add_argument("--y-min", type=float, default=-80.0)
    parser.add_argument("--y-max", type=float, default=80.0)
    parser.add_argument("--y-step", type=float, default=1.0)
    parser.add_argument("--x-bin-m", type=float, default=1.0)
    parser.add_argument("--gain-vote-weight", type=float, default=0.02)
    parser.add_argument("--min-votes", type=float, default=20.0)
    parser.add_argument("--preselect-peaks", type=int, default=200)
    parser.add_argument("--peak-sep-m", type=float, default=6.0)
    parser.add_argument("--range-tol", type=float, default=0.35)
    parser.add_argument("--min-frames", type=int, default=25)
    parser.add_argument("--max-range-rmse", type=float, default=0.6)
    parser.add_argument("--top-global-paths", type=int, default=120)
    parser.add_argument("--range-bin-m", type=float, default=0.5)
    parser.add_argument("--top-per-range-bin", type=int, default=2)
    parser.add_argument("--max-paths-per-frame", type=int, default=900)
    parser.add_argument("--cluster-sep", type=float, default=8.0)
    parser.add_argument("--max-outputs", type=int, default=30)
    parser.add_argument("--vote-score-weight", type=float, default=0.5)
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
    counts = [len(c.ranges) for c in frame_candidates]
    print(f"已加载 {args.npz}，UE 帧数={len(ue_positions)}")
    print(f"每帧代表路径数：min={min(counts)}, mean={np.mean(counts):.1f}, max={max(counts)}")
    peaks = build_hough_candidates(ue_positions, frame_candidates, args)
    print(f"Hough 预选 peaks 数量：{len(peaks)}")
    estimates = estimate_candidates(peaks, ue_positions, frame_candidates, args)
    elapsed_s = time.perf_counter() - start
    print_estimates(estimates, elapsed_s)


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))
    main()
