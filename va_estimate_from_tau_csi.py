"""
基于路径时延和逐径CSI估计虚拟锚点(Virtual Anchor)坐标。

==== 算法核心思路 ====

虚拟锚点是反射体的镜像位置。当一条多径信号从BS出发、经反射体到达UE时，
其传播时延 τ 乘以光速 c 得到的距离，等于 UE 到虚拟锚点的直线距离：
    c * τ_i ~= ||UE_i - VA||

本脚本的核心思想是：不依赖 AoA/AoD（到达角/离开角）字段，而是通过以下两步估计VA：

第一步 -- 时延网格搜索（粗定位）：
  在 3D 空间中按网格枚举候选 VA 位置 (|x|, y, z)，对每个候选位置，计算它与
  所有 UE 帧之间的预测距离，然后在每帧的实际多径时延中寻找匹配的径。
  如果某个候选位置能在足够多的帧中找到匹配（count >= min_frames），则
  认为该位置是一个"时延一致"的 VA 候选。这一步只用了时延信息。

第二步 -- CSI 路径关联（精确定位与跟踪）：
  当多条路径具有相近的时延时（即在同一距离容差范围内的多个径），用 CSI
  来区分并选择时间上最连续的那条路径。具体做法是：对每个候选 VA，沿时间轴
  逐帧选择与前一帧信道向量相关性最高的那条径（而非简单选能量最强的），
  从而得到一条在时间上连续、CSI 一致的路径轨迹。

第三步 -- 轨迹精细化：
  利用收集到的 (帧索引, 距离) 数据对，通过最小二乘拟合进一步精细化 |x| 和 y 坐标，
  同时计算距离 RMSE 作为置信度参考。

==== 关于左右镜像歧义 ====

当前数据集中 UE 沿 x=0 直线移动，仅靠时延无法区分 +x 和 -x 方向
（因为距离公式中 x 以平方形式出现）。因此脚本对每个 VA 输出一对正负 x
镜像候选，需额外的 UE 轨迹、第二个 BS 或天线阵列导向模型来消除此歧义。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np


# ---------- 物理常量与基站位置 ----------
C = 299792458.0                        # 光速 (m/s)
BS_POS = np.array([40.0, -40.0, 15.0]) # 基站 3D 坐标


# ---------- 数据结构 ----------
@dataclass
class TrackEstimate:
    """单个 VA 的估计结果"""
    x_abs: float                       # |x| 坐标（取绝对值，左右镜像歧义）
    y: float                           # y 坐标
    z: float                           # z 坐标
    selected_sign: int                 # CSI 空间一致性选择的符号：+1 或 -1
    selected_va: np.ndarray            # CSI 重排序后最终选择的 VA 坐标
    count: int                         # 匹配到的帧数
    mean_gain_db: float                # 平均路径增益 (dB)
    mean_adjacent_csi_corr: float      # 相邻帧 CSI 的平均相关系数
    mean_array_corr: float             # 理论 UPA steering 与实测 CSI 的平均相关
    mean_beam_gain: float              # 候选 VA 诱导波束的平均归一化增益
    mean_rank1_ratio: float            # 8x8 CSI 矩阵 rank-1 纯度
    mean_phase_slope_error: float      # 实测/理论相位斜率误差
    csi_spatial_score: float           # CSI 空间一致性综合评分
    range_rmse_m: float                # 距离拟合 RMSE (米)
    score: float                       # 最终综合评分
    matched_frames: np.ndarray         # 匹配上的帧索引数组
    matched_ranges_m: np.ndarray       # 匹配上的路径距离数组 (米)


def _norm_rows(h: np.ndarray) -> np.ndarray:
    if h.size == 0:
        return np.array([], dtype=float)
    return np.linalg.norm(h, axis=1)


def same_xy_quadrant(p, q, eps=1e-9):
    """
    判断 p 和 q 在 x-y 平面是否处于同一象限。
    坐标接近 0 时不强行判定，避免边界误杀。
    """
    px, py = p[0], p[1]
    qx, qy = q[0], q[1]

    if abs(px) < eps or abs(py) < eps or abs(qx) < eps or abs(qy) < eps:
        return False

    return (np.sign(px) == np.sign(qx)) and (np.sign(py) == np.sign(qy))


def quadrant_prior_score(va, bs_pos, mode="penalty"):
    """
    当前场景先验：目标 VA 不应与 BS0 位于同一 x-y 象限。
    mode="hard": 同象限直接拒绝
    mode="penalty": 同象限扣分
    """
    same = same_xy_quadrant(va, bs_pos)

    if mode == "hard":
        return -np.inf if same else 0.0

    if mode == "penalty":
        return -2.0 if same else 0.5

    return 0.0


def mirror_plane_from_va(bs_pos, va):
    """
    由 BS 和候选 VA 反推镜面反射平面。

    VA 是 BS 关于反射平面的镜像点，因此反射平面是 BS-VA 连线的垂直平分面。
    返回单位法向量 normal 和平面参数 d，使 normal @ x + d = 0。
    """
    bs_pos = np.asarray(bs_pos, dtype=float)
    va = np.asarray(va, dtype=float)
    normal = bs_pos - va
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return None, None
    normal = normal / norm
    midpoint = 0.5 * (bs_pos + va)
    d = -float(normal @ midpoint)
    return normal, d


def reflection_point_from_va(bs_pos, va, ue):
    """
    计算 VA->UE 直线与候选反射平面的交点 R。

    如果直线与平面近似平行，或交点不在线段 VA->UE 上，则返回 None。
    """
    normal, d = mirror_plane_from_va(bs_pos, va)
    if normal is None:
        return None

    va = np.asarray(va, dtype=float)
    ue = np.asarray(ue, dtype=float)
    direction = ue - va
    denom = float(normal @ direction)
    if abs(denom) < 1e-12:
        return None

    t = -float(normal @ va + d) / denom
    if t < 0.0 or t > 1.0:
        return None
    return va + t * direction


def world_to_bs_local_direction(u_world, yaw_deg):
    """
    将世界坐标系方向向量转换到 BS 局部坐标系。

    这里先只使用 yaw 旋转。BS0 在 ue_multipath_sim.py 中约为 135 deg。
    """
    u_world = np.asarray(u_world, dtype=float)
    norm = np.linalg.norm(u_world)
    if norm < 1e-12:
        return None
    u_world = u_world / norm

    yaw = np.deg2rad(yaw_deg)
    # Sionna 当前 orientation/yaw 与 h_complex 阵列相位约定下，使用 +yaw
    # 能把世界方向映射到与 BS 侧 CSI 一致的局部 steering 坐标。
    c = np.cos(yaw)
    s = np.sin(yaw)
    rot = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    u_local = rot @ u_world
    local_norm = np.linalg.norm(u_local)
    if local_norm < 1e-12:
        return None
    return u_local / local_norm


def upa_steering_from_direction(
    u_local,
    num_rows=8,
    num_cols=8,
    spacing=0.5,
    array_plane="xz",
):
    """
    根据 BS 局部方向向量构造 UPA steering vector。

    spacing 以波长为单位；0.5 表示半波长间距。array_plane="xz" 表示阵列横向
    轴使用局部 x、纵向轴使用局部 z；"yz" 表示横向轴使用局部 y、纵向轴使用局部 z。
    """
    u_local = np.asarray(u_local, dtype=float)
    norm = np.linalg.norm(u_local)
    if norm < 1e-12:
        return None
    u_local = u_local / norm

    if array_plane == "xz":
        horizontal_dir = u_local[0]
        vertical_dir = u_local[2]
    elif array_plane == "yz":
        horizontal_dir = u_local[1]
        vertical_dir = u_local[2]
    else:
        raise ValueError(f"Unsupported array_plane={array_plane!r}")

    row_idx = np.arange(num_rows, dtype=float) - 0.5 * (num_rows - 1)
    col_idx = np.arange(num_cols, dtype=float) - 0.5 * (num_cols - 1)

    steering = []
    for r in row_idx:
        for c in col_idx:
            phase = -2.0 * np.pi * spacing * (c * horizontal_dir + r * vertical_dir)
            steering.append(np.exp(1j * phase))
    steering = np.asarray(steering, dtype=complex)
    return steering / (np.linalg.norm(steering) + 1e-30)


def normalized_csi_corr(h, a):
    """计算 |h^H a| / (||h|| ||a||)。"""
    h = np.asarray(h, dtype=complex)
    a = np.asarray(a, dtype=complex)
    if h.size == 0 or a.size == 0 or h.shape != a.shape:
        return float("nan")
    denom = np.linalg.norm(h) * np.linalg.norm(a)
    if denom < 1e-30:
        return float("nan")
    return float(abs(np.vdot(h, a)) / denom)


def rank1_ratio_from_csi(h, num_rows=8, num_cols=8):
    """
    将 64 维 CSI reshape 成 8x8 矩阵，返回 s[0]^2 / sum(s^2)。
    """
    h = np.asarray(h, dtype=complex)
    if h.size != num_rows * num_cols:
        return float("nan")
    mat = h.reshape(num_rows, num_cols)
    s = np.linalg.svd(mat, compute_uv=False)
    power = float(np.sum(s**2))
    if power < 1e-30:
        return float("nan")
    return float((s[0] ** 2) / power)


def _circular_mean_phase(phases: np.ndarray) -> float:
    phases = np.asarray(phases, dtype=float)
    if phases.size == 0:
        return float("nan")
    return float(np.angle(np.mean(np.exp(1j * phases))))


def phase_slope_from_csi(h, num_rows=8, num_cols=8):
    """
    计算 8x8 CSI 矩阵横向和纵向平均相邻天线相位差。
    """
    h = np.asarray(h, dtype=complex)
    if h.size != num_rows * num_cols:
        return float("nan"), float("nan")
    mat = h.reshape(num_rows, num_cols)
    horiz = np.angle(mat[:, 1:] * np.conj(mat[:, :-1])).ravel()
    vert = np.angle(mat[1:, :] * np.conj(mat[:-1, :])).ravel()
    return _circular_mean_phase(horiz), _circular_mean_phase(vert)


def phase_slope_from_steering(a, num_rows=8, num_cols=8):
    """对理论 steering vector 计算同样的横向/纵向相位斜率。"""
    return phase_slope_from_csi(a, num_rows=num_rows, num_cols=num_cols)


def wrap_phase_error(a, b):
    """返回 abs(arctan2(sin(a-b), cos(a-b)))，避免 +-pi 跳变。"""
    if np.isnan(a) or np.isnan(b):
        return float("nan")
    return float(abs(np.arctan2(np.sin(a - b), np.cos(a - b))))


def evaluate_candidate_csi_spatial_consistency(
    va,
    ue_positions,
    frames,
    h_units_or_h_vectors,
    bs_pos,
    args,
):
    """
    计算一个具体 VA 候选的 CSI 空间一致性。

    该函数不使用 Sionna 显式 AoA/AoD 字段，而是由候选 VA 诱导反射点和 BS 侧
    UPA steering，再与 matched path 的原始 h_complex 做空间相关和相位斜率比较。
    """
    va = np.asarray(va, dtype=float)
    h_vectors = np.asarray(h_units_or_h_vectors, dtype=complex)

    array_corrs = []
    beam_gains = []
    rank1_ratios = []
    phase_errors = []

    for frame_idx, h in zip(frames, h_vectors):
        ue = ue_positions[frame_idx]
        refl = reflection_point_from_va(bs_pos, va, ue)
        if refl is None:
            continue

        u_world = refl - bs_pos
        u_local = world_to_bs_local_direction(u_world, args.yaw_deg)
        if u_local is None:
            continue

        steering = upa_steering_from_direction(
            u_local,
            num_rows=args.num_rows,
            num_cols=args.num_cols,
            spacing=args.spacing,
            array_plane=args.array_plane,
        )
        if steering is None:
            continue

        corr = normalized_csi_corr(h, steering)
        if not np.isnan(corr):
            array_corrs.append(corr)
            beam_gains.append(corr**2)

        rank1 = rank1_ratio_from_csi(
            h,
            num_rows=args.num_rows,
            num_cols=args.num_cols,
        )
        if not np.isnan(rank1):
            rank1_ratios.append(rank1)

        h_slope_h, h_slope_v = phase_slope_from_csi(
            h,
            num_rows=args.num_rows,
            num_cols=args.num_cols,
        )
        a_slope_h, a_slope_v = phase_slope_from_steering(
            steering,
            num_rows=args.num_rows,
            num_cols=args.num_cols,
        )
        err_h = wrap_phase_error(h_slope_h, a_slope_h)
        err_v = wrap_phase_error(h_slope_v, a_slope_v)
        errs = [e for e in (err_h, err_v) if not np.isnan(e)]
        if errs:
            phase_errors.append(float(np.mean(errs)))

    mean_array_corr = float(np.mean(array_corrs)) if array_corrs else float("nan")
    mean_beam_gain = float(np.mean(beam_gains)) if beam_gains else float("nan")
    mean_rank1_ratio = float(np.mean(rank1_ratios)) if rank1_ratios else float("nan")
    mean_phase_slope_error = (
        float(np.mean(phase_errors)) if phase_errors else float("nan")
    )

    csi_spatial_score = (
        4.0 * np.nan_to_num(mean_array_corr, nan=0.0)
        + 2.0 * np.nan_to_num(mean_beam_gain, nan=0.0)
        + 1.0 * np.nan_to_num(mean_rank1_ratio, nan=0.0)
        - 1.0 * np.nan_to_num(mean_phase_slope_error, nan=np.pi)
    )

    return (
        mean_array_corr,
        mean_beam_gain,
        mean_rank1_ratio,
        mean_phase_slope_error,
        float(csi_spatial_score),
    )


def load_multipath(npz_path: str):
    """加载多径数据：UE 位置、每帧各径时延 τ、每帧各径复信道增益 h"""
    data = np.load(npz_path, allow_pickle=True)
    return (
        np.asarray(data["ue_positions"], dtype=float),   # (N_frames, 3)
        data["taus"],                                    # 时延列表，每帧可能有不同数量的径
        data["h_complexs"],                              # 复 CSI 矩阵，与 taus 一一对应
    )


def build_frame_cache(taus, h_complexs):
    """
    构建逐帧缓存，预处理每帧的多径信息：
    - 将时延 τ 转换为距离：r = τ * c
    - 按距离从小到大排序（方便后续二分查找）
    - 返回 [{"ranges", "h", "norms"}, ...] 列表
    """
    cache = []
    for ta, hh in zip(taus, h_complexs):
        ranges_m = np.asarray(ta, dtype=float) * C      # 时延 → 距离
        hh = np.asarray(hh)
        norms = _norm_rows(hh)                           # 每条径的 CSI 幅度
        order = np.argsort(ranges_m)                     # 按距离排序，便于二分查找
        cache.append(
            {
                "ranges": ranges_m[order],
                "h": hh[order],
                "norms": norms[order],
            }
        )
    return cache


def best_delay_match(frame, target_range_m: float, tol_m: float):
    """
    在单帧的多径中，寻找与目标距离最匹配的径。

    策略：在 [target - tol, target + tol] 距离窗口内，选 CSI 幅度最大的径。
    这基于一个合理假设：与候选 VA 真正对应的反射径通常具有较强的能量。
    返回 (径索引, 实际距离, CSI幅度, 复信道向量)，找不到则返回 None。
    """
    ranges = frame["ranges"]
    # 用二分查找快速定位距离窗口 [target-tol, target+tol]
    left = np.searchsorted(ranges, target_range_m - tol_m, side="left")
    right = np.searchsorted(ranges, target_range_m + tol_m, side="right")
    if right <= left:
        return None
    local = slice(left, right)
    norms = frame["norms"][local]
    if len(norms) == 0:
        return None
    rel = int(np.argmax(norms))   # 在窗口内选能量最强的径
    idx = left + rel
    return idx, ranges[idx], norms[rel], frame["h"][idx]


def delay_grid_score(
    ue_positions: np.ndarray,
    frame_cache,
    x_abs_grid: np.ndarray,
    y_grid: np.ndarray,
    z_va: float,
    range_tol_m: float,
    min_frames: int,
    exclude_bs_profile_radius_m: float,
):
    """
    === 第一步：时延网格搜索（粗定位）===

    在 x-y 平面按网格枚举候选 VA 位置，对每个候选位置：
    1. 计算它到所有 UE 帧之间的预测距离：d_i = ||UE_i - (|x|, y, z)||
    2. 在每帧的多径时延中寻找匹配（距离在 ±range_tol 内的径）
    3. 统计匹配上的帧数 count，如果 count >= min_frames，保留为候选

    评分 = count + 0.02 * gain_sum，帧数为主、增益为辅。

    排除基站附近区域（直接径/LoS 会产生与基站位置相同的距离轮廓，
    干扰对真实反射体的搜索）。
    """
    candidates = []
    dz = z_va - ue_positions[:, 2]             # UE 与候选 VA 的高度差（所有帧）

    for x_abs in x_abs_grid:
        if x_abs < 1e-9:
            continue
        for y0 in y_grid:
            # 排除基站附近的候选位置，避免将 LoS 直接径误判为反射径的 VA
            if (
                np.linalg.norm([x_abs - abs(BS_POS[0]), y0 - BS_POS[1]])
                < exclude_bs_profile_radius_m
            ):
                continue

            # VA 到所有 UE 帧的预测距离（欧氏距离）
            pred_ranges = np.sqrt(
                (ue_positions[:, 0] - x_abs) ** 2
                + (ue_positions[:, 1] - y0) ** 2
                + dz**2
            )

            # 逐帧检查：预测距离是否在实测多径中能找到匹配
            count = 0
            gain_sum = 0.0
            for frame, target in zip(frame_cache, pred_ranges):
                match = best_delay_match(frame, target, range_tol_m)
                if match is None:
                    continue
                _, _, norm_h, _ = match
                count += 1
                gain_sum += np.log10(norm_h + 1e-30)   # dB 标度累积

            if count >= min_frames:                    # 足够多的帧匹配 → 有效候选
                score = count + 0.02 * gain_sum        # 综合评分：帧数权重 >> 增益权重
                candidates.append((score, count, x_abs, y0))

    candidates.sort(reverse=True, key=lambda item: item[0])  # 按评分降序排列
    return candidates


def collect_csi_track(
    ue_positions: np.ndarray,
    frame_cache,
    x_abs: float,
    y0: float,
    z_va: float,
    range_tol_m: float,
    csi_weight: float,
):
    """
    === 第二步：CSI 路径关联（时间连续性跟踪）===

    对时延网格搜索得到的候选 VA，沿时间轴逐帧选择最合适的那条径。

    关键问题：距离窗口 [target-tol, target+tol] 内可能有多条径
    （来自不同反射体但距离相近），每一帧独立选能量最强的径会导致
    路径在帧间跳变（这次选了反射体A的径，下次选了反射体B的径）。

    解决方案：引入 CSI 相干性约束。除了考虑径的幅度 norms，还计算
    当前帧候选径与上一帧已选径的 CSI 复相关系数 corr。综合评分：
        score = log10(norm) + csi_weight * |unit @ conj(prev_h)|
    选综合评分最高的径，确保选出的径在时间上 CSI 连续（同一反射体）。

    首帧：没有上一帧参考，直接选能量最强的径。
    """
    # 候选 VA 到所有 UE 帧的预测距离
    pred_ranges = np.linalg.norm(
        ue_positions - np.array([x_abs, y0, z_va], dtype=float), axis=1
    )
    frames = []       # 匹配上的帧索引
    ranges = []       # 匹配上的实际距离
    gains = []        # 匹配上的 CSI 幅度
    h_units = []      # 匹配上的归一化 CSI 向量
    h_vectors = []    # 匹配上的原始 64 维复 CSI 向量
    prev_h = None     # 上一帧选择的归一化 CSI 向量

    for frame_idx, (frame, target) in enumerate(zip(frame_cache, pred_ranges)):
        ranges_sorted = frame["ranges"]
        # 在距离窗口中定位候选径范围
        left = np.searchsorted(ranges_sorted, target - range_tol_m, side="left")
        right = np.searchsorted(ranges_sorted, target + range_tol_m, side="right")
        if right <= left:
            continue

        h_cand = frame["h"][left:right]      # 窗口内的所有候选 CSI 向量
        norms = frame["norms"][left:right]    # 对应的幅度
        if len(norms) == 0:
            continue

        unit = h_cand / (norms[:, None] + 1e-30)  # 归一化到单位向量
        if prev_h is None:
            # 首帧：选能量最强的径
            rel = int(np.argmax(norms))
        else:
            # 后续帧：综合考虑幅度和与上一帧的 CSI 相关性
            corr = np.abs(unit @ np.conj(prev_h))           # 复相关系数的模
            norm_score = np.log10(norms + 1e-30)            # 幅度评分（dB 标度）
            rel = int(np.argmax(norm_score + csi_weight * corr))  # 综合评分

        h_unit = unit[rel]
        prev_h = h_unit
        frames.append(frame_idx)
        ranges.append(ranges_sorted[left + rel])
        gains.append(norms[rel])
        h_units.append(h_unit)
        h_vectors.append(h_cand[rel])

    return (
        np.asarray(frames, dtype=int),
        np.asarray(ranges, dtype=float),
        np.asarray(gains, dtype=float),
        np.asarray(h_units),
        np.asarray(h_vectors),
    )


def refine_line_track(
    ue_positions: np.ndarray,
    frames: np.ndarray,
    ranges_m: np.ndarray,
    z_va: float,
):
    """
    === 第三步：最小二乘精细化 ===

    利用收集到的 (帧索引, 实测距离) 对，通过最小二乘拟合精细化 VA 的 |x| 和 y 坐标。

    距离方程：r² = (x_va)² + (y - y₀)² + (z_va - z_ue)²
    展开后：r² - y² - dz² = (-2y)·y₀ + (ρ² + y₀²)
    其中 ρ² = x²（因为我们只用 |x|），这是一个关于 y₀ 和 (ρ² + y₀²) 的线性方程，
    可用最小二乘直接求解。再由 y₀ 和 ρ² + y₀² 反算 |x|。
    """
    y = ue_positions[frames, 1]
    dz = z_va - ue_positions[frames, 2]
    b = ranges_m**2 - y**2 - dz**2
    # A @ [y0, ρ² + y0²] = b
    a = np.column_stack((-2.0 * y, np.ones_like(y)))
    y0, rho2_plus_y02 = np.linalg.lstsq(a, b, rcond=None)[0]
    x_abs2 = max(rho2_plus_y02 - y0**2, 0.0)
    x_abs = float(np.sqrt(x_abs2))
    # 用细化后的坐标反算预测距离，评估 RMSE
    pred = np.sqrt(x_abs2 + (y - y0) ** 2 + dz**2)
    rmse = float(np.sqrt(np.mean((pred - ranges_m) ** 2)))
    return x_abs, float(y0), rmse


def adjacent_csi_corr(h_units: np.ndarray, frames: np.ndarray) -> float:
    """
    计算相邻帧之间 CSI 的平均相关系数，用于衡量路径跟踪的时间连续性。
    只计算帧索引连续的相邻对（跳过非连续帧）。
    相关性高 → 说明选出的径确实来自同一反射体。
    """
    vals = []
    for i in range(len(h_units) - 1):
        if frames[i + 1] != frames[i] + 1:
            continue
        vals.append(float(abs(np.vdot(h_units[i], h_units[i + 1]))))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def cluster_candidates(candidates, min_sep_m: float, max_clusters: int):
    """
    对时延网格搜索得到的候选进行空间聚类，去重：
    相邻的网格点可能对应同一个真实 VA，通过最小间距 min_sep_m
    过滤掉过于接近的候选，保留评分最高的前 max_clusters 个。
    """
    clusters = []
    for _, count, x_abs, y0 in candidates:
        point = np.array([x_abs, y0], dtype=float)
        if any(np.linalg.norm(point - np.array([c[0], c[1]])) < min_sep_m for c in clusters):
            continue
        clusters.append((x_abs, y0, count))
        if len(clusters) >= max_clusters:
            break
    return clusters


def estimate(args):
    """
    === 主估计流程 ===

    三步走：
    1. delay_grid_score():  时延网格搜索 → 粗定位，得到候选 VA 列表
    2. cluster_candidates(): 空间聚类去重   → 合并相邻候选，保留独立 VA
    3. collect_csi_track():  逐候选执行 CSI 路径关联  → 精确跟踪单个反射体
    4. refine_line_track():  最小二乘精细化坐标     → 最终 VA 位置

    最终评分 = 匹配帧数 - 2*RMSE + 5*相邻CSI相关系数
    帧数多、距离误差小、CSI 时间连续 → 评分高 → 高置信度 VA
    """
    # ---- 加载数据 ----
    ue_positions, taus, h_complexs = load_multipath(args.npz)
    frame_cache = build_frame_cache(taus, h_complexs)

    # ---- 定义时延搜索网格 ----
    x_abs_grid = np.arange(args.x_min, args.x_max + 0.5 * args.grid_step, args.grid_step)
    y_grid = np.arange(args.y_min, args.y_max + 0.5 * args.grid_step, args.grid_step)

    print(f"已加载数据：{args.npz}，共 {len(ue_positions)} 帧 UE 位置")
    print(
        f"时延粗搜索网格：|x| ∈ [{x_abs_grid[0]:.1f}, {x_abs_grid[-1]:.1f}] m，"
        f"y ∈ [{y_grid[0]:.1f}, {y_grid[-1]:.1f}] m，步长={args.grid_step:.2f} m"
    )
    print(
        f"CSI 空间重排序配置：BS yaw={args.yaw_deg:.1f}°，"
        f"阵列平面={args.array_plane}，UPA={args.num_rows}x{args.num_cols}，"
        f"阵元间距={args.spacing:.2f}λ，"
        f"是否关闭 CSI 空间评分={args.disable_csi_spatial_score}"
    )
    print(f"象限先验模式：{args.quadrant_prior_mode}")

    # ---- 第一步：时延网格搜索（只使用时延 τ）----
    candidates = delay_grid_score(
        ue_positions,
        frame_cache,
        x_abs_grid,
        y_grid,
        args.z_va,
        args.range_tol,
        args.min_frames,
        args.exclude_bs_profile_radius,
    )
    if not candidates:
        raise RuntimeError("没有找到时延一致的 VA 候选；可以尝试增大 --range-tol。")

    # ---- 去重：空间聚类，合并相邻的网格候选 ----
    clusters = cluster_candidates(candidates, args.cluster_sep, args.max_outputs)

    # ---- 第二步 + 第三步：对每个候选 VA 进行 CSI 跟踪 + 坐标精细化 ----
    estimates = []
    for x_abs, y0, _ in clusters:
        # 第二步：CSI 路径关联跟踪 —— 在距离窗口内选 CSI 最连续的那条径
        frames, ranges, gains, h_units, h_vectors = collect_csi_track(
            ue_positions,
            frame_cache,
            x_abs,
            y0,
            args.z_va,
            args.range_tol,
            args.csi_weight,
        )
        if len(frames) < args.min_frames:
            continue
        # 第三步：最小二乘精细化坐标
        x_refined, y_refined, rmse = refine_line_track(
            ue_positions, frames, ranges, args.z_va
        )
        mean_gain_db = float(20.0 * np.log10(np.mean(gains) + 1e-30))
        csi_corr = adjacent_csi_corr(h_units, frames)

        va_plus = np.array([x_refined, y_refined, args.z_va], dtype=float)
        va_minus = np.array([-x_refined, y_refined, args.z_va], dtype=float)

        if args.disable_csi_spatial_score:
            selected_sign = 0
            selected_va = va_plus
            mean_array_corr = float("nan")
            mean_beam_gain = float("nan")
            mean_rank1_ratio = float("nan")
            mean_phase_slope_error = float("nan")
            csi_spatial_score = float("nan")
            score = len(frames) - 2.0 * rmse + 5.0 * np.nan_to_num(csi_corr, nan=0.0)
        else:
            plus_metrics = evaluate_candidate_csi_spatial_consistency(
                va_plus,
                ue_positions,
                frames,
                h_vectors,
                BS_POS,
                args,
            )
            minus_metrics = evaluate_candidate_csi_spatial_consistency(
                va_minus,
                ue_positions,
                frames,
                h_vectors,
                BS_POS,
                args,
            )

            plus_prior = quadrant_prior_score(
                va_plus,
                BS_POS,
                mode=args.quadrant_prior_mode,
            )
            minus_prior = quadrant_prior_score(
                va_minus,
                BS_POS,
                mode=args.quadrant_prior_mode,
            )
            plus_total = plus_metrics[-1] + plus_prior
            minus_total = minus_metrics[-1] + minus_prior

            if minus_total > plus_total:
                selected_sign = -1
                selected_va = va_minus
                selected_metrics = minus_metrics
            else:
                selected_sign = 1
                selected_va = va_plus
                selected_metrics = plus_metrics

            (
                mean_array_corr,
                mean_beam_gain,
                mean_rank1_ratio,
                mean_phase_slope_error,
                csi_spatial_score,
            ) = selected_metrics

            hit_ratio = len(frames) / len(ue_positions)
            phase_error_for_score = np.nan_to_num(mean_phase_slope_error, nan=np.pi)
            score = (
                2.0 * hit_ratio
                - 1.5 * rmse
                + 3.0 * np.nan_to_num(csi_corr, nan=0.0)
                + 8.0 * np.nan_to_num(mean_array_corr, nan=0.0)
                + 4.0 * np.nan_to_num(mean_beam_gain, nan=0.0)
                + 2.0 * np.nan_to_num(mean_rank1_ratio, nan=0.0)
                - 2.0 * phase_error_for_score
            )

        estimates.append(
            TrackEstimate(
                x_abs=x_refined,
                y=y_refined,
                z=args.z_va,
                selected_sign=selected_sign,
                selected_va=selected_va,
                count=len(frames),
                mean_gain_db=mean_gain_db,
                mean_adjacent_csi_corr=csi_corr,
                mean_array_corr=mean_array_corr,
                mean_beam_gain=mean_beam_gain,
                mean_rank1_ratio=mean_rank1_ratio,
                mean_phase_slope_error=mean_phase_slope_error,
                csi_spatial_score=csi_spatial_score,
                range_rmse_m=rmse,
                score=score,
                matched_frames=frames,
                matched_ranges_m=ranges,
            )
        )

    estimates.sort(key=lambda item: item.score, reverse=True)  # 按评分降序
    return estimates[: args.max_outputs]


def print_estimates(estimates):
    print("\n基于“时延粗筛 + CSI 空间一致性重排序”的 VA 估计结果")
    print("=" * 72)
    for idx, est in enumerate(estimates, start=1):
        selected_text = (
            f"[{est.selected_va[0]:9.3f}, {est.selected_va[1]:9.3f}, {est.selected_va[2]:7.3f}]"
            if est.selected_sign != 0
            else "已关闭 CSI 空间评分，未执行 +x/-x 空间重排序"
        )
        print(f"[{idx}] CSI 重排序后的 VA 候选")
        print(f"    时延一致 +x 候选：[{ est.x_abs:9.3f}, {est.y:9.3f}, {est.z:7.3f}]")
        print(f"    时延一致 -x 候选：[{-est.x_abs:9.3f}, {est.y:9.3f}, {est.z:7.3f}]")
        print(f"    最终选择的 VA：{selected_text}")
        print(f"    selected_sign={est.selected_sign}  （+1 表示选择 +x，-1 表示选择 -x）")
        print(f"    matched_frames={est.count}  （成功跟踪到该路径的 UE 帧数）")
        print(f"    range_rmse={est.range_rmse_m:.3f} m  （时延距离拟合误差）")
        print(f"    mean_gain={est.mean_gain_db:.2f} dB  （匹配路径平均增益）")
        print(f"    adjacent_CSI_corr={est.mean_adjacent_csi_corr:.3f}  （相邻帧 CSI 连续性）")
        print(f"    mean_array_corr={est.mean_array_corr:.3f}  （理论阵列流形与实测 CSI 的相关系数）")
        print(f"    mean_beam_gain={est.mean_beam_gain:.3f}  （候选 VA 诱导波束的归一化增益）")
        print(f"    mean_rank1_ratio={est.mean_rank1_ratio:.3f}  （8x8 CSI 的 rank-1 纯度）")
        print(f"    mean_phase_slope_error={est.mean_phase_slope_error:.3f}  （实测/理论相位斜率误差，越小越好）")
        print(f"    csi_spatial_score={est.csi_spatial_score:.3f}  （CSI 空间一致性评分）")
        print(f"    final_score={est.score:.3f}  （最终综合评分，用于候选排序）")
    print("=" * 72)
    print(
        "说明：本脚本没有使用 Sionna 显式 AoA/AoD 字段。+x/-x 的选择由 BS 侧 "
        "CSI 空间一致性与象限先验共同决定。若阵列相关或相位斜率诊断较弱，"
        "可调节 --yaw-deg、--array-plane，或使用 --disable-csi-spatial-score "
        "回退到旧评分方式做对比。"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用时延粗筛与 CSI 空间一致性重排序估计虚拟锚点 VA。"
    )
    parser.add_argument("--npz", default="ue_mimo_multipath_data.npz")
    parser.add_argument("--z-va", type=float, default=float(BS_POS[2]))
    parser.add_argument("--range-tol", type=float, default=0.35)
    parser.add_argument("--grid-step", type=float, default=1.0)
    parser.add_argument("--x-min", type=float, default=5.0)
    parser.add_argument("--x-max", type=float, default=100.0)
    parser.add_argument("--y-min", type=float, default=-80.0)
    parser.add_argument("--y-max", type=float, default=80.0)
    parser.add_argument("--min-frames", type=int, default=25)
    parser.add_argument("--max-outputs", type=int, default=6)
    parser.add_argument("--cluster-sep", type=float, default=8.0)
    parser.add_argument("--csi-weight", type=float, default=1.0)
    parser.add_argument("--exclude-bs-profile-radius", type=float, default=15.0)
    parser.add_argument("--yaw-deg", type=float, default=135.0)
    parser.add_argument("--array-plane", choices=("xz", "yz"), default="xz")
    parser.add_argument("--num-rows", type=int, default=8)
    parser.add_argument("--num-cols", type=int, default=8)
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument(
        "--disable-csi-spatial-score",
        action="store_true",
        help="关闭 CSI 空间重排序，回退到旧的时延 + CSI 连续性评分方式。",
    )
    parser.add_argument(
        "--quadrant-prior-mode",
        choices=("off", "penalty", "hard"),
        default="penalty",
        help=(
            "用于 +x/-x 选择的场景象限先验：目标 VA 不应与 BS0 位于同一 x-y "
            "象限。off 表示关闭，penalty 表示软扣分，hard 表示硬拒绝同象限候选。"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))
    estimates_ = estimate(parse_args())
    print_estimates(estimates_)
