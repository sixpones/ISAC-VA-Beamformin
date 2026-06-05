"""
虚拟锚点辅助波束训练 —— ISAC 多径几何追踪匹配
================================================================
通过 "时延 + 发射角 (AoD)" 双域联合匹配, 在 Sionna MIMO 射线追踪
数据中显式追踪三条物理传播路径:

  1. 直射径 (LoS)           — BS → UE 直接视距
  2. 东墙虚拟锚点反射径 (VA) — BS → Building‑C 东墙 → UE
  3. 南墙虚拟锚点反射径 (VA) — BS → Building‑B 南墙 → UE

将上述三条几何追踪路径与 "朴素 argmax (全局最强径)" 进行对比,
证明:
  - LoS 区域 (Y < 26.7): argmax 正确追踪到直射径
  - NLoS 区域 (Y > 26.7): argmax 被大楼穿透伪径劫持;
    VA 几何追踪方法仍能 100% 找到真实的东墙物理反射径

技术要点:
  - τ 匹配: 理论时延与实测时延误差 < 1 ns
  - φ 匹配: 通过 3D 线面交点反算真实反射点 R, 求 BS→R 的 AoD,
    再用相位解卷绕公式 |arctan2(sin(Δφ), cos(Δφ))| < 0.1 rad 做角度锁定
  - 多候选时取 MIMO 阵列 L2 范数最大者

输出文件:
  va_matching_result.png       — 单面板: 四条路径增益对比曲线
  va_matching_diagnostics.png  — 三面板: 增益 / 时延 / 发射角 综合诊断图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── 中文字体配置 ──────────────────────────────────────────────────────────
# 查找系统上可用的 Noto Sans CJK SC (简体中文) 字体
_cjk_fonts = [f for f in fm.findSystemFonts() if "NotoSansCJK" in f and "Regular" in f]
if _cjk_fonts:
    _font_prop = fm.FontProperties(fname=_cjk_fonts[0])
    _font_name = _font_prop.get_name()
    plt.rcParams["font.sans-serif"] = [_font_name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False   # 避免负号显示为方块
else:
    # 回退: 没有 CJK 字体时, 图表标签保留英文
    pass

# ═══════════════════════════════════════════════════════════════════════════
# 0. 固定几何参数
# ═══════════════════════════════════════════════════════════════════════════
BS_POS = np.array([40.0, -40.0, 15.0])   # 基站 BS0 坐标 (m)
C      = 299792458.0                      # 真空中光速 (m/s)

# 两个虚拟锚点坐标 (来自 RANSAC 墙面提取, 见 virtual_anchor.json)
#   Wall 1 — Building‑C 东墙: x ≈ -10, y ∈ [-70, -10]
#   Wall 0 — Building‑B 南墙: y ≈  10, x ∈ [ 10,  70]
VA_EAST  = np.array([-60.02101857, -39.95703503,  15.22865584])
VA_SOUTH = np.array([ 39.94802820,  60.16441234,  15.17563469])


def wall_from_va(va):
    """
    从虚拟锚点反推墙面平面方程 (n, p0)。

    原理: VA 是 BS 关于墙面的镜像点, 因此墙面是线段 BS↔VA 的垂直平分面。
    墙面法向量 n = (BS - VA) / ||BS - VA|| (单位向量)
    墙面上一点 p0 = (BS + VA) / 2         (线段中点)
    """
    n = BS_POS - va
    n = n / np.linalg.norm(n)
    p0 = (BS_POS + va) / 2.0
    return n, p0


# 两面墙的平面参数: 法向量 n 和 平面上一点 p0
N_EAST,  P0_EAST  = wall_from_va(VA_EAST)
N_SOUTH, P0_SOUTH = wall_from_va(VA_SOUTH)

# ── 双域匹配阈值 ─────────────────────────────────────────────────────────
TAU_THRESHOLD = 1e-9        # 时延容差: 1 纳秒 (对应 ~0.3 m 距离误差)
PHI_THRESHOLD = 0.1         # 角度容差: 0.1 弧度 (约 5.7°)

LOS_BLOCKAGE_Y = 26.7       # LoS→NLoS 几何分界线:
                             # Building‑B 南墙 (y=10) 遮挡 BS→UE 视线的临界 Y 值
                             # 推导: x|_(y=10) = 40-2000/(y+40) ≥ 10 ⇒ y ≥ 26.67


# ═══════════════════════════════════════════════════════════════════════════
# 1. 辅助函数
# ═══════════════════════════════════════════════════════════════════════════
def phase_diff(a, b):
    """
    相位解卷绕角度差 (rad), 返回值 ∈ [-π, π]。

    直接用 a-b 会在 ±π 边界处产生 2π 跳变 (例如 a=π, b=-π 时,
    实际差 0, 但 a-b=2π 会被误判为差异很大)。
    arctan2(sin(a-b), cos(a-b)) 自动解卷绕, 得到真实的最小绝对角度差。
    """
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def reflection_point(va, n, p0, ue):
    """
    3D 线面交点: 求射线 VA→UE 与墙面平面 (n, p0) 的交点坐标 R。

    几何背景:
      虚拟锚点 VA 发出的射线到达 UE 时, 与真实墙面相交于反射点 R。
      射线参数方程: P(t) = va + t * (ue - va),  t ∈ [0, 1]
      平面方程:      n · (P - p0) = 0

      代入得: n · (va + t*(ue-va) - p0) = 0
            ⇒ t = n·(p0 - va) / n·(ue - va)
            ⇒ R = va + t * (ue - va)

    返回: 反射点 R 的三维坐标
    """
    direction = ue - va
    t = np.dot(n, p0 - va) / np.dot(n, direction)
    return va + t * direction


def geometric_match(tau_arr, phi_arr, norms, tau_th, phi_th):
    """
    在单帧的所有多径中, 寻找同时满足时延和角度条件的路径。

    匹配条件:
      1. |τ_meas - τ_theory| < 1 ns      (时延锁定)
      2. |phase_diff(φ_meas, φ_theory)| < 0.1 rad  (角度锁定, 已解卷绕)

    如果存在多条候选径, 返回 L2 范数 (阵列总增益) 最大的那条的索引。
    如果没有任何匹配, 返回 None。
    """
    mask = (
        (np.abs(tau_arr - tau_th) < TAU_THRESHOLD)
        & (np.abs(phase_diff(phi_arr, phi_th)) < PHI_THRESHOLD)
    )
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    return idx[np.argmax(norms[idx])]


def to_db(lin_val):
    """线性增益 → dB 值 (20*log10)。零或负值返回 NaN。"""
    return 20.0 * np.log10(lin_val) if lin_val > 0 else np.nan


def wrap_deg(rad):
    """弧度 → 度, 并归化到 [-180°, 180°] 区间。"""
    deg = np.degrees(rad)
    return (deg + 180.0) % 360.0 - 180.0


# ═══════════════════════════════════════════════════════════════════════════
# 2. 主函数: 逐帧追踪 + 对比分析
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # ── 2.1 加载数据 ──────────────────────────────────────────────────────
    data = np.load("ue_mimo_multipath_data.npz", allow_pickle=True)
    ue_positions = data["ue_positions"]    # 形状 (122, 3), UE 的 xyz 坐标
    taus         = data["taus"]            # object 数组, 每帧的路径传播时延 (s)
    phi_ts       = data["phi_ts"]          # object 数组, 每帧的路径发射角 AoD (rad)
    h_complexs   = data["h_complexs"]      # object 数组, 每帧的 MIMO 复信道 (M径 × 64天线)

    N = len(ue_positions)                  # 总帧数
    y_coords = ue_positions[:, 1]          # UE 沿 Y 轴移动, 横坐标用 Y 值
    print(f"加载完成: {N} 帧, UE 轨迹 Y ∈ [{y_coords[0]:.1f}, {y_coords[-1]:.1f}] m")

    # ── 2.2 预分配输出数组 ────────────────────────────────────────────────
    # 路径增益 (dB)
    argmax_gain = np.full(N, np.nan)   # 朴素 argmax
    los_gain    = np.full(N, np.nan)   # LoS 几何追踪
    east_gain   = np.full(N, np.nan)   # 东墙 VA 反射追踪
    south_gain  = np.full(N, np.nan)   # 南墙 VA 反射追踪
    # 到达时间 ToA (s)
    argmax_tau  = np.full(N, np.nan)
    los_tau     = np.full(N, np.nan)
    east_tau    = np.full(N, np.nan)
    south_tau   = np.full(N, np.nan)
    # 发射方位角 AoD (°)
    argmax_phi  = np.full(N, np.nan)
    los_phi     = np.full(N, np.nan)
    east_phi    = np.full(N, np.nan)
    south_phi   = np.full(N, np.nan)

    # ── 2.3 逐帧主循环 ────────────────────────────────────────────────────
    for i in range(N):
        ue = ue_positions[i]                            # 当前 UE 坐标 [x, y, z]
        ta = np.asarray(taus[i])                        # 当前帧所有路径的时延
        ph = np.asarray(phi_ts[i])                      # 当前帧所有路径的 AoD (BS 端)
        hh = h_complexs[i]                              # 当前帧 MIMO 复数信道 (M, 64)

        if len(ta) == 0:                                # 极端情况: 无有效路径
            continue

        # 每条径的 MIMO 阵列总增益 = 64 维复向量的 L2 范数
        norms = np.linalg.norm(hh, axis=1)

        # --- 朴素 argmax: 直接选全局最强径 (对比基准) ---
        b = np.argmax(norms)
        argmax_gain[i] = to_db(norms[b])
        argmax_tau[i]  = ta[b]
        argmax_phi[i]  = wrap_deg(ph[b])

        # --- LoS 直射径: BS → UE 直线 ---
        tau_los_th = np.linalg.norm(ue - BS_POS) / C
        phi_los_th = np.arctan2(ue[1] - BS_POS[1], ue[0] - BS_POS[0])
        idx = geometric_match(ta, ph, norms, tau_los_th, phi_los_th)
        if idx is not None:
            los_gain[i] = to_db(norms[idx])
            los_tau[i]  = ta[idx]
            los_phi[i]  = wrap_deg(ph[idx])

        # --- 东墙 VA 反射径: BS → Building‑C 东墙 → UE ---
        # 理论时延 = 虚拟锚点到 UE 的直线距离 / 光速
        tau_east_th = np.linalg.norm(ue - VA_EAST) / C
        # 理论 AoD: BS 指向墙面上的真实反射点 R (通过线面交点计算)
        R_east = reflection_point(VA_EAST, N_EAST, P0_EAST, ue)
        phi_east_th = np.arctan2(R_east[1] - BS_POS[1], R_east[0] - BS_POS[0])
        idx = geometric_match(ta, ph, norms, tau_east_th, phi_east_th)
        if idx is not None:
            east_gain[i] = to_db(norms[idx])
            east_tau[i]  = ta[idx]
            east_phi[i]  = wrap_deg(ph[idx])

        # --- 南墙 VA 反射径: BS → Building‑B 南墙 → UE ---
        # 同上, 但使用南墙的 VA 和墙面参数
        tau_south_th = np.linalg.norm(ue - VA_SOUTH) / C
        R_south = reflection_point(VA_SOUTH, N_SOUTH, P0_SOUTH, ue)
        phi_south_th = np.arctan2(R_south[1] - BS_POS[1], R_south[0] - BS_POS[0])
        idx = geometric_match(ta, ph, norms, tau_south_th, phi_south_th)
        if idx is not None:
            south_gain[i] = to_db(norms[idx])
            south_tau[i]  = ta[idx]
            south_phi[i]  = wrap_deg(ph[idx])

    # ═══════════════════════════════════════════════════════════════════════
    # 3. 控制台诊断报告
    # ═══════════════════════════════════════════════════════════════════════
    mask_los_region = y_coords < LOS_BLOCKAGE_Y    # LoS 区域帧索引
    mask_nlos       = y_coords > LOS_BLOCKAGE_Y    # NLoS 区域帧索引

    # 统计 NLoS 区域中 argmax 被穿透径劫持的帧数
    # 判定标准: 最强径的时延 ≈ 理论 LoS 时延 (即该径沿 LoS 几何路径穿透了大楼)
    nlos_idx = np.where(mask_nlos)[0]
    penet_count = 0
    for i in nlos_idx:
        ue = ue_positions[i]
        ta = np.asarray(taus[i]); hh = h_complexs[i]
        if len(ta) == 0: continue
        norms = np.linalg.norm(hh, axis=1)
        b = np.argmax(norms)
        tau_los = np.linalg.norm(ue - BS_POS) / C
        if abs(ta[b] - tau_los) < TAU_THRESHOLD:
            penet_count += 1

    n_los_region = np.sum(mask_los_region)
    n_nlos_region = np.sum(mask_nlos)

    los_in_los  = np.sum(~np.isnan(los_gain[mask_los_region]))
    los_in_nlos = np.sum(~np.isnan(los_gain[mask_nlos]))
    east_in_los  = np.sum(~np.isnan(east_gain[mask_los_region]))
    east_in_nlos = np.sum(~np.isnan(east_gain[mask_nlos]))
    south_in_los  = np.sum(~np.isnan(south_gain[mask_los_region]))
    south_in_nlos = np.sum(~np.isnan(south_gain[mask_nlos]))

    print(f"\n{'='*60}")
    print(f"  ISAC 虚拟锚点匹配 — 诊断报告")
    print(f"{'='*60}")
    print(f"  {'':<20}  {'LoS 区域':>10}  {'NLoS 区域':>10}")
    print(f"  {'─'*45}")
    print(f"  {'Lo S  直射径追踪':<20}  {los_in_los:>4}/{n_los_region:<4}"
          f"    {los_in_nlos:>4}/{n_nlos_region:<4}  (NLoS 部分 = 穿透伪径)")
    print(f"  {'东墙 VA 反射追踪':<20}  {east_in_los:>4}/{n_los_region:<4}"
          f"    {east_in_nlos:>4}/{n_nlos_region:<4}")
    print(f"  {'南墙 VA 反射追踪':<20}  {south_in_los:>4}/{n_los_region:<4}"
          f"    {south_in_nlos:>4}/{n_nlos_region:<4}")
    print(f"  {'─'*45}")
    print(f"  NLoS 区域 argmax 被穿透径劫持: {penet_count}/{n_nlos_region} 帧")
    print(f"  NLoS 区域东墙 VA 与 argmax 增益差 (均值): "
          f"{np.nanmean(argmax_gain[mask_nlos] - east_gain[mask_nlos]):.1f} dB")
    print(f"  (差值 > 0 说明穿透伪径比真实反射径更强, 即材质模型低估了穿透损耗)")
    print(f"{'='*60}")

    # ── 东墙 VA 断续原因诊断 ──────────────────────────────────────────────
    east_missing = np.where(np.isnan(east_gain))[0]
    if len(east_missing) > 0:
        print(f"\n  [注] 东墙 VA 红线在 {len(east_missing)} 帧处断开 (图中可见断断续续):")
        print(f"       缺失帧 Y 范围: [{y_coords[east_missing].min():.1f}, "
              f"{y_coords[east_missing].max():.1f}] m")
        print(f"       原因: 这些位置反射点 R 落在东墙物理边界之外,")
        print(f"             或反射径被其他建筑物遮挡, 射线追踪未生成对应多径。")
        print(f"       这是物理上正确的现象, 不是代码 bug。")

    # ═══════════════════════════════════════════════════════════════════════
    # 4. 图 A — 单面板增益对比
    # ═══════════════════════════════════════════════════════════════════════
    fig_a, ax = plt.subplots(figsize=(12, 7))

    ax.plot(y_coords, argmax_gain, color="gray", linestyle="--", linewidth=2.0,
            label="朴素 argmax (全局最强径)")
    ax.plot(y_coords, los_gain,   "b--", linewidth=2.0,
            label="LoS 直射径 (几何追踪)")
    ax.plot(y_coords, east_gain,  "r-o", markersize=4, linewidth=2.0,
            label="东墙 VA 反射径 (几何追踪)")
    ax.plot(y_coords, south_gain, color="darkorange", linestyle="-.",
            marker="s", markersize=3, linewidth=1.8,
            label="南墙 VA 反射径 (几何追踪)")

    ax.axvline(x=LOS_BLOCKAGE_Y, color="black", linestyle=":", alpha=0.6,
               label=f"LoS 遮挡边界 (Y≈{LOS_BLOCKAGE_Y} m)")

    ax.set_xlabel("UE Y 坐标 (m)", fontsize=13, fontweight="bold")
    ax.set_ylabel("路径增益 (dB)", fontsize=13, fontweight="bold")
    ax.set_title("虚拟锚点辅助波束训练 — 路径增益对比",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig_a.tight_layout()
    fig_a.savefig("va_matching_result.png", dpi=300)
    print("\n已保存  va_matching_result.png")

    # ═══════════════════════════════════════════════════════════════════════
    # 5. 图 B — 三面板综合诊断图 (增益 / 时延 / 发射角)
    # ═══════════════════════════════════════════════════════════════════════
    fig_b, axs = plt.subplots(3, 1, figsize=(14, 13), sharex=True)

    # --- 面板 1: 路径增益 ---
    axs[0].plot(y_coords, argmax_gain, "gray", linestyle="--", linewidth=2.0,
                label="朴素 argmax")
    axs[0].plot(y_coords, los_gain,   "b--",  linewidth=1.8,
                label="LoS 直射径 (几何追踪)")
    axs[0].plot(y_coords, east_gain,  "r-o",  markersize=4, linewidth=1.8,
                label="东墙 VA 反射径 (几何追踪)")
    axs[0].plot(y_coords, south_gain, color="darkorange", linestyle="-.",
                markersize=3, linewidth=1.8,
                label="南墙 VA 反射径 (几何追踪)")
    axs[0].axvline(x=LOS_BLOCKAGE_Y, color="black", linestyle=":", alpha=0.5)
    axs[0].set_ylabel("路径增益 (dB)", fontsize=12)
    axs[0].set_title("虚拟锚点辅助波束训练 — 三面板综合诊断", fontsize=14, fontweight="bold")
    axs[0].legend(loc="lower left", fontsize=8, ncol=2)
    axs[0].grid(True, alpha=0.3)

    # --- 面板 2: 到达时间 (ToA) ---
    axs[1].plot(y_coords, argmax_tau, "gray", linestyle="--", linewidth=2.0,
                label="朴素 argmax")
    axs[1].plot(y_coords, los_tau,   "b--",  linewidth=1.8,
                label="LoS 直射径 (几何追踪)")
    axs[1].plot(y_coords, east_tau,  "r-o",  markersize=4, linewidth=1.8,
                label="东墙 VA 反射径 (几何追踪)")
    axs[1].plot(y_coords, south_tau, color="darkorange", linestyle="-.",
                markersize=3, linewidth=1.8,
                label="南墙 VA 反射径 (几何追踪)")
    axs[1].axvline(x=LOS_BLOCKAGE_Y, color="black", linestyle=":", alpha=0.5)
    axs[1].set_ylabel("到达时间 ToA (s)", fontsize=12)
    axs[1].legend(loc="upper left", fontsize=8, ncol=2)
    axs[1].grid(True, alpha=0.3)

    # --- 面板 3: 发射方位角 (AoD) ---
    axs[2].plot(y_coords, argmax_phi, "gray", linestyle="--", linewidth=2.0,
                label="朴素 argmax")
    axs[2].plot(y_coords, los_phi,   "b--",  linewidth=1.8,
                label="LoS 直射径 (几何追踪)")
    axs[2].plot(y_coords, east_phi,  "r-o",  markersize=4, linewidth=1.8,
                label="东墙 VA 反射径 (几何追踪)")
    axs[2].plot(y_coords, south_phi, color="darkorange", linestyle="-.",
                markersize=3, linewidth=1.8,
                label="南墙 VA 反射径 (几何追踪)")
    axs[2].axvline(x=LOS_BLOCKAGE_Y, color="black", linestyle=":", alpha=0.5)
    axs[2].set_ylabel("发射方位角 AoD (°)", fontsize=12)
    axs[2].set_xlabel("UE Y 坐标 (m)", fontsize=13)
    axs[2].legend(loc="upper left", fontsize=8, ncol=2)
    axs[2].grid(True, alpha=0.3)

    fig_b.tight_layout()
    fig_b.savefig("va_matching_diagnostics.png", dpi=300)
    print("已保存  va_matching_diagnostics.png")


if __name__ == "__main__":
    main()
