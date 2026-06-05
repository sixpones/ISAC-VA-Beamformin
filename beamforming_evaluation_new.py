"""
ISAC 波束追踪方案性能对比 — 频谱效率 & 波束训练开销 (5 方案版)
==============================================================
基于 Sionna 8×8 MIMO 射线追踪数据集 (ue_mimo_multipath_data.npz),
对五种波束追踪方案进行频谱效率 (SE) 评估:

  方案 A — MRT 穷举搜索 / 理想上限 (MRT Ideal Bound)
        逐径计算 L2 范数, 取全局最强径, MRT 发射。
        开销: 64 次波束扫描。

  方案 B — 3GPP 2D-DFT 码本穷举搜索 (DFT Codebook Exhaustive)
        计算全径合成等效信道 H_total, 遍历 64 个 DFT 波束。
        开销: 64 次波束扫描。

  方案 C — 传统 LoS 直射径追踪 (LoS Tracking, Baseline)
        时延+发射角双域匹配 LoS 理论值, MRT 发射。
        开销: 1 次预测发波。缺陷: 宽松容差下可能匹配到穿透伪径。

  方案 D — VA 辅助反射追踪 (VA-Only, 消融分析)
        3D 线面交点反算反射点, 双域匹配东墙 VA 反射径, MRT 发射。
        引入 ISAC 零阶保持记忆。开销: 1 次。
        用途: 证明反射径的独立贡献。

  方案 E — VA 辅助自适应波束选择 (VA-Assisted Adaptive, 本文完整方案)
        每帧先 LoS 双域匹配, 若 LoS 有效且 RSRP > VA 反射径, 则用 LoS;
        否则退回 VA 反射径。VA 侧带零阶保持记忆。
        开销: 1 次。核心创新: VA 几何约束可鉴别穿透伪径。

输出: isac_beamforming_eval_5schemes.png (1×2 IEEE 风格面板, DPI=300)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ═══════════════════════════════════════════════════════════════════════════
# 0. 全局物理参数
# ═══════════════════════════════════════════════════════════════════════════
C          = 299792458.0                     # 真空中光速 (m/s)
F_C        = 28e9                            # 载波频率 (Hz)
SNR_SCALE  = 1e11                            # 接收信噪比缩放因子
BS_POS     = np.array([40.0, -40.0, 15.0])   # 基站 BS0 坐标
VA_POS     = np.array([-60.024687, -39.953964, 15.291423])  # 东墙虚拟锚点

# ── 双域匹配容差 ─────────────────────────────────────────────────────────
TOL_TAU = 50e-9       # 时延容差: 50 ns
TOL_PHI = 0.6         # 角度容差: 0.6 rad

LOS_BLOCKAGE_Y = 26.7  # LoS 遮挡边界

# ── 中文字体 ──────────────────────────────────────────────────────────────
_cjk_fonts = [f for f in fm.findSystemFonts()
              if "NotoSansCJK" in f and "Regular" in f]
if _cjk_fonts:
    _font_name = fm.FontProperties(fname=_cjk_fonts[0]).get_name()
    plt.rcParams["font.sans-serif"] = [_font_name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


# ═══════════════════════════════════════════════════════════════════════════
# 1. 辅助函数
# ═══════════════════════════════════════════════════════════════════════════
def phase_diff(a, b):
    """相位解卷绕角度差 (rad), 返回值 ∈ [-π, π]"""
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def generate_2d_dft_codebook(Ny=8, Nz=8):
    """
    生成 8×8 UPA 的 2D-DFT 波束码本 (符合 3GPP 5G NR Type I 单面板码本结构)。

    每个波束权重为 1D DFT 向量的 Kronecker 积:
        w_{ky,kz} = a_z(kz) ⊗ a_y(ky)  ∈ C^{64}

    返回: W ∈ C^{64×64}, 每列对应一个归一化波束权重向量。
    """
    n_y = np.arange(Ny)
    n_z = np.arange(Nz)

    a_y = np.exp(1j * 2.0 * np.pi * np.outer(n_y, np.arange(Ny)) / Ny)
    a_z = np.exp(1j * 2.0 * np.pi * np.outer(n_z, np.arange(Nz)) / Nz)

    W = np.zeros((Ny * Nz, Ny * Nz), dtype=np.complex128)
    col = 0
    for kz in range(Nz):
        for ky in range(Ny):
            W[:, col] = np.kron(a_z[:, kz], a_y[:, ky])
            col += 1

    W /= np.sqrt(Ny * Nz)
    return W


def compute_h_total(hh, ta):
    """
    计算全径合成等效信道 H_total ∈ C^{64}。
        H_total = Σ_k h_k · exp(-j·2π·fc·τ_k)
    """
    K = len(ta)
    if K == 0:
        return np.zeros(64, dtype=np.complex128)
    phase_shift = np.exp(-1j * 2.0 * np.pi * F_C * ta)
    return np.sum(hh * phase_shift[:, np.newaxis], axis=0)


def compute_mrt_rsrp(h_k, tau_k):
    """
    对给定多径向量 h_k (64,) 执行 MRT 波束赋形, 返回等效 RSRP。
    w = h_k^H / ||h_k||,  h_eq = h_k · w · exp(-j 2π fc τ),  RSRP = |h_eq|² × SNR_SCALE
    """
    norm_h = np.linalg.norm(h_k)
    if norm_h < 1e-15:
        return 0.0
    w = np.conj(h_k) / norm_h
    h_eq = np.dot(h_k, w) * np.exp(-1j * 2.0 * np.pi * F_C * tau_k)
    return np.abs(h_eq) ** 2 * SNR_SCALE


def se_from_rsrp(rsrp):
    """Shannon 频谱效率: SE = log₂(1 + RSRP)  (bps/Hz)"""
    return np.log2(1.0 + rsrp)


# ═══════════════════════════════════════════════════════════════════════════
# 2. 主函数
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # ── 2.1 加载 MIMO 射线追踪数据 ────────────────────────────────────────
    data = np.load("ue_mimo_multipath_data.npz", allow_pickle=True)
    ue_positions = data["ue_positions"]      # (N, 3)
    taus         = data["taus"]              # object 数组, 时延 (s)
    phi_ts       = data["phi_ts"]            # object 数组, 发射方位角 AoD (rad)
    h_complexs   = data["h_complexs"]        # object 数组, (K_i, 64) 复信道

    N = len(ue_positions)
    y_coords = ue_positions[:, 1]
    print(f"数据加载完成: {N} 帧, UE Y ∈ [{y_coords[0]:.1f}, {y_coords[-1]:.1f}] m")

    # ── 2.2 生成 2D-DFT 码本 ──────────────────────────────────────────────
    W_codebook = generate_2d_dft_codebook(Ny=8, Nz=8)  # (64, 64)

    # ── 2.3 预分配五种方案的 SE 数组 ──────────────────────────────────────
    se_exhaustive = np.full(N, np.nan)      # A: MRT 理想上限
    se_dft        = np.full(N, np.nan)      # B: DFT 码本穷举
    se_los        = np.full(N, np.nan)      # C: LoS 追踪 (传统基线)
    se_va         = np.full(N, np.nan)      # D: VA 纯反射 (消融分析)
    se_adaptive   = np.full(N, np.nan)      # E: VA 自适应波束 (完整方案)

    # ISAC 记忆: 方案 D/E 各自的零阶保持
    last_valid_rsrp_va = 0.0
    last_valid_rsrp_adaptive = 0.0

    # ── 2.4 逐帧处理 ──────────────────────────────────────────────────────
    for i in range(N):
        ue   = ue_positions[i]
        ta   = np.asarray(taus[i])
        ph   = np.asarray(phi_ts[i])
        hh   = h_complexs[i]                          # (K, 64)

        if len(ta) == 0:
            continue

        norms = np.linalg.norm(hh, axis=1)            # 每条径的 L2 范数

        # ── 方案 A: MRT 穷举搜索 (理想上限) ───────────────────────────────
        best_a = np.argmax(norms)
        rsrp_a = compute_mrt_rsrp(hh[best_a], ta[best_a])
        se_exhaustive[i] = se_from_rsrp(rsrp_a)

        # ── 方案 B: 3GPP 2D-DFT 码本穷举搜索 (3GPP 基线) ──────────────────
        H_total = compute_h_total(hh, ta)
        beam_powers = np.abs(W_codebook.conj().T @ H_total) ** 2
        rsrp_dft = np.max(beam_powers) * SNR_SCALE
        se_dft[i] = se_from_rsrp(rsrp_dft)

        # ── 公共: LoS 双域匹配 (方案 C / E 共用) ──────────────────────────
        tau_los_th = np.linalg.norm(ue - BS_POS) / C
        phi_los_th = np.arctan2(ue[1] - BS_POS[1], ue[0] - BS_POS[0])

        mask_los = (
            (np.abs(ta - tau_los_th) < TOL_TAU)
            & (np.abs(phase_diff(ph, phi_los_th)) < TOL_PHI)
        )
        idx_los = np.where(mask_los)[0]
        los_matched = len(idx_los) > 0
        rsrp_los = 0.0
        if los_matched:
            best_los = idx_los[np.argmax(norms[idx_los])]
            rsrp_los = compute_mrt_rsrp(hh[best_los], ta[best_los])

        # ── 公共: VA 反射径双域匹配 (方案 D / E 共用) ────────────────────
        n_wall = BS_POS - VA_POS
        n_wall = n_wall / np.linalg.norm(n_wall)
        p0_wall = (BS_POS + VA_POS) / 2.0

        tau_va_th = np.linalg.norm(ue - VA_POS) / C
        direction = ue - VA_POS
        t_param = np.dot(n_wall, p0_wall - VA_POS) / np.dot(n_wall, direction)
        R = VA_POS + t_param * direction
        phi_va_th = np.arctan2(R[1] - BS_POS[1], R[0] - BS_POS[0])

        mask_va = (
            (np.abs(ta - tau_va_th) < TOL_TAU)
            & (np.abs(phase_diff(ph, phi_va_th)) < TOL_PHI)
        )
        idx_va = np.where(mask_va)[0]
        va_matched = len(idx_va) > 0
        rsrp_va = 0.0
        if va_matched:
            best_va = idx_va[np.argmax(norms[idx_va])]
            rsrp_va = compute_mrt_rsrp(hh[best_va], ta[best_va])

        # ── 方案 C: LoS 直射径追踪 (传统基线) ──────────────────────────────
        if los_matched:
            se_los[i] = se_from_rsrp(rsrp_los)
        else:
            se_los[i] = 0.0

        # ── 方案 D: VA 纯反射 (消融分析) ──────────────────────────────────
        if va_matched:
            last_valid_rsrp_va = rsrp_va
            se_va[i] = se_from_rsrp(rsrp_va)
        else:
            se_va[i] = se_from_rsrp(last_valid_rsrp_va)

        # ── 方案 E: VA 辅助自适应波束选择 (本文完整方案) ───────────────────
        # 核心逻辑: LoS 匹配成功 且 其 RSRP 优于反射径 → 采用 LoS;
        #           否则退回 VA 反射径。
        # VA 几何约束可天然筛除穿透建筑物的"假 LoS 径"。
        if los_matched and rsrp_los > rsrp_va:
            # LoS 径真实有效 (VA 几何验证通过: LoS RSRP 不应弱于反射径)
            last_valid_rsrp_adaptive = rsrp_los
            se_adaptive[i] = se_from_rsrp(rsrp_los)
        elif va_matched:
            # LoS 不可用或被穿透伪径污染, 退回可靠的 VA 反射径
            last_valid_rsrp_adaptive = rsrp_va
            se_adaptive[i] = se_from_rsrp(rsrp_va)
        else:
            # 双径均失效, ISAC 零阶保持
            se_adaptive[i] = se_from_rsrp(last_valid_rsrp_adaptive)

    # ═══════════════════════════════════════════════════════════════════════
    # 3. 控制台统计输出
    # ═══════════════════════════════════════════════════════════════════════
    mask_los_region = y_coords < LOS_BLOCKAGE_Y
    mask_nlos       = y_coords > LOS_BLOCKAGE_Y

    print(f"\n{'='*75}")
    print(f"  波束追踪方案性能对比 — 频谱效率 (SE) 统计 (5 方案)")
    print(f"{'='*75}")
    print(f"  {'方案':<38} {'LoS 区域':>10}  {'NLoS 区域':>10}  {'全局均值':>10}")
    print(f"  {'─'*69}")
    for name, se_arr in [("A: MRT 穷举搜索 (理想上限)",       se_exhaustive),
                          ("B: 2D-DFT 码本穷举 (3GPP 基线)",  se_dft),
                          ("C: LoS 直射径追踪 (传统基线)",     se_los),
                          ("D: VA 纯反射 (消融分析)",          se_va),
                          ("E: VA 自适应波束 (完整方案)",      se_adaptive)]:
        print(f"  {name:<36}  {np.nanmean(se_arr[mask_los_region]):>8.3f}   "
              f"{np.nanmean(se_arr[mask_nlos]):>8.3f}   {np.nanmean(se_arr):>8.3f}")
    print(f"  {'─'*69}")
    print(f"  波束训练开销:  A=64次  B=64次  C=1次  D=1次  E=1次")
    print(f"{'='*75}\n")

    # ═══════════════════════════════════════════════════════════════════════
    # 4. 学术绘图 — 1×2 IEEE 风格面板
    # ═══════════════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5))

    COLOR_A = "black"
    COLOR_B = "#7B2D8B"     # 紫色 (DFT 码本)
    COLOR_C = "#2166AC"     # 深蓝 (LoS 追踪)
    COLOR_D = "#E69F00"     # 橙色 (VA 纯反射, 消融)
    COLOR_E = "#B2182B"     # 深红 (VA 自适应, 完整方案)

    # ── 子图 1: 频谱效率随 UE Y 坐标变化 ──────────────────────────────────
    ax1.plot(y_coords, se_exhaustive, color=COLOR_A, linestyle="-",
             linewidth=2.2, label="A: MRT Exhaustive (Ideal Bound)")
    ax1.plot(y_coords, se_dft,        color=COLOR_B, linestyle="--",
             linewidth=2.0, label="B: 2D-DFT Codebook (3GPP Baseline)")
    ax1.plot(y_coords, se_los,        color=COLOR_C, linestyle="-.",
             linewidth=2.0, label="C: LoS Tracking (Baseline)")
    ax1.plot(y_coords, se_va,         color=COLOR_D, linestyle=":",
             linewidth=2.0, label="D: VA-Only Reflection (Ablation)")
    ax1.plot(y_coords, se_adaptive,   color=COLOR_E, linestyle="-",
             linewidth=2.5, marker="o", markersize=3, markevery=5,
             label="E: VA-Assisted Adaptive (Proposed)")

    # LoS 遮挡竖线 + 文字标注
    ylim_top = ax1.get_ylim()[1]
    ax1.axvline(x=LOS_BLOCKAGE_Y, color="gray", linestyle=":", linewidth=2.0,
                alpha=0.8)
    ax1.annotate("LoS Blockage\n(Y = 26.7 m)",
                 xy=(LOS_BLOCKAGE_Y, ylim_top * 0.55),
                 xytext=(LOS_BLOCKAGE_Y + 4, ylim_top * 0.55),
                 fontsize=10, fontweight="bold", color="gray",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    ax1.set_xlabel("UE Y-Coordinate (m)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Spectral Efficiency (bps/Hz)", fontsize=13, fontweight="bold")
    ax1.set_title("(a) Spectral Efficiency vs. UE Position", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9,
               edgecolor="gray")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xlim(y_coords[0], y_coords[-1])

    # ── 子图 2: 波束训练开销柱状图 ────────────────────────────────────────
    methods_label = ["A: MRT\nExhaustive", "B: DFT\nCodebook",
                     "C: LoS\nTracking",   "D: VA-Only\n(Ablation)",
                     "E: VA-Adaptive\n(Proposed)"]
    overhead_vals = [64, 64, 1, 1, 1]
    bar_colors    = [COLOR_A, COLOR_B, COLOR_C, COLOR_D, COLOR_E]

    bars = ax2.bar(methods_label, overhead_vals, color=bar_colors,
                   alpha=0.85, edgecolor="black", linewidth=1.5, width=0.55)

    for bar, val in zip(bars, overhead_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2.0,
                 bar.get_height() + 1.5,
                 str(val),
                 ha="center", va="bottom",
                 fontsize=15, fontweight="bold")

    ax2.set_ylabel("Number of Beam Scans", fontsize=13, fontweight="bold")
    ax2.set_title("(b) Beam Training Overhead", fontsize=14, fontweight="bold")
    ax2.set_yscale("log")
    ax2.set_ylim(0.5, 200)
    ax2.grid(True, alpha=0.3, axis="y", which="both", linestyle="--")

    plt.tight_layout()
    plt.savefig("isac_beamforming_eval_5schemes.png", dpi=300, bbox_inches="tight")
    print("图片已保存: isac_beamforming_eval_5schemes.png")


if __name__ == "__main__":
    main()
