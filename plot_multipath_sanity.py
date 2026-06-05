"""
Multipath Sanity Check — Strongest-Path Visualization
======================================================
Loads ue_multipath_data.npz, extracts the globally strongest path per frame
(via argmax on path gain magnitude), and plots 3 panels vs UE Y-coordinate:
  1. Max Path Gain (dB)       — with multipath scatter background
  2. Time of Arrival (s)      — with theoretical LoS reference
  3. AoA Azimuth (degrees)    — with theoretical LoS reference

A vertical dashed line marks the geometric LoS → NLoS boundary (Y ≈ 26.7).
"""

import numpy as np
import matplotlib.pyplot as plt

NPZ_PATH = "ue_multipath_data.npz"
OUTPUT_PNG = "multipath_sanity_check.png"
LOS_BLOCKAGE_Y = 26.7   # Geometric boundary: Building B (south face y=10) blocks BS→UE ray
BS_POS = np.array([40.0, -40.0, 15.0])
C = 299792458.0
SCATTER_MAX_PER_FRAME = 80   # downsample multipath scatter to keep plot readable


def load_data(path):
    data = np.load(path, allow_pickle=True)
    print(f"Loaded keys: {list(data.keys())}")
    return (data["ue_positions"], data["taus"], data["phi_rs"], data["a_abss"])


def extract_strongest_and_scatter(ue_positions, taus, phi_rs, a_abss):
    """Per frame: pick the strongest path, and collect a downsampled scatter pool."""
    num_frames = len(ue_positions)
    y_coords = ue_positions[:, 1]

    max_gain_db = np.full(num_frames, np.nan)
    max_tau_s   = np.full(num_frames, np.nan)
    max_phi_deg = np.full(num_frames, np.nan)

    # Scatter buffers (all-path background)
    sc_y, sc_gain, sc_tau, sc_phi = [], [], [], []

    rng = np.random.default_rng(42)

    for i in range(num_frames):
        a_frame = np.asarray(a_abss[i])
        if len(a_frame) == 0:
            continue

        tau_arr = np.asarray(taus[i])
        phi_arr = np.asarray(phi_rs[i])

        # --- Strongest path (argmax) ---
        best_idx = np.argmax(a_frame)
        gain_lin = a_frame[best_idx]
        max_gain_db[i] = 20.0 * np.log10(gain_lin) if gain_lin > 0 else np.nan
        max_tau_s[i]   = tau_arr[best_idx]

        phi_deg = np.degrees(phi_arr[best_idx])
        phi_deg = (phi_deg + 180.0) % 360.0 - 180.0
        max_phi_deg[i] = phi_deg

        # --- Scatter: downsample this frame's paths ---
        n_paths = len(a_frame)
        n_sample = min(SCATTER_MAX_PER_FRAME, n_paths)
        if n_sample > 0:
            idx_sample = rng.choice(n_paths, n_sample, replace=False)
            gains_sample = 20.0 * np.log10(np.maximum(a_frame[idx_sample], 1e-15))
            phi_sample = np.degrees(phi_arr[idx_sample])
            phi_sample = (phi_sample + 180.0) % 360.0 - 180.0

            sc_y.extend([y_coords[i]] * n_sample)
            sc_gain.extend(gains_sample)
            sc_tau.extend(tau_arr[idx_sample])
            sc_phi.extend(phi_sample)

    return (y_coords, max_gain_db, max_tau_s, max_phi_deg,
            np.array(sc_y), np.array(sc_gain), np.array(sc_tau), np.array(sc_phi))


def theoretical_los_reference(ue_positions):
    """Compute pure-geometry LoS ToA and AoA (as if the building didn't exist)."""
    y = ue_positions[:, 1]
    dist = np.linalg.norm(ue_positions - BS_POS, axis=1)
    tau_ref = dist / C
    # AoA at UE: direction from UE → BS
    phi_ref = np.degrees(np.arctan2(BS_POS[1] - ue_positions[:, 1],
                                    BS_POS[0] - ue_positions[:, 0]))
    phi_ref = (phi_ref + 180.0) % 360.0 - 180.0
    return tau_ref, phi_ref


def plot_sanity_check(y, gain_db, tau_s, phi_deg,
                      sc_y, sc_gain, sc_tau, sc_phi,
                      tau_ref, phi_ref):
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # ── Panel 1: Path Gain ─────────────────────────────────────────────
    axs[0].scatter(sc_y, sc_gain, color="lightgray", s=6, alpha=0.35,
                   label="Multipath components (sampled)")
    axs[0].plot(y, gain_db, "b-o", markersize=4, linewidth=1.8,
                label="Strongest path (argmax)")
    axs[0].axvline(x=LOS_BLOCKAGE_Y, color="red", linestyle="--", alpha=0.8,
                   label=f"LoS → NLoS Boundary (Y≈{LOS_BLOCKAGE_Y})")
    axs[0].set_ylabel("Path Gain (dB)")
    axs[0].set_title("Multipath Sanity Check — Strongest Path vs UE Y-Coordinate")
    axs[0].legend(loc="lower left", fontsize=8, ncol=2)
    axs[0].grid(True, alpha=0.3)

    # ── Panel 2: Time of Arrival ───────────────────────────────────────
    axs[1].scatter(sc_y, sc_tau, color="lightgray", s=6, alpha=0.35)
    axs[1].plot(y, tau_ref, "k--", linewidth=1.5, alpha=0.8,
                label="Theoretical LoS ToA (geometry)")
    axs[1].plot(y, tau_s, "g-o", markersize=4, linewidth=1.8,
                label="Strongest path ToA (argmax)")
    axs[1].axvline(x=LOS_BLOCKAGE_Y, color="red", linestyle="--", alpha=0.8)
    axs[1].set_ylabel("Time of Arrival (s)")
    axs[1].legend(loc="upper left", fontsize=8)
    axs[1].grid(True, alpha=0.3)

    # ── Panel 3: AoA Azimuth ───────────────────────────────────────────
    axs[2].scatter(sc_y, sc_phi, color="lightgray", s=6, alpha=0.35)
    axs[2].plot(y, phi_ref, "k--", linewidth=1.5, alpha=0.8,
                label="Theoretical LoS AoA (geometry)")
    axs[2].plot(y, phi_deg, "r-o", markersize=4, linewidth=1.8,
                label="Strongest path AoA (argmax)")
    axs[2].axvline(x=LOS_BLOCKAGE_Y, color="red", linestyle="--", alpha=0.8)
    axs[2].set_ylabel("AoA Azimuth (deg)")
    axs[2].set_xlabel("UE Y-Coordinate (m)")
    axs[2].legend(loc="upper left", fontsize=8)
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    print(f"Plot saved to {OUTPUT_PNG}")


def main():
    print(f"Loading {NPZ_PATH} …")
    ue_positions, taus, phi_rs, a_abss = load_data(NPZ_PATH)
    print(f"Frames: {len(ue_positions)}, "
          f"UE Y range: [{ue_positions[:, 1].min():.1f}, {ue_positions[:, 1].max():.1f}]")

    (y, gain_db, tau_s, phi_deg,
     sc_y, sc_gain, sc_tau, sc_phi) = extract_strongest_and_scatter(
        ue_positions, taus, phi_rs, a_abss
    )

    tau_ref, phi_ref = theoretical_los_reference(ue_positions)

    n_valid = np.sum(~np.isnan(gain_db))
    print(f"Valid frames: {n_valid} / {len(ue_positions)}, "
          f"scatter points: {len(sc_y)}")

    plot_sanity_check(y, gain_db, tau_s, phi_deg,
                      sc_y, sc_gain, sc_tau, sc_phi,
                      tau_ref, phi_ref)


if __name__ == "__main__":
    main()
