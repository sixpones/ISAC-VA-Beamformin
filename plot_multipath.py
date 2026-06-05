import numpy as np
import matplotlib.pyplot as plt

# ── Fixed geometry ──────────────────────────────────────────────────────
BS_POS = np.array([40.0, -40.0, 15.0])
VA_POS = np.array([-60.024687, -39.953964, 15.291423])
C = 299792458.0

# Precompute wall geometry for VA reflection-point AoD
n_wall = BS_POS - VA_POS
n_wall = n_wall / np.linalg.norm(n_wall)
p0_wall = (BS_POS + VA_POS) / 2.0


def phase_diff(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def track_geometric_paths(data):
    """Extract LoS and VA reflected paths via delay+angle dual-domain matching."""
    ue_positions = data["ue_positions"]
    taus = data["taus"]
    phi_ts = data["phi_ts"]
    h_complexs = data["h_complexs"]
    num_frames = len(ue_positions)

    los_gain = np.full(num_frames, np.nan)
    los_tau  = np.full(num_frames, np.nan)
    los_phi  = np.full(num_frames, np.nan)
    va_gain  = np.full(num_frames, np.nan)
    va_tau   = np.full(num_frames, np.nan)
    va_phi   = np.full(num_frames, np.nan)

    for i in range(num_frames):
        ue = ue_positions[i]
        tau_arr = np.asarray(taus[i])
        phi_arr = np.asarray(phi_ts[i])
        h_arr   = h_complexs[i]

        if len(tau_arr) == 0:
            continue

        # ── LoS path ────────────────────────────────────────────────────
        tau_los_theo = np.linalg.norm(ue - BS_POS) / C
        phi_los_theo = np.arctan2(ue[1] - BS_POS[1], ue[0] - BS_POS[0])

        mask_los = (
            (np.abs(tau_arr - tau_los_theo) < 1e-9)
            & (np.abs(phase_diff(phi_arr, phi_los_theo)) < 0.1)
        )
        idx_los = np.where(mask_los)[0]
        if len(idx_los) > 0:
            norms = np.linalg.norm(h_arr[idx_los], axis=1)
            best = idx_los[np.argmax(norms)]
            los_gain[i] = 20.0 * np.log10(norms[np.argmax(norms)])
            los_tau[i] = tau_arr[best]
            los_phi[i] = np.degrees(phi_arr[best])

        # ── East-wall VA reflected path ─────────────────────────────────
        tau_va_theo = np.linalg.norm(ue - VA_POS) / C
        direction = ue - VA_POS
        t = np.dot(n_wall, p0_wall - VA_POS) / np.dot(n_wall, direction)
        R = VA_POS + t * direction
        phi_va_theo = np.arctan2(R[1] - BS_POS[1], R[0] - BS_POS[0])

        mask_va = (
            (np.abs(tau_arr - tau_va_theo) < 1e-9)
            & (np.abs(phase_diff(phi_arr, phi_va_theo)) < 0.1)
        )
        idx_va = np.where(mask_va)[0]
        if len(idx_va) > 0:
            norms = np.linalg.norm(h_arr[idx_va], axis=1)
            best = idx_va[np.argmax(norms)]
            va_gain[i] = 20.0 * np.log10(norms[np.argmax(norms)])
            va_tau[i] = tau_arr[best]
            va_phi[i] = np.degrees(phi_arr[best])

    # Map AoD to [-180, 180]
    los_phi = (los_phi + 180.0) % 360.0 - 180.0
    va_phi  = (va_phi  + 180.0) % 360.0 - 180.0
    return los_gain, los_tau, los_phi, va_gain, va_tau, va_phi


def main():
    print("Loading ue_mimo_multipath_data.npz for raw dataset verification...")
    data = np.load("ue_mimo_multipath_data.npz", allow_pickle=True)

    ue_positions = data["ue_positions"]
    taus = data["taus"]
    phi_ts = data["phi_ts"]
    h_complexs = data["h_complexs"]

    num_frames = len(ue_positions)
    y_coords = ue_positions[:, 1]

    # ── Theoretical LoS references (no tracking, pure geometry) ─────────
    tau_los_ref = np.array([
        np.linalg.norm(ue_positions[i] - BS_POS) / C for i in range(num_frames)
    ])
    phi_los_ref = np.array([
        np.degrees(np.arctan2(
            ue_positions[i][1] - BS_POS[1],
            ue_positions[i][0] - BS_POS[0]
        )) for i in range(num_frames)
    ])
    phi_los_ref = (phi_los_ref + 180.0) % 360.0 - 180.0

    # ── Scatter data (all multipath components) ─────────────────────────
    all_y = []
    all_gains = []
    all_taus = []
    all_phis = []

    # ── argmax strongest path ───────────────────────────────────────────
    max_gain = np.full(num_frames, np.nan)
    max_tau = np.full(num_frames, np.nan)
    max_phi = np.full(num_frames, np.nan)

    for i in range(num_frames):
        y = y_coords[i]
        tau_arr = np.asarray(taus[i])
        phi_arr = np.asarray(phi_ts[i])
        h_arr = h_complexs[i]

        if len(tau_arr) == 0:
            continue

        norms = np.linalg.norm(h_arr, axis=1)
        gains_db = 20.0 * np.log10(norms + 1e-15)
        phi_deg = np.degrees(phi_arr)
        phi_deg = (phi_deg + 180.0) % 360.0 - 180.0

        for j in range(len(tau_arr)):
            all_y.append(y)
            all_gains.append(gains_db[j])
            all_taus.append(tau_arr[j])
            all_phis.append(phi_deg[j])

        best_idx = np.argmax(norms)
        max_gain[i] = gains_db[best_idx]
        max_tau[i] = tau_arr[best_idx]
        max_phi[i] = phi_deg[best_idx]

    # ── Geometrically tracked paths ─────────────────────────────────────
    los_g, los_t, los_p, va_g, va_t, va_p = track_geometric_paths(data)

    # ── Plotting ────────────────────────────────────────────────────────
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # ---- Subplot 1: Array Gain ------------------------------------------
    axs[0].scatter(all_y, all_gains, color="lightgray", s=8, alpha=0.4,
                   label="All Multipath Components")
    axs[0].plot(y_coords, max_gain, "b-o", markersize=5, linewidth=2,
                label="Strongest Path (argmax)")
    axs[0].plot(y_coords, los_g, "b--", linewidth=1.5, alpha=0.7,
                label="LoS (geometric tracked)")
    axs[0].plot(y_coords, va_g, "r--", linewidth=1.5, alpha=0.7,
                label="East Wall VA (geometric tracked)")
    axs[0].set_ylabel("Array Gain (dB)")
    axs[0].set_title("Raw MIMO Dataset Verification — with Theoretical References")
    axs[0].axvline(x=26.7, color="gray", linestyle="--", alpha=0.8,
                   label="LoS Blockage (Y=26.7)")
    axs[0].legend(loc="lower left", fontsize=8)
    axs[0].grid(True)

    # ---- Subplot 2: ToA -------------------------------------------------
    axs[1].scatter(all_y, all_taus, color="lightgray", s=8, alpha=0.4)
    axs[1].plot(y_coords, max_tau, "g-o", markersize=5, linewidth=2,
                label="Strongest Path ToA (argmax)")
    axs[1].plot(y_coords, tau_los_ref, "k--", linewidth=1.5, alpha=0.8,
                label="Theoretical LoS ToA (pure geometry)")
    axs[1].plot(y_coords, los_t, "b--", linewidth=1.5, alpha=0.7,
                label="LoS ToA (geometric tracked)")
    axs[1].plot(y_coords, va_t, "r--", linewidth=1.5, alpha=0.7,
                label="East Wall VA ToA (geometric tracked)")
    axs[1].set_ylabel("Time of Arrival (s)")
    axs[1].axvline(x=26.7, color="gray", linestyle="--", alpha=0.8)
    axs[1].legend(loc="upper left", fontsize=8)
    axs[1].grid(True)

    # ---- Subplot 3: AoD -------------------------------------------------
    axs[2].scatter(all_y, all_phis, color="lightgray", s=8, alpha=0.4)
    axs[2].plot(y_coords, max_phi, "r-o", markersize=5, linewidth=2,
                label="Strongest Path AoD (argmax)")
    axs[2].plot(y_coords, phi_los_ref, "k--", linewidth=1.5, alpha=0.8,
                label="Theoretical LoS AoD (pure geometry)")
    axs[2].plot(y_coords, los_p, "b--", linewidth=1.5, alpha=0.7,
                label="LoS AoD (geometric tracked)")
    axs[2].plot(y_coords, va_p, "r--", linewidth=1.5, alpha=0.7,
                label="East Wall VA AoD (geometric tracked)")
    axs[2].set_ylabel("Tx Azimuth AoD (degrees)")
    axs[2].set_xlabel("UE Y-Coordinate (m)")
    axs[2].axvline(x=26.7, color="gray", linestyle="--", alpha=0.8)
    axs[2].legend(loc="upper left", fontsize=8)
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig("mimo_dataset_raw_verification.png", dpi=300)
    print("Verification plot saved to mimo_dataset_raw_verification.png")


if __name__ == "__main__":
    main()
