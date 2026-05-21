import numpy as np
import matplotlib.pyplot as plt

def main():
    npz_file = "ue_multipath_data.npz"
    try:
        data = np.load(npz_file, allow_pickle=True)
    except Exception as e:
        print(f"Error loading {npz_file}: {e}")
        return

    print("Keys in npz file:", data.files)
    
    ue_positions = data['ue_positions']
    taus = data['taus']
    theta_rs = data['theta_rs']
    phi_rs = data['phi_rs']
    a_abss = data['a_abss']
    
    num_frames = len(ue_positions)
    
    y_coords = []
    max_gain_db = []
    max_tau = []
    max_phi_deg = []
    
    for i in range(num_frames):
        y = ue_positions[i][1]
        y_coords.append(y)
        
        a_abs_frame = a_abss[i]
        if len(a_abs_frame) == 0:
            max_gain_db.append(np.nan)
            max_tau.append(np.nan)
            max_phi_deg.append(np.nan)
            continue
            
        max_idx = np.argmax(a_abs_frame)
        
        gain_val = a_abs_frame[max_idx]
        gain_db = 20 * np.log10(gain_val) if gain_val > 0 else np.nan
        
        tau_val = taus[i][max_idx]
        phi_rad = phi_rs[i][max_idx]
        phi_deg = np.degrees(phi_rad)
        
        # Normalize to [-180, 180]
        phi_deg = (phi_deg + 180) % 360 - 180
        
        max_gain_db.append(gain_db)
        max_tau.append(tau_val)
        max_phi_deg.append(phi_deg)

    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Subplot 1: Max Path Gain
    axs[0].plot(y_coords, max_gain_db, 'bo-', markersize=4)
    axs[0].set_ylabel('Max Path Gain (dB)')
    axs[0].set_title('Strongest Multipath Evaluation vs. UE Y-Coordinate')
    axs[0].axvline(x=20, color='r', linestyle='--', label='LoS/NLoS Boundary (Y=20)')
    axs[0].legend()
    axs[0].grid(True)
    
    # Subplot 2: ToA
    axs[1].plot(y_coords, max_tau, 'go-', markersize=4)
    axs[1].set_ylabel('ToA / tau (s)')
    axs[1].axvline(x=20, color='r', linestyle='--')
    axs[1].grid(True)
    
    # Subplot 3: AoA (Azimuth)
    axs[2].plot(y_coords, max_phi_deg, 'ro-', markersize=4)
    axs[2].set_ylabel('AoA Azimuth (deg)')
    axs[2].set_xlabel('UE Y-Coordinate (m)')
    axs[2].axvline(x=20, color='r', linestyle='--')
    axs[2].grid(True)
    
    plt.tight_layout()
    output_png = "multipath_sanity_check.png"
    plt.savefig(output_png, dpi=300)
    print(f"Plot saved successfully to {output_png}")

if __name__ == "__main__":
    main()
