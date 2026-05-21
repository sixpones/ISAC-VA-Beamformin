import numpy as np
import matplotlib.pyplot as plt

def main():
    # Global parameters
    c = 299792458.0
    SNR_SCALE = 1e12
    N_t = 64  # 8x8 array
    
    bs_pos = np.array([40, -40, 15])
    va_pos = np.array([-60.024687, -39.953964, 15.291423])
    
    # Thresholds
    tau_threshold = 1e-9
    phi_threshold = 0.1
    
    # Load data
    npz_file = "ue_multipath_data.npz"
    try:
        data = np.load(npz_file, allow_pickle=True)
    except Exception as e:
        print(f"Error loading {npz_file}: {e}")
        return
    
    print("Loading multipath data for beamforming evaluation...")
    ue_positions = data['ue_positions']
    taus = data['taus']
    phi_rs = data['phi_rs']
    a_abss = data['a_abss']
    
    num_frames = len(ue_positions)
    
    # Storage for results
    y_coords = []
    se_exhaustive = []
    se_los = []
    se_va = []
    
    print(f"Processing {num_frames} frames...\n")
    
    for frame_idx in range(num_frames):
        ue_pos = ue_positions[frame_idx]
        y_coord = ue_pos[1]
        y_coords.append(y_coord)
        
        tau_frame = taus[frame_idx]
        phi_frame = phi_rs[frame_idx]
        a_abs_frame = a_abss[frame_idx]
        
        # ========== Method 1: Exhaustive Sweeping ==========
        if len(a_abs_frame) > 0:
            max_gain_idx = np.argmax(a_abs_frame)
            a_abs_exhaustive = a_abs_frame[max_gain_idx]
        else:
            a_abs_exhaustive = 0
        
        snr_exhaustive = (a_abs_exhaustive ** 2) * SNR_SCALE * N_t
        se_exhaustive.append(np.log2(1 + snr_exhaustive))
        
        # ========== Method 2: LoS-only Beamforming ==========
        d_los = np.linalg.norm(ue_pos - bs_pos)
        tau_los = d_los / c
        
        a_abs_los = 0
        for path_idx in range(len(tau_frame)):
            tau_meas = tau_frame[path_idx]
            tau_error = abs(tau_meas - tau_los)
            if tau_error < tau_threshold:
                a_abs_los = max(a_abs_los, a_abs_frame[path_idx])
        
        snr_los = (a_abs_los ** 2) * SNR_SCALE * N_t
        se_los.append(np.log2(1 + snr_los))
        
        # ========== Method 3: Proposed VA-assisted ==========
        d_va = np.linalg.norm(ue_pos - va_pos)
        tau_va = d_va / c
        phi_va = np.arctan2(va_pos[1] - ue_pos[1], va_pos[0] - ue_pos[0])
        
        a_abs_va = 0
        for path_idx in range(len(tau_frame)):
            tau_meas = tau_frame[path_idx]
            phi_meas = phi_frame[path_idx]
            
            tau_error = abs(tau_meas - tau_va)
            if tau_error >= tau_threshold:
                continue
            
            delta_phi = abs(np.arctan2(np.sin(phi_meas - phi_va), np.cos(phi_meas - phi_va)))
            if delta_phi >= phi_threshold:
                continue
            
            a_abs_va = max(a_abs_va, a_abs_frame[path_idx])
        
        snr_va = (a_abs_va ** 2) * SNR_SCALE * N_t
        se_va.append(np.log2(1 + snr_va))
    
    # Print summary
    print("="*70)
    print("Beamforming Evaluation Results")
    print("="*70)
    print(f"Exhaustive SE (avg): {np.mean(se_exhaustive):.3f} bps/Hz")
    print(f"LoS-only SE (avg):   {np.mean(se_los):.3f} bps/Hz")
    print(f"VA-assisted SE (avg):{np.mean(se_va):.3f} bps/Hz")
    print("="*70 + "\n")
    
    # Create figure with 2 subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ========== Subplot 1: Spectral Efficiency vs Y-Coordinate ==========
    ax1.plot(y_coords, se_exhaustive, 'k--', label='Exhaustive Sweeping', linewidth=2.5)
    ax1.plot(y_coords, se_los, 'g-.', label='LoS-only Beamforming', linewidth=2.5)
    ax1.plot(y_coords, se_va, 'r-o', label='Proposed VA-Assisted', linewidth=2.5, markersize=4)
    
    # Add LoS blockage boundary
    ax1.axvline(x=26.7, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='LoS Blockage')
    
    ax1.set_xlabel('UE Y-Coordinate (m)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Spectral Efficiency (bps/Hz)', fontsize=13, fontweight='bold')
    ax1.set_title('Spectral Efficiency Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # ========== Subplot 2: Beam Training Overhead Comparison ==========
    methods = ['Exhaustive\nSweeping', 'LoS-only', 'VA-Assisted']
    overhead = [64, 1, 1]
    colors_bar = ['black', 'green', 'red']
    
    bars = ax2.bar(methods, overhead, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, val in zip(bars, overhead):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Number of Candidate Beams Evaluated', fontsize=13, fontweight='bold')
    ax2.set_title('Beam Training Overhead Comparison', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y', which='both', linestyle='--')
    
    plt.tight_layout()
    plt.savefig("beamforming_evaluation.png", dpi=300, bbox_inches='tight')
    print("✓ Figure saved to beamforming_evaluation.png")

if __name__ == "__main__":
    main()
