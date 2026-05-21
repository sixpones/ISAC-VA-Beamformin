import numpy as np
import matplotlib.pyplot as plt

def main():
    # Fixed parameters
    va_pos = np.array([-60.024687, -39.953964, 15.291423])
    c = 299792458.0
    
    npz_file = "ue_multipath_data.npz"
    
    try:
        data = np.load(npz_file, allow_pickle=True)
    except Exception as e:
        print(f"Error loading {npz_file}: {e}")
        return
    
    print("Loading multipath data...")
    ue_positions = data['ue_positions']
    taus = data['taus']
    phi_rs = data['phi_rs']
    a_abss = data['a_abss']
    
    num_frames = len(ue_positions)
    
    # Threshold definitions
    tau_threshold = 1e-9  # 1 nanosecond
    phi_threshold = 0.1   # 0.1 radian
    
    # Results storage
    y_coords = []
    benchmark_gains_db = []
    va_matched_gains_db = []
    matched_frames = 0
    
    print(f"\nProcessing {num_frames} frames...")
    
    for frame_idx in range(num_frames):
        ue_pos = ue_positions[frame_idx]
        y_coord = ue_pos[1]
        y_coords.append(y_coord)
        
        # Theoretical values
        d_th = np.linalg.norm(va_pos - ue_pos)
        tau_th = d_th / c
        phi_th = np.arctan2(va_pos[1] - ue_pos[1], va_pos[0] - ue_pos[0])
        
        # Get current frame multipath data
        tau_frame = taus[frame_idx]
        phi_frame = phi_rs[frame_idx]
        a_abs_frame = a_abss[frame_idx]
        
        # Global maximum gain (benchmark)
        if len(a_abs_frame) > 0:
            max_gain = np.max(a_abs_frame)
            benchmark_db = 20 * np.log10(max_gain) if max_gain > 0 else -120
        else:
            benchmark_db = -120
        
        benchmark_gains_db.append(benchmark_db)
        
        # VA matched multipath maximum gain
        va_matched_gains = []
        
        for path_idx in range(len(tau_frame)):
            tau_meas = tau_frame[path_idx]
            phi_meas = phi_frame[path_idx]
            a_abs_meas = a_abs_frame[path_idx]
            
            # Condition A: Delay lock
            tau_error = abs(tau_meas - tau_th)
            if tau_error >= tau_threshold:
                continue
            
            # Condition B: Angle lock (phase unwrapping to avoid ±π wrapping)
            delta_phi = abs(np.arctan2(np.sin(phi_meas - phi_th), np.cos(phi_meas - phi_th)))
            if delta_phi >= phi_threshold:
                continue
            
            # Both conditions satisfied, record this multipath gain
            va_matched_gains.append(a_abs_meas)
        
        # Record VA matched gain
        if len(va_matched_gains) > 0:
            va_max_gain = np.max(va_matched_gains)
            va_matched_db = 20 * np.log10(va_max_gain) if va_max_gain > 0 else -120
            matched_frames += 1
        else:
            va_matched_db = -120
        
        va_matched_gains_db.append(va_matched_db)
    
    # Terminal output
    print(f"\n{'='*60}")
    print(f"ISAC Virtual Anchor Matching Results")
    print(f"{'='*60}")
    print(f"Total frames analyzed: {num_frames}")
    print(f"Frames with successful VA match: {matched_frames}")
    print(f"Match success rate: {100 * matched_frames / num_frames:.2f}%")
    print(f"{'='*60}\n")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(y_coords, benchmark_gains_db, 'b--', label='Benchmark (Max Gain)', linewidth=2.5)
    ax.plot(y_coords, va_matched_gains_db, 'r-o', label='Proposed Method (VA Matching)', 
            markersize=4, linewidth=2.5)
    
    # Add LoS Blockage Boundary
    ax.axvline(x=26.7, color='gray', linestyle='--', alpha=0.7, label='LoS Blockage Boundary', linewidth=2)
    
    # Labels and formatting
    ax.set_xlabel('UE Y-Coordinate (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Received Power Gain (dB)', fontsize=13, fontweight='bold')
    ax.set_title('Virtual Anchor Assisted Beam Training: ISAC Matching Performance', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig("va_matching_result.png", dpi=300)
    print("✓ Plot saved to va_matching_result.png")

if __name__ == "__main__":
    main()
