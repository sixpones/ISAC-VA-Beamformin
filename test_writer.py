code = """import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray

class ISACVirtualAnchorMatching:
    def __init__(self, scene_file="temp_sim.xml"):
        self.scene_file = scene_file
        self.bs_pos = np.array([40.0, -40.0, 15.0])
        self.ue_start = np.array([0.0, -20.0, 1.5])
        self.ue_end = np.array([0.0, 40.0, 1.5])
        self.num_points = 50
        self.va_pos = np.array([-19.94, -40.00, 14.92])
        self.c = 299792458.0
        self.tol = 1e-9
        
    def setup_scene(self):
        print(f"Loading Sionna scene from: {self.scene_file}...")
        try:
            self.scene = load_scene(self.scene_file)
        except Exception as e:
            print(f"[Warning] Failed to load scene, using empty target. Error: {e}")
            self.scene = sionna.rt.Scene()
            
        if "BS0" not in self.scene.transmitters:
            self.scene.add(Transmitter(name="BS0", position=self.bs_pos))
        if "UE" not in self.scene.receivers:
            self.scene.add(Receiver(name="UE", position=self.ue_start))
            
        self.scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="iso", polarization="V")
        self.scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="iso", polarization="V")

    def run_simulation(self):
        ue_positions = np.linspace(self.ue_start, self.ue_end, self.num_points)
        y_coords = ue_positions[:, 1]
        
        max_gains_db, va_matched_gains_db = [], []
        
        for i, pos in enumerate(ue_positions):
            self.scene.get("UE").position = pos
            try:
                paths = self.scene.compute_paths(max_depth=3)
                a_flat = paths.a.numpy().flatten()
                tau_flat = paths.tau.numpy().flatten()
                a_abs = np.abs(a_flat)
                
                valid_mask = tau_flat > 0
                a_abs = a_abs[valid_mask]
                tau_flat = tau_flat[valid_mask]
            except Exception:
                a_abs, tau_flat = np.array([]), np.array([])
            
            if len(a_abs) > 0:
                max_gains_db.append(20 * np.log10(np.max(a_abs) + 1e-15))
            else:
                max_gains_db.append(-120)
                
            dist_theory = np.linalg.norm(pos - self.va_pos)
            tau_theory = dist_theory / self.c
            
            matched = False
            if len(tau_flat) > 0:
                errors = np.abs(tau_flat - tau_theory)
                matched_indices = np.where(errors < self.tol)[0]
                if len(matched_indices) > 0:
                    va_matched_gains_db.append(20 * np.log10(np.max(a_abs[matched_indices]) + 1e-15))
                    matched = True
            
            if not matched:
                va_matched_gains_db.append(-120)

        return y_coords, max_gains_db, va_matched_gains_db
        
    def plot_results(self, y_coords, max_gains_db, va_matched_gains_db):
        plt.figure(figsize=(10, 6))
        plt.plot(y_coords, max_gains_db, label="Global Max Path Gain", color='orange')
        plt.plot(y_coords, va_matched_gains_db, label="VA Matched Path Gain", linestyle='--', marker='o')
        plt.axvline(x=0, color='gray', linestyle=':', label='LoS to NLoS Transition')
        plt.xlabel("UE Y-Coordinate (m)")
        plt.ylabel("Received Path Gain (dB)")
        plt.legend(loc='lower left')
        plt.savefig("va_multipath_matching.png", dpi=300)

if __name__ == '__main__':
    sim = ISACVirtualAnchorMatching()
    sim.setup_scene()
    y, max_g, va_g = sim.run_simulation()
    sim.plot_results(y, max_g, va_g)
"""
with open("/home/xu/xdh/isac/ISAC_beam/isac_va_matching.py", "w") as f:
    f.write(code)
