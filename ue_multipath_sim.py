import os
import numpy as np
import trimesh
import mitsuba as mi

mi.set_variant('cuda_ad_mono_polarized')
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver


class SionnaUESimulator:
    def __init__(self):
        self.files = {'build': "metis_scene.ply", 'ground': "metis_ground.ply"}
        self.xml_path = "temp_sim.xml"
        self.output_npz = "ue_multipath_data.npz"
        self.carrier_freq = 28e9

        self._build_scene()
        self._generate_xml()

        print("加载场景...")
        self.scene = load_scene(self.xml_path, merge_shapes=False)
        self.scene.frequency = self.carrier_freq

        # 与感知链路一致: 8x8 UPA, 3GPP TR38.901
        self.scene.tx_array = PlanarArray(
            num_rows=8, num_cols=8,
            vertical_spacing=0.5, horizontal_spacing=0.5,
            pattern="tr38901", polarization="V"
        )
        self.scene.rx_array = PlanarArray(
            num_rows=8, num_cols=8,
            vertical_spacing=0.5, horizontal_spacing=0.5,
            pattern="tr38901", polarization="V"
        )

        tx = Transmitter(name="BS0", position=[40, -40, 15],
                         orientation=[np.pi * 0.75, 0, 0])
        self.scene.add(tx)

        self.ue = Receiver(name="UE", position=[0, -20, 1.5],
                           orientation=[0, 0, 0])
        self.scene.add(self.ue)

    def _build_scene(self):
        print("构建物理场景模型...")
        if os.path.exists(self.files['build']):
            os.remove(self.files['build'])
        if os.path.exists(self.files['ground']):
            os.remove(self.files['ground'])

        buildings_config = [
            (-40, 40, 25.0),
            (40, 40, 30.0),
            (-40, -40, 35.0)
        ]

        buildings = []
        for cx, cy, h in buildings_config:
            b = trimesh.creation.box(
                extents=[60.0, 60.0, h],
                transform=trimesh.transformations.translation_matrix([cx, cy, h / 2])
            )
            buildings.append(b)

        trimesh.util.concatenate(buildings).export(self.files['build'])
        trimesh.creation.box(extents=[300, 300, 1]).export(self.files['ground'])

    def _generate_xml(self):
        print("生成 XML 材质文件...")
        xml = f"""<scene version="2.1.0">
        <default name="integrator" value="path"/>
        <integrator type="path"/>

        <bsdf type="radio-material" id="custom_concrete">
            <float name="relative_permittivity" value="5.0"/>
            <float name="conductivity" value="0.05"/>
            <float name="scattering_coefficient" value="0.7"/>
            <float name="thickness" value="0.1"/>
            <float name="xpd_coefficient" value="0.0"/>
        </bsdf>

        <bsdf type="radio-material" id="ground_mat">
            <float name="relative_permittivity" value="4.0"/>
            <float name="conductivity" value="0.1"/>
            <float name="scattering_coefficient" value="0.5"/>
            <float name="thickness" value="0.1"/>
            <float name="xpd_coefficient" value="0.0"/>
        </bsdf>

        <shape type="ply" id="b">
            <string name="filename" value="{self.files['build']}"/>
            <boolean name="face_normals" value="true"/>
            <ref id="custom_concrete" name="bsdf"/>
        </shape>
        <shape type="ply" id="g">
            <string name="filename" value="{self.files['ground']}"/>
            <boolean name="face_normals" value="true"/>
            <ref id="ground_mat" name="bsdf"/>
        </shape>
        </scene>"""
        with open(self.xml_path, "w") as f:
            f.write(xml)

    def run_simulation(self):
        print("开始射线追踪仿真...")
        solver = PathSolver()

        points = 122
        ue_xs = np.linspace(0, 0, points)
        ue_ys = np.linspace(-20, 40, points)
        ue_zs = np.linspace(1.5, 1.5, points)

        all_results = []

        for i in range(points):
            pos_i = [float(ue_xs[i]), float(ue_ys[i]), float(ue_zs[i])]
            self.ue.position = pos_i

            paths = solver(self.scene, max_depth=2, synthetic_array=True,
                           samples_per_src=200000, max_num_paths_per_src=200000,
                           diffuse_reflection=True)
            paths.normalize_delays = False

            tau = paths.tau.numpy()
            theta_r = paths.theta_r.numpy()
            phi_r = paths.phi_r.numpy()
            a_real = paths.a[0].numpy()
            a_imag = paths.a[1].numpy()

            if tau.ndim >= 3:
                tau = tau[0, 0, :]
                theta_r = theta_r[0, 0, :]
                phi_r = phi_r[0, 0, :]

            if a_real.ndim == 5:
                num_paths = a_real.shape[-1]
                a_real_2d = a_real.reshape(-1, num_paths)
                a_imag_2d = a_imag.reshape(-1, num_paths)
                a_abs = np.sqrt(np.sum(a_real_2d ** 2 + a_imag_2d ** 2, axis=0))
            elif a_real.ndim == 3:
                a_real = a_real[0, 0, :]
                a_imag = a_imag[0, 0, :]
                a_abs = np.hypot(a_real, a_imag)
            else:
                a_abs = np.hypot(a_real, a_imag)

            a_abs = a_abs.flatten()
            tau = tau.flatten()
            theta_r = theta_r.flatten()
            phi_r = phi_r.flatten()

            valid_idx = a_abs > 1e-15
            tau_valid = tau[valid_idx]
            theta_r_valid = theta_r[valid_idx]
            phi_r_valid = phi_r[valid_idx]
            a_abs_valid = a_abs[valid_idx]

            print(f"帧 {i+1:3d}/{points}  "
                  f"UE=[{pos_i[0]:5.1f}, {pos_i[1]:5.1f}, {pos_i[2]:4.1f}]  "
                  f"多径数={len(tau_valid)}")

            all_results.append({
                'ue_pos': pos_i,
                'tau': tau_valid,
                'theta_r': theta_r_valid,
                'phi_r': phi_r_valid,
                'a_abs': a_abs_valid
            })

        print(f"\n仿真完成，保存数据到 {self.output_npz} ...")
        np.savez_compressed(
            self.output_npz,
            ue_positions=np.array([r['ue_pos'] for r in all_results]),
            taus=np.array([r['tau'] for r in all_results], dtype=object),
            theta_rs=np.array([r['theta_r'] for r in all_results], dtype=object),
            phi_rs=np.array([r['phi_r'] for r in all_results], dtype=object),
            a_abss=np.array([r['a_abs'] for r in all_results], dtype=object)
        )

        for f in [self.files['build'], self.files['ground'], self.xml_path]:
            if os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    sim = SionnaUESimulator()
    sim.run_simulation()
