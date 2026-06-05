import os
import numpy as np
import trimesh
import mitsuba as mi

# 初始化 Sionna 引擎
mi.set_variant('cuda_ad_mono_polarized')
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver


class SionnaUESimulator:
    def __init__(self):
        self.files = {'build': "metis_scene.ply", 'ground': "metis_ground.ply"}
        self.xml_path = "temp_sim.xml"
        self.output_npz = "ue_mimo_multipath_data.npz"
        self.carrier_freq = 28e9

        # 构建基础物理场景 (Building A, B, C)
        self._build_scene()
        self._generate_xml()

        print("加载场景...")
        self.scene = load_scene(self.xml_path, merge_shapes=False)
        self.scene.frequency = self.carrier_freq

        # ==============================================================
        # 收发天线阵列配置 (Downlink Beamforming 架构)
        # ==============================================================
        # 基站 (BS0) 配置为 8x8 均匀平面阵列 (UPA)，用于发射波束
        self.scene.tx_array = PlanarArray(
            num_rows=8, num_cols=8,
            vertical_spacing=0.5, horizontal_spacing=0.5,
            pattern="tr38901", polarization="V"
        )
        # 用户 (UE) 降为单天线 (1x1)，保留 3GPP 终端辐射特性
        self.scene.rx_array = PlanarArray(
            num_rows=1, num_cols=1,
            vertical_spacing=0.5, horizontal_spacing=0.5,
            pattern="tr38901", polarization="V"
        )

        # 部署基站，指向西北方向 (Orientation: [135 deg, 0, 0])
        tx = Transmitter(name="BS0", position=[40, -40, 15],
                         orientation=[np.pi * 0.75, 0, 0])
        self.scene.add(tx)

        # 部署用户终端初始位置
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
        # 中心点坐标 (x, y) 和 高度 (h)
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

        points = 122  # 沿 Y 轴采样 122 个轨迹点
        ue_xs = np.linspace(0, 0, points)
        ue_ys = np.linspace(-20, 40, points)
        ue_zs = np.linspace(1.5, 1.5, points)

        all_results = []

        for i in range(points):
            pos_i = [float(ue_xs[i]), float(ue_ys[i]), float(ue_zs[i])]
            self.ue.position = pos_i

            # 执行射线追踪
            paths = solver(self.scene, max_depth=2, synthetic_array=True,
                           samples_per_src=200000, max_num_paths_per_src=200000,
                           diffuse_reflection=True)
            paths.normalize_delays = False

            # ==============================================================
            # [终极防御版提取]: 无惧 Sionna 任何版本的维度变化
            # ==============================================================
            tau_raw = paths.tau.numpy()
            theta_t_raw = paths.theta_t.numpy()
            phi_t_raw = paths.phi_t.numpy()
            a_real = paths.a[0].numpy()
            a_imag = paths.a[1].numpy()

            # 1. 提取几何参数：因为我们是点对点通信，直接展平为 1D 数组 (长度为 max_paths)
            # ToAu_raw 的 shape 可能是 (1, 1, 1, max_paths) 或 (max_paths,) 等等，使用 flatten() 强制转为 1D 数组，长度即为实际的路径数量（包含无效路径）
            tau = tau_raw.flatten()
            # 同理，AOA/AOD 也可能有冗余维度，使用 flatten() 强制转为 1D 数组
            theta_t = theta_t_raw.flatten()
            phi_t = phi_t_raw.flatten()

            num_paths = a_real.shape[-1]
            if num_paths == 0:
                # 处理被完全阻挡、无路径的情况
                valid_idx = np.array([], dtype=bool)
                tau_valid = np.array([])
                theta_t_valid = np.array([])
                phi_t_valid = np.array([])
                h_complex_valid = np.array([])
            else:
                # 2. 动态提取复数矩阵：将前面所有冗余维度压缩，强制转为 (64, num_paths)
                # 使用 -1 自动推导，结果必然是 64，因为场景只配置了总共 64 根 TX 天线
                a_real_tx = a_real.reshape(-1, num_paths)
                a_imag_tx = a_imag.reshape(-1, num_paths)

                # 增加一道防线：确保天线数没配错
                if a_real_tx.shape[0] != 64:
                    raise ValueError(f"Antenna dimension mismatch! Expected 64 TX antennas, but got {a_real_tx.shape[0]}. Please check UE/BS config.")

                # 组合成复数，并转置为 (num_paths, 64)
                h_complex = (a_real_tx + 1j * a_imag_tx).T

                # 3. 过滤物理无效路径
                a_abs_path = np.linalg.norm(h_complex, axis=1)
                valid_idx = (a_abs_path > 1e-15) & (tau > 0)
                
                tau_valid = tau[valid_idx]
                theta_t_valid = theta_t[valid_idx]
                phi_t_valid = phi_t[valid_idx]
                h_complex_valid = h_complex[valid_idx]

            print(f"Frame {i+1:2d}/{points} | "
                  f"UE Pos=[{pos_i[0]:5.1f}, {pos_i[1]:5.1f}, {pos_i[2]:4.1f}] | "
                  f"Valid Paths: {len(tau_valid)}")

            all_results.append({
                'ue_pos': pos_i,
                'tau': tau_valid,
                'theta_t': theta_t_valid,
                'phi_t': phi_t_valid,
                'h_complex': h_complex_valid
            })

        print(f"\n仿真完成，保存数据到 {self.output_npz} ...")
        
        # 将数据序列化保存，使用 dtype=object 应对变长多径数组
        np.savez_compressed(
            self.output_npz,
            ue_positions=np.array([r['ue_pos'] for r in all_results]),
            taus=np.array([r['tau'] for r in all_results], dtype=object),
            theta_ts=np.array([r['theta_t'] for r in all_results], dtype=object),
            phi_ts=np.array([r['phi_t'] for r in all_results], dtype=object),
            h_complexs=np.array([r['h_complex'] for r in all_results], dtype=object)
        )

if __name__ == "__main__":
    sim = SionnaUESimulator()
    sim.run_simulation()