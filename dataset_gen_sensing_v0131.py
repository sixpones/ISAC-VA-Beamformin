"""Generate a simple ISAC sensing point-cloud dataset with Sionna RT.

输出文件:
    isac_pure_sensing.h5

H5 数据结构:
    sensing_pcl: [num_frames, 4096, 5]
    最后一维 5 个值依次是 [x, y, z, bs_index, path_gain]
"""

import os

# Dr.Jit/Mitsuba 会在 HOME 下写缓存。Codex/部分沙箱里用户 HOME 可能只读，
# 所以把缓存目录固定到当前工程目录，避免 ".drjit 写入失败" 这类错误。
os.environ['HOME'] = os.getcwd()
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), ".matplotlib")

# 这些 TensorFlow 环境变量保留给老版本 Sionna/TF 组合；当前脚本不直接导入 TF。
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import argparse
import numpy as np
import trimesh
import h5py
from tqdm import tqdm


def configure_mitsuba_variant(use_gpu=False):
    """选择 Mitsuba/Dr.Jit 后端。

    必须在导入 sionna.rt 之前调用，否则 Sionna 会绑定默认后端。
    *_polarized 后端用于极化信道计算；如果用 rgb/mono 非极化后端，
    RadioMaterial 的 Jones 矩阵维度会不匹配。
    """
    import mitsuba as mi

    variant = "cuda_ad_mono_polarized" if use_gpu else "llvm_ad_mono_polarized"
    mi.set_variant(variant)
    print(f"Mitsuba variant: {mi.variant()}")


def load_sionna_rt():
    """延迟导入 Sionna RT。

    这样可以先调用 configure_mitsuba_variant()，再让 Sionna RT 使用对应 CPU/GPU 后端。
    """
    global load_scene, Transmitter, Receiver, PlanarArray, PathSolver

    try:
        import sionna
        from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver

        print(f"✅ Sionna {sionna.__version__} 环境加载成功")
    except ImportError:
        raise RuntimeError("❌ Sionna 导入失败")


class ISAC_METIS_Variable_Height:
    def __init__(self):
        # Sionna/Mitsuba 场景使用 PLY mesh + 临时 XML 描述。
        self.files = {'build': "metis_scene.ply", 'ground': "metis_ground.ply"}
        self.h5_path = "isac_pure_sensing.h5"

        # [场景尺寸]
        self.block_size = 60.0
        self.street_width = 20.0

        # [基站配置]
        self.bs_list = [
            # BS0: 右下角空地 (40, -40, 15), 朝向西北 (135度)
            {'name': 'BS_0_Corner', 'pos': [40, -40, 15], 'ori': [np.pi * 0.75, 0, 0]},
            # BS1: 北侧 (0, 60, 15), 朝南 (270度)
            {'name': 'BS_1_North', 'pos': [0, 60, 15], 'ori': [np.pi * 1.5, 0, 0]},
            # BS2: 西侧 (-60, 0, 15), 朝东 (0度)
            {'name': 'BS_2_West', 'pos': [-60, 0, 15], 'ori': [0, 0, 0]}
        ]

        self.max_pcl_points = 4096  # 每帧最多保存的点数，多余点随机下采样
        self.carrier_freq = 28e9    # 28 GHz 载频
        self.c = 299792458.0        # 光速，用 tau 转距离

        # 初始化时重建场景和 H5。注意：每次运行会覆盖旧的 isac_pure_sensing.h5。
        self._build_scene()
        self._generate_xml()
        self._init_storage()

    def _build_scene(self):
        if os.path.exists(self.files['build']): os.remove(self.files['build'])
        print("[Info] 构建 2x2 场景 (变高版: 25m, 30m, 35m)...")
        buildings = []

        # [修改点] 定义每栋楼的中心坐标和高度
        # 左上(NW): 25m  西北
        # 右上(NE): 30m  东北
        # 左下(SW): 35m  西南
        buildings_config = [
            (-40, 40, 25.0),
            (40, 40, 30.0),
            (-40, -40, 35.0)
        ]

        for cx, cy, h in buildings_config:
            b = trimesh.creation.box(
                extents=[self.block_size, self.block_size, h],
                # 注意：Z轴偏移量必须是高度的一半 (h/2)，保证建筑物底部紧贴地面 (Z=0)
                transform=trimesh.transformations.translation_matrix([cx, cy, h / 2])
            )
            buildings.append(b)

        # 建筑物和地面分别导出为 PLY，后续由 XML 绑定无线材质。
        trimesh.util.concatenate(buildings).export(self.files['build'])
        trimesh.creation.box(extents=[300, 300, 1]).export(self.files['ground'])

    def _generate_xml(self):
        # Sionna 1.2 加载场景时要求 shape 已经绑定 RadioMaterial。
        # 这里直接在 XML 中定义 radio-material 并用 <ref> 绑定给建筑/地面。
        # 注意 custom_concrete 不能命名为 concrete，否则会和 Sionna 内置 ITU 材质命名规则冲突。
        xml = f"""<scene version="2.1.0">
        <default name="integrator" value="path"/>
        <integrator type="path"/>
        <!--用于建筑物的无线材质，用于建筑物，相对介电常数设为 5.0,电导率 0.05,散射系数 0.7参数可根据需要调整。相较于默认的理想反射，增加了散射系数和导电性，更接近现实建筑。-->
        <bsdf type="radio-material" id="custom_concrete"> 
            <float name="relative_permittivity" value="5.0"/>
            <float name="conductivity" value="0.05"/>
            <float name="scattering_coefficient" value="0.7"/>
            <float name="thickness" value="0.1"/>
            <float name="xpd_coefficient" value="0.0"/>
        </bsdf>
        <!--用于地面的无线材质，用于地面，相对介电常数 4.0,电导率 0.1,散射系数 0.5参数可根据需要调整。相较于默认的理想反射，增加了散射系数和导电性，更接近现实地面。-->
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
        with open("temp_sim.xml", "w") as f: f.write(xml)

    def _init_storage(self):
        # shape=(0, ...) 表示先创建一个空的可扩展数据集；每生成一帧再 append。
        with h5py.File(self.h5_path, 'w') as f:
            f.create_dataset('sensing_pcl', shape=(0, self.max_pcl_points, 5), maxshape=(None, self.max_pcl_points, 5),
                             dtype=np.float32)

    def run(self, num_frames=1, samples_per_src=100000, max_depth=2, diffuse_reflection=True,
            bs_index=None, max_num_paths_per_src=200000, array_size=8, allow_empty=False):
        """运行射线追踪并写入 H5。

        samples_per_src:
            每个发射端采样的射线数量，越大越容易得到散射点，但显存/内存也越高。
        max_depth:
            最大交互次数，例如反射/散射的路径深度。
        max_num_paths_per_src:
            路径缓存上限。RTX 5080 16GB 下，200000 比默认 1000000 更稳。
        bs_index:
            None 表示跑 3 个基站；0/1/2 表示只跑单基站，适合大 samples 测试。
        allow_empty:
            False 时没有有效点会直接报错，避免把 OOM 误当作“正常空数据”。
        """
        if bs_index is None:
            bs_indices = list(range(len(self.bs_list)))
        else:
            if bs_index < 0 or bs_index >= len(self.bs_list):
                raise ValueError(f"bs_index must be in [0, {len(self.bs_list) - 1}]")
            bs_indices = [bs_index]

        # merge_shapes=True 在这个小 PLY 场景上曾触发 Dr.Jit 底层段错误，
        # 且本场景只有建筑和地面两个 mesh，禁用合并不会造成明显性能损失。
        scene = load_scene("temp_sim.xml", merge_shapes=False)
        scene.frequency = self.carrier_freq

        print("正在检查场景材质...")
        for name, obj in scene.objects.items():
            print(f"   -> {name}: {obj.radio_material.name}")

        # MIMO 阵列。array_size=8 表示 8x8；调小到 4 可以显著降低显存压力。
        scene.tx_array = PlanarArray(num_rows=array_size, num_cols=array_size, vertical_spacing=0.5, horizontal_spacing=0.5,
                                     pattern="tr38901", polarization="V")
        scene.rx_array = PlanarArray(num_rows=array_size, num_cols=array_size, vertical_spacing=0.5, horizontal_spacing=0.5,
                                     pattern="tr38901", polarization="V")

        solver = PathSolver()
        print(f"开始生成 (BS0 @ {self.bs_list[0]['pos']})...")

        for i in tqdm(range(num_frames)):
            batch_fused_pcl = []
            failed_bs = []

            for b_idx in bs_indices:
                bs = self.bs_list[b_idx]
                tx_name = bs['name']
                rx_name = f"Radar_Rx_{b_idx}"
                # 接收机位置微调，朝向归零
                rx_pos = [bs['pos'][0] + 0.1, bs['pos'][1] + 0.1, bs['pos'][2]]

                # 重要：每次只把当前 BS 的 TX 放进场景。
                # 如果 3 个 TX 同时存在，PathSolver 会为 3 个 TX 一起分配路径缓存，
                # 在 --samples 2000000 / --max-depth 3 时很容易直接 OOM。
                scene.add(Transmitter(name=tx_name, position=bs['pos'], orientation=bs['ori']))
                scene.add(Receiver(name=rx_name, position=rx_pos, orientation=[0, 0, 0]))

                print(f"Frame {i}, BS{b_idx}: samples={samples_per_src}, max_depth={max_depth}, "
                      f"max_paths={max_num_paths_per_src}, array={array_size}x{array_size}")
                try:
                    # Sionna 1.2 使用 PathSolver()(scene, ...) 计算路径。
                    # diffuse_reflection=True 时会有更多散射点，但也是主要显存开销来源。
                    paths_r = solver(scene, max_depth=max_depth, max_num_paths_per_src=max_num_paths_per_src,
                                     samples_per_src=samples_per_src, diffuse_reflection=diffuse_reflection,
                                     synthetic_array=True)

                    if paths_r.a is not None:
                        # paths_r.a 在 Sionna 1.2 中是 (real, imag)，这里转为幅度。
                        a_real, a_imag = paths_r.a
                        a_abs = np.hypot(np.asarray(a_real), np.asarray(a_imag))

                        def select_tx(arr):
                            # 当前场景只保留一个 TX 和一个 RX，因此 tx/rx 维度都取 0。
                            # synthetic_array=True 时常见形状是 [rx, tx, paths]；
                            # 非 synthetic 或不同后端下可能是 [rx, rx_ant, tx, tx_ant, paths]。
                            arr = np.asarray(arr)
                            if arr.ndim == 5:
                                return arr[0, 0, 0, 0, :].flatten()
                            if arr.ndim == 3:
                                return arr[0, 0, :].flatten()
                            return arr.flatten()

                        raw_gains = select_tx(a_abs)
                        taus = select_tx(paths_r.tau)
                        thetas = select_tx(paths_r.theta_r)
                        phis = select_tx(paths_r.phi_r)

                        # 不同张量在极端情况下长度可能不同，截到共同长度防止越界。
                        min_len = min(len(raw_gains), len(taus))
                        raw_gains = raw_gains[:min_len]

                        # 过滤太弱的路径。阈值越低，保留点越多，也可能包含更多噪声/数值尾巴。
                        valid_idx = np.where(raw_gains > 1e-40)[0]

                        for pid in valid_idx:
                            # Sionna 给的是传播延迟 tau。这里按雷达往返距离近似 dist = tau*c/2。
                            dist = taus[pid] * self.c / 2
                            th, ph = thetas[pid], phis[pid]

                            # 根据到达角构造单位方向向量，再从接收机位置反推出散射点坐标。
                            u_vec = np.array([
                                np.sin(th) * np.cos(ph),
                                np.sin(th) * np.sin(ph),
                                np.cos(th)
                            ])
                            pt = np.array(rx_pos) + dist * u_vec

                            # 过滤地面 (Z > -1.0) 保留部分地杂波
                            if dist > 2.0 and pt[2] > -1.0:
                                batch_fused_pcl.append([pt[0], pt[1], pt[2], float(b_idx), raw_gains[pid]])

                except Exception as e:
                    failed_bs.append(b_idx)
                    print(f"Error BS{b_idx}: {e}")

                finally:
                    # 无论成功还是失败，都移除本轮设备，保证下一轮只有一个 TX/RX。
                    scene.remove(rx_name)
                    scene.remove(tx_name)

            # 存储
            final_pcl = np.zeros((1, self.max_pcl_points, 5))
            if len(batch_fused_pcl) > 0:
                arr = np.array(batch_fused_pcl)
                # 固定每帧点数为 4096，方便后续训练/可视化代码读取。
                if len(arr) > self.max_pcl_points:
                    sel = np.random.choice(len(arr), self.max_pcl_points, replace=False)
                    final_pcl[0] = arr[sel]
                else:
                    final_pcl[0, :len(arr)] = arr
            else:
                message = (
                    f"Frame {i}: 没有生成有效点。失败 BS: {failed_bs}. "
                    "请降低 --samples/--max-depth/--max-paths，或指定 --bs-index 单基站运行。"
                )
                if not allow_empty:
                    raise RuntimeError(message)
                print(message)
                print(f"Frame {i}: --allow-empty 已开启，写入全 0 占位帧")

            with h5py.File(self.h5_path, 'a') as f:
                f['sensing_pcl'].resize(f['sensing_pcl'].shape[0] + 1, axis=0)
                f['sensing_pcl'][-1:] = final_pcl

        print(f"\n数据生成完毕: {self.h5_path}")
        if os.path.exists("temp_sim.xml"): os.remove("temp_sim.xml")


if __name__ == "__main__":
    # 例子：
    #   CPU 单基站高采样:
    #       python dataset_gen_sensing_v0131.py --samples 2000000 --max-depth 3 --bs-index 0 --max-paths 200000
    #   GPU 单基站高采样:
    #       python dataset_gen_sensing_v0131.py --gpu --samples 2000000 --max-depth 3 --max-paths 200000
    #       python dataset_gen_sensing_v0131.py --gpu --samples 2000000 --max-depth 3 --bs-index 0 --max-paths 200000    
    parser = argparse.ArgumentParser(description="Generate ISAC METIS sensing point cloud data.")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-paths", type=int, default=200000)
    parser.add_argument("--array-size", type=int, default=8)
    parser.add_argument("--bs-index", type=int, default=None, help="Only run one BS: 0, 1, or 2.")
    parser.add_argument("--no-diffuse", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--gpu", action="store_true", help="Use Mitsuba/Dr.Jit CUDA backend for ray tracing.")
    args = parser.parse_args()

    configure_mitsuba_variant(use_gpu=args.gpu)
    load_sionna_rt()
    sim = ISAC_METIS_Variable_Height()
    sim.run(num_frames=args.frames, samples_per_src=args.samples, max_depth=args.max_depth,
            diffuse_reflection=not args.no_diffuse, bs_index=args.bs_index,
            max_num_paths_per_src=args.max_paths, array_size=args.array_size,
            allow_empty=args.allow_empty)
