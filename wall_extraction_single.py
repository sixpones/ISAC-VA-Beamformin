import h5py
import numpy as np
import open3d as o3d
import plotly.graph_objects as go

def load_pointcloud(h5_path, frame_idx=0, power_thresh=1e-50):
    """
    1. 读取点云
    使用和原版相同的全量点云读取方式，以保证 RANSAC 能利用全量数据进行更稳定、完整的平面拟合。
    """
    print(f"Loading point cloud from frame {frame_idx} in {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        # shape: (N_frame, 4096, 5) -> [x, y, z, bs_id, power]
        pcl_data = f['sensing_pcl'][frame_idx]
    
    # 过滤无效点 (仅依赖功率)
    valid_mask = pcl_data[:, 4] > power_thresh
    valid_points = pcl_data[valid_mask]
    
    # 提取坐标 [x, y, z] 和 bs_id
    xyz = valid_points[:, 0:3]
    bs_ids = valid_points[:, 3]
    print(f"Loaded {len(xyz)} valid points (power > {power_thresh}).")
    
    # 转换为 Open3D 格式
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd, xyz, bs_ids

def extract_planes_ransac(pcd, distance_threshold=0.5, ransac_n=3, num_iterations=1000):
    """
    2. RANSAC 平面提取
    """
    print("Starting RANSAC plane extraction...")
    xyz = np.asarray(pcd.points)
    remaining_indices = np.arange(len(xyz))
    
    plane_models = []
    plane_inliers_indices = []
    
    while len(remaining_indices) > 50:
        pcd_tmp = pcd.select_by_index(remaining_indices)
        plane_model, inliers_tmp = pcd_tmp.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        if len(inliers_tmp) == 0:
            break
            
        original_inliers = remaining_indices[inliers_tmp]
        plane_models.append(plane_model)
        plane_inliers_indices.append(original_inliers)
        remaining_indices = np.delete(remaining_indices, inliers_tmp)
        
    print(f"Extracted {len(plane_models)} raw planes.")
    return plane_models, plane_inliers_indices, remaining_indices

def filter_wall_planes(plane_models, plane_inliers_indices):
    """
    3. 地面过滤 & 主反射墙筛选
    """
    print("Filtering wall planes...")
    valid_models = []
    valid_inliers = []
    
    for i, (model, inliers) in enumerate(zip(plane_models, plane_inliers_indices)):
        a, b, c, d = model
        inlier_num = len(inliers)
        
        # 条件 1：地面过滤 (法向量Z分量绝对值 > 0.9 则为地面)
        if abs(c) > 0.9:
            continue
            
        # 条件 2：面积(点数)过滤
        if inlier_num <= 300:
            continue
            
        valid_models.append(model)
        valid_inliers.append(inliers)
        
    print(f"Kept {len(valid_models)} valid wall planes after filtering.")
    return valid_models, valid_inliers

def visualize_planes(original_xyz, wall_inliers_list, bs_positions, output_html, bs_ids, target_bs_id=0):
    """
    4. 墙面可视化
    """
    print("Generating Plotly 3D visualization...")
    fig = go.Figure()

    # 取出仅属于目标基站的点
    target_bs_mask = (bs_ids == target_bs_id)

    # 绘制属于该目标基站的离群点
    wall_points_set = set(np.concatenate(wall_inliers_list)) if wall_inliers_list else set()
    outlier_indices = [i for i in range(len(original_xyz)) if i not in wall_points_set and target_bs_mask[i]]
    outliers_xyz = original_xyz[outlier_indices]
    
    if len(outliers_xyz) > 0:
        fig.add_trace(go.Scatter3d(
            x=outliers_xyz[:, 0], y=outliers_xyz[:, 1], z=outliers_xyz[:, 2],
            mode='markers', marker=dict(size=2, color='gray', opacity=0.3), name='Other Points'
        ))

    # 绘制有效墙面 (单基站视角下)
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown']
    for i, inliers in enumerate(wall_inliers_list):
        # 仅保留属于该目标基站的墙面点
        wall_pts_idx = [idx for idx in inliers if target_bs_mask[idx]]
        if not wall_pts_idx:
            continue
            
        wall_pts = original_xyz[wall_pts_idx]
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter3d(
            x=wall_pts[:, 0], y=wall_pts[:, 1], z=wall_pts[:, 2],
            mode='markers', marker=dict(size=4, color=color, opacity=0.9), name=f'Wall {i}'
        ))

    # 仅高亮显示目标基站
    bs_xyz = np.array([bs_positions[target_bs_id]])
    fig.add_trace(go.Scatter3d(
        x=bs_xyz[:, 0], y=bs_xyz[:, 1], z=bs_xyz[:, 2],
        mode='markers+text', marker=dict(size=10, color='red', symbol='diamond'),
        text=[f"BS_{target_bs_id}"], textposition="top center", name='Target BS'
    ))

    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        title=f"ISAC NLoS Wall Extraction for BS_{target_bs_id}",
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.write_html(output_html)
    print(f"Visualization saved to {output_html}")

def main():
    h5_path = 'isac_pure_sensing.h5'
    output_html = 'wall_extraction_single.html'
    bs_positions = [
        [40, -40, 15],  # BS_0
        [0, 60, 15],    # BS_1
        [-60, 0, 15]    # BS_2
    ]
    TARGET_BS = 0
    
    try:
        pcd, original_xyz, bs_ids = load_pointcloud(h5_path, frame_idx=0, power_thresh=1e-50)
    except FileNotFoundError:
        print(f"Error: {h5_path} not found.")
        return

    if len(original_xyz) == 0:
        print("No valid points found to process.")
        return

    plane_models, plane_inliers, _ = extract_planes_ransac(pcd)
    wall_models, wall_inliers = filter_wall_planes(plane_models, plane_inliers)

    print("\n--- Extracted Wall Statistics ---")
    for i, (model, inliers) in enumerate(zip(wall_models, wall_inliers)):
        a, b, c, d = model
        print(f"Wall {i}: Equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0 | Inliers: {len(inliers)}")

    visualize_planes(original_xyz, wall_inliers, bs_positions, output_html, bs_ids, target_bs_id=TARGET_BS)

if __name__ == '__main__':
    main()