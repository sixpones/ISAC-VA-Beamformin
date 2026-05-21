import h5py
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import random

def load_pointcloud(h5_path, frame_idx=0, power_thresh=1e-50):
    """
    1. 读取点云
    从 H5 文件读取指定帧的点云数据，并根据功率过滤无效点。
    """
    print(f"Loading point cloud from frame {frame_idx} in {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        # shape: (N_frame, 4096, 5) -> [x, y, z, bs_id, power]
        pcl_data = f['sensing_pcl'][frame_idx]
    
    # 过滤无效点
    valid_mask = pcl_data[:, 4] > power_thresh
    valid_points = pcl_data[valid_mask]
    
    # 提取坐标 [x, y, z]
    xyz = valid_points[:, 0:3]
    print(f"Loaded {len(xyz)} valid points (power > {power_thresh}).")
    
    # 转换为 Open3D 格式
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd, xyz

def extract_planes_ransac(pcd, distance_threshold=0.5, ransac_n=3, num_iterations=1000):
    """
    2. RANSAC 平面提取
    循环提取多个平面，直到剩余点数量过少。
    """
    print("Starting RANSAC plane extraction...")
    xyz = np.asarray(pcd.points)
    remaining_indices = np.arange(len(xyz))
    
    plane_models = []
    plane_inliers_indices = []
    
    # 设定循环终止条件：剩余点数较少或无法再拟合
    while len(remaining_indices) > 50:
        pcd_tmp = pcd.select_by_index(remaining_indices)
        
        # 提取平面
        plane_model, inliers_tmp = pcd_tmp.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        if len(inliers_tmp) == 0:
            break
            
        # 映射回原始点的索引
        original_inliers = remaining_indices[inliers_tmp]
        
        plane_models.append(plane_model)
        plane_inliers_indices.append(original_inliers)
        
        # 更新剩余点的索引
        remaining_indices = np.delete(remaining_indices, inliers_tmp)
        
    print(f"Extracted {len(plane_models)} raw planes.")
    return plane_models, plane_inliers_indices, remaining_indices

def filter_wall_planes(plane_models, plane_inliers_indices):
    """
    3. 地面过滤 & 4. 主反射墙筛选
    删除可能为地面的平面 (法向量接近 z 轴) 及点数较少的小平面。
    """
    print("Filtering wall planes...")
    valid_models = []
    valid_inliers = []
    
    for i, (model, inliers) in enumerate(zip(plane_models, plane_inliers_indices)):
        a, b, c, d = model
        inlier_num = len(inliers)
        
        # 条件 1：地面过滤 (法向量Z分量绝对值 > 0.9)
        if abs(c) > 0.9:
            continue
            
        # 条件 2：面积(点数)过滤 (inlier_num > 300)
        if inlier_num <= 300:
            continue
            
        valid_models.append(model)
        valid_inliers.append(inliers)
        
    print(f"Kept {len(valid_models)} valid wall planes after filtering.")
    return valid_models, valid_inliers

def visualize_planes(original_xyz, wall_inliers_list, bs_positions, output_html):
    """
    6. 墙面可视化
    使用 Plotly 绘制 3D 图像：原始点云、各颜色墙面、基站位置。
    """
    print("Generating Plotly 3D visualization...")
    fig = go.Figure()

    # 绘制原始点云 (灰色，带有透明度以不遮挡墙面)
    # 提取所有不属于任何有效墙面的离群点
    wall_points_set = set(np.concatenate(wall_inliers_list)) if wall_inliers_list else set()
    outlier_indices = [i for i in range(len(original_xyz)) if i not in wall_points_set]
    outliers_xyz = original_xyz[outlier_indices]
    
    if len(outliers_xyz) > 0:
        fig.add_trace(go.Scatter3d(
            x=outliers_xyz[:, 0], y=outliers_xyz[:, 1], z=outliers_xyz[:, 2],
            mode='markers',
            marker=dict(size=2, color='gray', opacity=0.3),
            name='Other Points'
        ))

    # 绘制墙面 (不同颜色)
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown']
    for i, inliers in enumerate(wall_inliers_list):
        wall_pts = original_xyz[inliers]
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter3d(
            x=wall_pts[:, 0], y=wall_pts[:, 1], z=wall_pts[:, 2],
            mode='markers',
            marker=dict(size=4, color=color, opacity=0.9),
            name=f'Wall {i}'
        ))

    # 绘制基站位置 (红色 diamond，带文字标注)
    bs_xyz = np.array(bs_positions)
    fig.add_trace(go.Scatter3d(
        x=bs_xyz[:, 0], y=bs_xyz[:, 1], z=bs_xyz[:, 2],
        mode='markers+text',
        marker=dict(size=8, color='red', symbol='diamond'),
        text=[f"BS_{i}" for i in range(len(bs_positions))],
        textposition="top center",
        name='Base Stations'
    ))

    # 设置布局
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title="ISAC NLoS Wall Extraction (RANSAC)",
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.write_html(output_html)
    print(f"Visualization saved to {output_html}")


def main():
    # 参数配置
    h5_path = 'isac_pure_sensing.h5'
    output_html = 'wall_extraction.html'
    bs_positions = [
        [40, -40, 15],
        [0, 60, 15],
        [-60, 0, 15]
    ]
    
    # 1. 读取点云
    try:
        pcd, original_xyz = load_pointcloud(h5_path, frame_idx=0, power_thresh=1e-50)
    except FileNotFoundError:
        print(f"Error: {h5_path} not found. Please ensure the dataset exists.")
        return

    if len(original_xyz) == 0:
        print("No valid points found to process.")
        return

    # 2. RANSAC 提取所有可能平面
    plane_models, plane_inliers, _ = extract_planes_ransac(
        pcd,
        distance_threshold=0.5,
        ransac_n=3,
        num_iterations=1000
    )

    # 3 & 4. 过滤非墙面(地面)与过小的平面
    wall_models, wall_inliers = filter_wall_planes(plane_models, plane_inliers)

    # 5. 输出信息
    print("\n--- Extracted Wall Statistics ---")
    for i, (model, inliers) in enumerate(zip(wall_models, wall_inliers)):
        a, b, c, d = model
        inlier_num = len(inliers)
        print(f"Wall {i}:")
        print(f"  Equation:  {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
        print(f"  Normal:    [{a:.4f}, {b:.4f}, {c:.4f}]")
        print(f"  Inliers:   {inlier_num}")
        print("-" * 33)

    # 6. 可视化并保存
    visualize_planes(original_xyz, wall_inliers, bs_positions, output_html)


if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
