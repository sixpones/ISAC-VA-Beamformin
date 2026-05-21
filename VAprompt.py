import numpy as np
import plotly.graph_objects as go
import json

# 复用之前写好的 RANSAC 墙面提取逻辑
from wall_extraction import load_pointcloud, extract_planes_ransac, filter_wall_planes


def mirror_point_to_plane(p, plane_model):
    """
    计算点关于平面的镜像点 (Virtual Anchor)
    利用数学公式: p' = p - 2 * (a*x + b*y + c*z + d) * n
    其中:
      p = [x, y, z] 为基站坐标
      n = [a, b, c] 为平面法向量，平面方程为 ax + by + cz + d = 0
    """
    p = np.array(p, dtype=float)
    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=float)
    
    # 归一化法向量
    norm = np.linalg.norm(normal)
    normal = normal / norm
    d = d / norm
    
    # 距离项: (a*x + b*y + c*z + d)
    distance = np.dot(normal, p) + d
    
    # 镜像点: p' = p - 2 * distance * n
    p_mirror = p - 2 * distance * normal
    return p_mirror


def compute_virtual_anchors(bs_positions, plane_models, original_xyz, wall_inliers, max_dist=50.0):
    """
    为所有的基站和提取出的每一面墙计算对应的 Virtual Anchor (VA)
    包含物理约束剪枝 (Pruning)：
    1. 朝向过滤 (Orientation Check): v · n < 0 则剔除
    2. 距离过滤 (Distance Check): 距离 > max_dist 则剔除
    
    返回包含字典的列表，包含相关信息以便打印和保存。
    """
    virtual_anchors = []
    
    # 建立一个显式的映射关系，让代码“确定”每个基站去寻找与其对应的实体墙面
    # 这里通过字典硬编码基站索引与其对应的墙面索引（RANSAC提取后的列表索引）
    bs_to_walls_map = {
        0: [0, 1],  # BS_0 只针对 Wall 0 和 Wall 1 计算 VA
        1: [1, 2],  # 根据实际结构，如果 BS_1 对应 Wall 1/2
        2: [0, 2]   # 根据实际结构，如果 BS_2 对应 Wall 0/2
    }

    for bs_id, bs_pos in enumerate(bs_positions):
        # 仅对 BS_0 寻找虚拟锚点
        if bs_id != 0:
            continue
            
        # 取得该基站需要匹配的目标墙面列表
        target_walls = bs_to_walls_map.get(bs_id, [])
        if not target_walls:
            continue
            
        for plane_id, (plane_model, inliers) in enumerate(zip(plane_models, wall_inliers)):
            # ⭐ 核心过滤：只针对给该基站分配的墙体进行计算
            if plane_id not in target_walls:
                continue
            
            # 取得墙面上点云的中心作为墙面代表点 p_wall
            p_wall = np.mean(original_xyz[inliers], axis=0)
            
            # 向量 v: 从墙面指向基站
            v = np.array(bs_pos) - p_wall
            normal = np.array(plane_model[:3])
            
            # 如果 v·n < 0，说明找到的法向量指向了背离基站的一侧
            # 这里不能直接 continue 剔除，而是需要将法线强制“掰向”基站
            if np.dot(v, normal) < 0:
                plane_model = [-plane_model[0], -plane_model[1], -plane_model[2], -plane_model[3]]
                normal = -normal
            
            # 【剪枝 2：距离过滤 Distance Check】
            # 如果基站到这面墙的距离超过阈值（如 50m），则丢弃
            # dist = np.linalg.norm(v)
            # if dist > max_dist:
            #     continue

            va_pos = mirror_point_to_plane(bs_pos, plane_model)
            
            va_info = {
                "bs_id": bs_id,
                "plane_id": plane_id,
                "va_position": va_pos.tolist(),
                "plane_normal": plane_model[:3],
                "plane_d": plane_model[3]
            }
            virtual_anchors.append(va_info)
            
    return virtual_anchors


def visualize_va(original_xyz, wall_inliers_list, bs_positions, virtual_anchors, output_html):
    """
    使用 Plotly 绘制 3D 结果图像。
    包含：墙面点云、红色基站、蓝色 VA、BS->VA 虚线连线
    """
    fig = go.Figure()

    # 1. 绘制墙面点云
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown']
    for i, inliers in enumerate(wall_inliers_list):
        wall_pts = original_xyz[inliers]
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter3d(
            x=wall_pts[:, 0], y=wall_pts[:, 1], z=wall_pts[:, 2],
            mode='markers',
            marker=dict(size=3, color=color, opacity=0.8),
            name=f'Wall {i}',
            legendgroup='Walls'
        ))

    # 2. 绘制基站位置 (红色 diamond，加注索引文字)
    bs_xyz = np.array(bs_positions)
    fig.add_trace(go.Scatter3d(
        x=bs_xyz[:, 0], y=bs_xyz[:, 1], z=bs_xyz[:, 2],
        mode='markers+text',
        marker=dict(size=8, color='red', symbol='diamond'),
        text=[f"BS_{i}" for i in range(len(bs_positions))],
        textposition="top center",
        name='Base Stations'
    ))

    # 3. 绘制 VA (蓝色 circle) 以及连线 (虚线)
    for va in virtual_anchors:
        bs_id = va["bs_id"]
        plane_id = va["plane_id"]
        bs_pos = bs_positions[bs_id]
        va_pos = np.array(va["va_position"])
        
        # 绘制 VA 点
        fig.add_trace(go.Scatter3d(
            x=[va_pos[0]], y=[va_pos[1]], z=[va_pos[2]],
            mode='markers',
            marker=dict(size=6, color='blue', symbol='circle'),
            name=f'VA (BS{bs_id}-Wall{plane_id})',
            showlegend=False
        ))
        
        # 绘制 BS -> VA 的虚线连线
        fig.add_trace(go.Scatter3d(
            x=[bs_pos[0], va_pos[0]], 
            y=[bs_pos[1], va_pos[1]], 
            z=[bs_pos[2], va_pos[2]],
            mode='lines',
            line=dict(color='gray', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='none'
        ))

    # 布局设置
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        title="NLoS Virtual Anchors (VA) and Beams for ISAC",
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fig.write_html(output_html)
    print(f"VA visualization saved to {output_html}")


def save_va_json(virtual_anchors, output_file):
    """
    保存 VA 结果到指定 JSON 文件，仅提取要求字段
    """
    output_data = [
        {
            "bs_id": va["bs_id"],
            "plane_id": va["plane_id"],
            "va_position": va["va_position"]
        }
        for va in virtual_anchors
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Virtual Anchors JSON data saved to {output_file}")


def main():
    h5_path = 'isac_pure_sensing.h5'
    output_html = "va_visualization.html"
    output_json = "virtual_anchor.json"
    
    bs_positions = [
        [40, -40, 15],
        [0, 60, 15],
        [-60, 0, 15]
    ]
    
    # 1. 加载点云，提取墙面方程
    print("--- Step 1: Extracting Walls ---")
    pcd, original_xyz = load_pointcloud(h5_path, frame_idx=0, power_thresh=1e-50)
    plane_models, plane_inliers, _ = extract_planes_ransac(pcd, distance_threshold=0.5, ransac_n=3, num_iterations=1000)
    wall_models, wall_inliers = filter_wall_planes(plane_models, plane_inliers)

    # 2. 计算 Virtual Anchors (并进行剪枝 Pruning)
    print("\n--- Step 2: Computing Virtual Anchors ---")
    virtual_anchors = compute_virtual_anchors(bs_positions, wall_models, original_xyz, wall_inliers)
    
    for va in virtual_anchors:
        print(f"BS {va['bs_id']} -> Wall {va['plane_id']}:")
        print(f"  Plane Normal : [{va['plane_normal'][0]:.4f}, {va['plane_normal'][1]:.4f}, {va['plane_normal'][2]:.4f}]")
        print(f"  VA Position  : [{va['va_position'][0]:.4f}, {va['va_position'][1]:.4f}, {va['va_position'][2]:.4f}]")

    # 3. 可视化
    print("\n--- Step 3: Visualizing VA ---")
    visualize_va(original_xyz, wall_inliers, bs_positions, virtual_anchors, output_html)
    
    # 4. 保存 JSON
    save_va_json(virtual_anchors, output_json)


if __name__ == '__main__':
    main()
