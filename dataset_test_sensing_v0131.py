"""Visualize the generated ISAC sensing point cloud.

读取 dataset_gen_sensing_v0131.py 生成的 isac_pure_sensing.h5，
把建筑、基站和点云一起画到 Plotly 3D HTML 中。
"""

import h5py
import numpy as np
import plotly.graph_objects as go
import os

# 默认读取生成脚本输出的 H5 文件。
H5_FILE = "isac_pure_sensing.h5"

# 基站位置 (与生成脚本一致)
BS_POSITIONS = [
    {'pos': [40, -40, 15], 'name': 'BS_0 (Corner/Plaza)'},
    {'pos': [0, 60, 15], 'name': 'BS_1 (North)'},
    {'pos': [-60, 0, 15], 'name': 'BS_2 (West)'}
]

# [修改点] 建筑物定义 (增加 height 属性)
# 尺寸 60x60, 高度分别为 25, 30, 35
BUILDINGS = [
    {'center': [-40, 40], 'name': 'Building A (NW, 25m)', 'height': 25},
    {'center': [40, 40],  'name': 'Building B (NE, 30m)', 'height': 30},
    {'center': [-40, -40],'name': 'Building C (SW, 35m)', 'height': 35}
]

def draw_cube(fig, center_x, center_y, size=60, height=30, color='gray', opacity=0.2, name='Building'):
    """在 Plotly 中画一个建筑立方体。

    Plotly Mesh3d 需要顶点坐标和三角面索引；这里把一个长方体拆成 12 个三角形。
    """
    x_min, x_max = center_x - size / 2, center_x + size / 2
    y_min, y_max = center_y - size / 2, center_y + size / 2
    z_min, z_max = 0, height

    # 8个顶点
    x = [x_min, x_min, x_max, x_max, x_min, x_min, x_max, x_max]
    y = [y_min, y_max, y_max, y_min, y_min, y_max, y_max, y_min]
    z = [z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max]

    # 12个三角形面 (i, j, k 索引)
    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]

    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color=color,
        opacity=opacity,
        name=name,
        showscale=False,
        hoverinfo='name'
    ))


def view_metis_variable_height():
    print(f"读取数据集: {H5_FILE}...")
    if not os.path.exists(H5_FILE):
        print("❌ 文件不存在")
        return

    # H5 结构来自生成脚本：
    #   sensing_pcl[frame_id, point_id, :] = [x, y, z, bs_index, path_gain]
    # 当前只画第 0 帧；如果以后生成多帧，可以把 [0] 改成目标帧编号。
    with h5py.File(H5_FILE, 'r') as f:
        if 'sensing_pcl' not in f:
            print("❌ 文件中没有 sensing_pcl 数据集，请先重新运行生成脚本")
            return
        if f['sensing_pcl'].shape[0] == 0:
            print("❌ 数据集没有任何 frame，请先重新运行生成脚本")
            print("示例: ../.venv/bin/python dataset_gen_sensing_v0131.py --samples 100000 --max-depth 2")
            return
        pcl = f['sensing_pcl'][0]

    # 过滤无效点：生成脚本会用全 0 填充不足 4096 的位置。
    # 第 5 列 path_gain > 0 才表示真实路径点。
    mask = pcl[:, 4] > 1e-50
    pcl = pcl[mask]

    print(f"✅ 有效点数: {len(pcl)}")

    fig = go.Figure()

    # 1. 绘制建筑物。这里的楼高必须和生成脚本中的 buildings_config 保持一致。
    print("绘制不同高度的建筑物...")
    for b in BUILDINGS:
        draw_cube(fig,
                  center_x=b['center'][0],
                  center_y=b['center'][1],
                  height=b['height'],
                  name=b['name'])

    # 2. 绘制基站位置。红色菱形标记，方便和点云覆盖范围对照。
    for i, bs in enumerate(BS_POSITIONS):
        pos = bs['pos']
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='markers+text',
            marker=dict(size=12, color='red', symbol='diamond'),
            text=[f"BS_{i}"],
            textposition="top center",
            name=bs['name']
        ))

    # 3. 按 bs_index 拆分点云。不同 BS 用不同颜色显示。
    colors = ['#1f77b4', '#2ca02c', '#9467bd']  # 蓝, 绿, 紫
    labels = ['BS_0 Coverage', 'BS_1 Coverage', 'BS_2 Coverage']

    for i in range(3):
        idx = (pcl[:, 3] == i)
        pts = pcl[idx]
        if len(pts) > 0:
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(size=2, color=colors[i], opacity=0.8),
                name=labels[i],
                text=[f"Power: {p:.2e}" for p in pts[:, 4]],
                hovertemplate="X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<br>%{text}"
            ))

    # 4. 布局设置：固定坐标范围，保证不同运行结果之间视角/尺度可比较。
    fig.update_layout(
        title="METIS 2x2 Scenario (Variable Heights: 25m, 30m, 35m)",
        template="plotly_white",
        scene=dict(
            xaxis=dict(title="X", range=[-100, 100], showbackground=True, backgroundcolor="white", gridcolor="lightgray"),
            yaxis=dict(title="Y", range=[-100, 100], showbackground=True, backgroundcolor="white", gridcolor="lightgray"),
            # [修改点] Z轴稍微拉高一点，方便看高楼和鬼影
            zaxis=dict(title="Z", range=[0, 50], showbackground=True, backgroundcolor="white", gridcolor="lightgray"),
            aspectmode='data'
        ),
        width=1200, height=800,
        margin=dict(r=0, l=0, b=0, t=50)
    )

    output_file = "view_metis_v35.html"
    # 生成独立 HTML，浏览器直接打开即可交互旋转/缩放。
    fig.write_html(output_file)
    print(f"可视化已保存: {output_file}")


if __name__ == "__main__":
    view_metis_variable_height()
