import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ase.io import read
import tempfile
from io import BytesIO

from atomks_visual.read_xsf import read_xsf
from atomks_visual.read_den import read_den

# 示例数据生成函数
def generate_example_data():
    density = np.random.rand(100, 100, 100)
    lattice = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    position = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    return density, lattice, position

# 加载数据
def load_data():
    uploaded_density = st.sidebar.file_uploader("上传 3D 电荷密度数组")
    uploaded_crystal = st.sidebar.file_uploader("上传晶体")
    if uploaded_density and uploaded_crystal:
        if "den" in uploaded_density.name:
            density_data = uploaded_density.read().decode("utf-8").splitlines()
            density = read_den(density_data)
        elif "xsf" in uploaded_density.name:
            density_data = uploaded_density.read().decode("utf-8").splitlines()
            density = read_xsf(density_data)
        else:
            st.sidebar.error("不支持的文件格式")
            return None, None, None


        # 将上传的文件保存为临时文件并保留文件名
        original_name = uploaded_crystal.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{original_name}") as temp_file:
            temp_file.write(uploaded_crystal.read())
            temp_file_path = temp_file.name
        uploaded_crystal = read(temp_file_path)
        lattice = uploaded_crystal.cell
        position = uploaded_crystal.get_positions()
    else:
        st.sidebar.warning("未上传数据，使用示例数据")
        density, lattice, position = generate_example_data()
    return density, lattice, position

# 绘制晶格 3D 可视化
def plot_lattice_3d(lattice):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    origin = np.array([0, 0, 0])
    for vec in lattice:
        ax.quiver(*origin, *vec, length=1.0, normalize=True, color='r')
    ax.set_xlabel("X 轴")
    ax.set_ylabel("Y 轴")
    ax.set_zlabel("Z 轴")
    ax.set_title("晶格 3D 可视化")
    st.pyplot(fig)

# 绘制动态交互式 3D 晶格与等高线图
def plot_interactive_3d(density, lattice, position):
    axis_2d = st.selectbox("选择二维平面轴", ["XY", "XZ", "YZ"], key="2d_axis")
    index_2d = st.slider(f"选择 {axis_2d} 平面的索引", 0, density.shape[2 if axis_2d == "XY" else 1 if axis_2d == "XZ" else 0] - 1, 0, key="2d_index")
    plane_data, x, y, z = extract_plane_data(density, lattice, axis_2d, index_2d)
    fig = create_3d_plot(lattice, position, plane_data, x, y, z, axis_2d)
    st.plotly_chart(fig)

# 提取二维平面数据
def extract_plane_data(density, lattice, axis_2d, index_2d):
    if axis_2d == "XY":
        plane_data = density[:, :, index_2d]
        x, y = np.meshgrid(
            np.linspace(0, lattice[0, 0], plane_data.shape[0]),
            np.linspace(0, lattice[1, 1], plane_data.shape[1]),
            indexing="ij"
        )
        z = np.full_like(plane_data, index_2d / density.shape[2] * lattice[2, 2])
    elif axis_2d == "XZ":
        plane_data = density[:, index_2d, :]
        x, z = np.meshgrid(
            np.linspace(0, lattice[0, 0], plane_data.shape[0]),
            np.linspace(0, lattice[2, 2], plane_data.shape[1]),
            indexing="ij"
        )
        y = np.full_like(plane_data, index_2d / density.shape[1] * lattice[1, 1])
    else:  # YZ
        plane_data = density[index_2d, :, :]
        y, z = np.meshgrid(
            np.linspace(0, lattice[1, 1], plane_data.shape[0]),
            np.linspace(0, lattice[2, 2], plane_data.shape[1]),
            indexing="ij"
        )
        x = np.full_like(plane_data, index_2d / density.shape[0] * lattice[0, 0])
    return plane_data, x, y, z

# 创建 3D 图形
def create_3d_plot(lattice, position, plane_data, x, y, z, axis_2d):
    fig = go.Figure()
    lattice_points = np.array([
        [0, 0, 0],
        lattice[0],
        lattice[1],
        lattice[2],
        lattice[0] + lattice[1],
        lattice[0] + lattice[2],
        lattice[1] + lattice[2],
        lattice[0] + lattice[1] + lattice[2]
    ])
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7)
    ]
    for edge in edges:
        fig.add_trace(go.Scatter3d(
            x=[lattice_points[edge[0], 0], lattice_points[edge[1], 0]],
            y=[lattice_points[edge[0], 1], lattice_points[edge[1], 1]],
            z=[lattice_points[edge[0], 2], lattice_points[edge[1], 2]],
            mode='lines',
            line=dict(color='blue', width=2),
            showlegend=False
        ))
    for pos in position:
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='markers',
            marker=dict(size=5, color='red'),
            showlegend=False
        ))
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        surfacecolor=plane_data,
        colorscale="Viridis",
        cmin=np.min(plane_data),
        cmax=np.max(plane_data),
        showscale=True,
        colorbar=dict(title="电荷密度")
    ))
    camera = dict(
        eye=dict(
            x=2 if axis_2d == "YZ" else 0,
            y=2 if axis_2d == "XZ" else 0,
            z=2 if axis_2d == "XY" else 0
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="X 轴",
            yaxis_title="Y 轴",
            zaxis_title="Z 轴",
            aspectmode="data"
        ),
        title=f"晶格 3D 可视化与 {axis_2d} 平面等高线图",
        scene_camera=camera
    )
    return fig

# 绘制一维直线密度图
def plot_1d_density(density):
    axis_1d = st.selectbox("选择一维直线轴", ["X", "Y", "Z"], key="1d_axis")
    index1_1d = st.slider(f"选择 {axis_1d} 轴的固定索引 1", 0, density.shape[1 if axis_1d == "X" else 0 if axis_1d == "Y" else 0] - 1, 0, key="1d_index1")
    index2_1d = st.slider(f"选择 {axis_1d} 轴的固定索引 2", 0, density.shape[2 if axis_1d == "X" else 2 if axis_1d == "Y" else 1] - 1, 0, key="1d_index2")
    if axis_1d == "X":
        line_data = density[:, index1_1d, index2_1d]
    elif axis_1d == "Y":
        line_data = density[index1_1d, :, index2_1d]
    else:
        line_data = density[index1_1d, index2_1d, :]
    fig, ax = plt.subplots()
    ax.plot(line_data)
    ax.set_xlabel("位置索引")
    ax.set_ylabel("密度值")
    st.pyplot(fig)

# 主程序
st.title("3D 电荷密度可视化")
density, lattice, position = load_data()
st.subheader("晶格矩阵")
# st.write(lattice)
# st.subheader("晶格 3D 可视化")
# plot_lattice_3d(lattice)
st.subheader("晶格 3D 可视化（带等高线）")
plot_interactive_3d(density, lattice, position)
st.subheader("一维直线密度图")
plot_1d_density(density)
