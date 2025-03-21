import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ase.io import read
import tempfile
from io import BytesIO
import pandas as pd

from atomks_visual.read_xsf import read_xsf
from atomks_visual.read_den import read_den


# 示例数据生成函数
def generate_example_data():
    density = np.random.rand(100, 100, 100)
    lattice = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    position = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    return density, lattice, position


def load_density(upload_file):
    if "den" in upload_file.name:
        density_data = upload_file.read().decode("utf-8").splitlines()
        density = read_den(density_data)
    elif "xsf" in upload_file.name:
        density_data = upload_file.read().decode("utf-8").splitlines()
        density = read_xsf(density_data)
    else:
        st.sidebar.error("不支持的文件格式")
        return None
    return density


def load_crystal(upload_file):
    # 将上传的文件保存为临时文件并保留文件名
    original_name = upload_file.name
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f"_{original_name}"
    ) as temp_file:
        temp_file.write(upload_file.read())
        temp_file_path = temp_file.name
    uploaded_crystal = read(temp_file_path)
    lattice = uploaded_crystal.cell
    position = uploaded_crystal.get_positions()
    return lattice, position


# 加载数据
def load_data():
    uploaded_crystal = st.sidebar.file_uploader("上传晶体")

    if uploaded_crystal:
        lattice, position = load_crystal(uploaded_crystal)
        return lattice, position
    else:
        st.sidebar.warning("未上传数据，使用示例数据")
        density, lattice, position = generate_example_data()
    return lattice, position


# 绘制晶格 3D 可视化
def plot_lattice_3d(lattice):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    origin = np.array([0, 0, 0])
    for vec in lattice:
        ax.quiver(*origin, *vec, length=1.0, normalize=True, color="r")
    ax.set_xlabel("X 轴")
    ax.set_ylabel("Y 轴")
    ax.set_zlabel("Z 轴")
    ax.set_title("晶格 3D 可视化")
    st.pyplot(fig)


# 绘制动态交互式 3D 晶格与等高线图
def plot_interactive_3d(density, lattice, position):
    axis_2d = st.selectbox("选择二维平面轴", ["XY", "XZ", "YZ"], key="2d_axis")
    index_2d = st.slider(
        f"选择 {axis_2d} 平面的索引",
        0,
        density.shape[2 if axis_2d == "XY" else 1 if axis_2d == "XZ" else 0] - 1,
        0,
        key="2d_index",
    )
    plane_data, x, y, z = extract_plane_data(density, lattice, axis_2d, index_2d)
    create_3d_plot(lattice, position, plane_data, x, y, z, axis_2d)
    # st.plotly_chart(fig)


# 提取二维平面数据
def extract_plane_data(density, lattice, axis_2d, index_2d):
    if axis_2d == "XY":
        plane_data = density[:, :, index_2d]
        x, y = np.meshgrid(
            np.linspace(0, lattice[0, 0], plane_data.shape[0]),
            np.linspace(0, lattice[1, 1], plane_data.shape[1]),
            indexing="ij",
        )
        z = np.full_like(plane_data, index_2d / density.shape[2] * lattice[2, 2])
    elif axis_2d == "XZ":
        plane_data = density[:, index_2d, :]
        x, z = np.meshgrid(
            np.linspace(0, lattice[0, 0], plane_data.shape[0]),
            np.linspace(0, lattice[2, 2], plane_data.shape[1]),
            indexing="ij",
        )
        y = np.full_like(plane_data, index_2d / density.shape[1] * lattice[1, 1])
    else:  # YZ
        plane_data = density[index_2d, :, :]
        y, z = np.meshgrid(
            np.linspace(0, lattice[1, 1], plane_data.shape[0]),
            np.linspace(0, lattice[2, 2], plane_data.shape[1]),
            indexing="ij",
        )
        x = np.full_like(plane_data, index_2d / density.shape[0] * lattice[0, 0])
    return plane_data, x, y, z


# 创建 3D 图形
def create_3d_plot(lattice, position, plane_data, x, y, z, axis_2d):
    fig = go.Figure()
    lattice_points = np.array(
        [
            [0, 0, 0],
            lattice[0],
            lattice[1],
            lattice[2],
            lattice[0] + lattice[1],
            lattice[0] + lattice[2],
            lattice[1] + lattice[2],
            lattice[0] + lattice[1] + lattice[2],
        ]
    )
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ]
    for edge in edges:
        fig.add_trace(
            go.Scatter3d(
                x=[lattice_points[edge[0], 0], lattice_points[edge[1], 0]],
                y=[lattice_points[edge[0], 1], lattice_points[edge[1], 1]],
                z=[lattice_points[edge[0], 2], lattice_points[edge[1], 2]],
                mode="lines",
                line=dict(color="blue", width=2),
                showlegend=False,
            )
        )
    for pos in position:
        fig.add_trace(
            go.Scatter3d(
                x=[pos[0]],
                y=[pos[1]],
                z=[pos[2]],
                mode="markers",
                marker=dict(size=5, color="red"),
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=plane_data,
            colorscale="Viridis",
            cmin=np.min(plane_data),
            cmax=np.max(plane_data),
            showscale=True,
            colorbar=dict(title="电荷密度"),
        )
    )
    camera = dict(
        eye=dict(
            x=2 if axis_2d == "YZ" else 0,
            y=2 if axis_2d == "XZ" else 0,
            z=2 if axis_2d == "XY" else 0,
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="X 轴",
            yaxis_title="Y 轴",
            zaxis_title="Z 轴",
            aspectmode="data",
        ),
        title=f"晶格 3D 可视化与 {axis_2d} 平面等高线图",
        scene_camera=camera,
    )

    # 添加平面上的 z 轴电荷密度图，使用 surface 并将 z 轴设置为密度
    fig_density = go.Figure()
    fig_density.add_trace(
        go.Surface(
            z=plane_data,  # 将密度值作为 z 轴
            x=x,  # X 轴坐标
            y=y,  # Y 轴坐标
            colorscale="Viridis",
            cmin=np.min(plane_data),
            cmax=np.max(plane_data),
            showscale=True,
            colorbar=dict(title="电荷密度"),
        )
    )
    fig_density.update_layout(
        title=f"{axis_2d} 平面上的电荷密度分布 (Z 轴为密度)",
        scene=dict(
            xaxis_title="X 轴" if axis_2d in ["XY", "XZ"] else "Y 轴",
            yaxis_title="Y 轴" if axis_2d == "XY" else "Z 轴",
            zaxis_title="电荷密度",
            aspectmode="data",
        ),
    )

    # 使用三个列并列展示两个图
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig)
    with col2:
        st.plotly_chart(fig_density)


# 绘制一维直线密度图
def plot_1d_density(density):
    axis_1d = st.selectbox("选择一维直线轴", ["X", "Y", "Z"], key="1d_axis")
    index1_1d = st.slider(
        f"选择 {axis_1d} 轴的固定索引 1",
        0,
        density.shape[1 if axis_1d == "X" else 0 if axis_1d == "Y" else 0] - 1,
        0,
        key="1d_index1",
    )
    index2_1d = st.slider(
        f"选择 {axis_1d} 轴的固定索引 2",
        0,
        density.shape[2 if axis_1d == "X" else 2 if axis_1d == "Y" else 1] - 1,
        0,
        key="1d_index2",
    )
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


def dynamic_table():
    # 初始化表格数据
    if "table_data" not in st.session_state:
        st.session_state.table_data = pd.DataFrame(
            columns=["Label", "电荷密度文件", "density"]
        )

    # 添加新条目
    def add_new_entry():
        new_row = {
            "Label": f"Label {len(st.session_state.table_data) + 1}",
            "电荷密度文件": "",
            "density": None,
        }
        st.session_state.table_data = pd.concat(
            [st.session_state.table_data, pd.DataFrame([new_row])], ignore_index=True
        )

    # 显示表格
    st.title("动态表格：电荷密度上传")
    st.write("点击加号按钮添加新的条目")

    # 添加新条目按钮
    if st.button("➕ 添加新条目"):
        add_new_entry()

    # 动态表格显示
    for index, row in st.session_state.table_data.iterrows():
        col1, col2 = st.columns([1, 3])
        with col1:
            label = st.text_input(
                f"Label {index + 1}", value=row["Label"], key=f"label_{index}"
            )
            st.session_state.table_data.at[index, "Label"] = label
        with col2:
            uploaded_file = st.file_uploader(
                f"上传电荷密度文件 {index + 1}", key=f"file_{index}"
            )
            if uploaded_file is not None:
                density = load_density(uploaded_file)
                if density is not None:
                    st.session_state.table_data.at[index, "density"] = density


def plot_interactive_3d_from_table(lattice, position):
    # 从表格中获取数据
    density_data = st.session_state.table_data["density"]
    # lattice_data = st.session_state.table_data["lattice"]
    # position_data = st.session_state.table_data["position"]

    # 选择要显示的数据
    if st.session_state.table_data.empty:
        st.warning("表格中没有数据可供选择")
        return
    selected_data = st.selectbox(
        "选择要显示的数据", st.session_state.table_data["Label"]
    )

    # 从表格数据中获取所选数据的索引
    index = st.session_state.table_data[
        st.session_state.table_data["Label"] == selected_data
    ].index[0]

    # 绘制交互式 3D 图
    plot_interactive_3d(density_data[index], lattice, position)


def plot_density_difference(lattice, position):
    # 检查表格中是否有足够的数据
    if st.session_state.table_data.empty or len(st.session_state.table_data) < 2:
        st.warning("表格中需要至少两个条目才能计算电荷密度差值")
        return

    # 选择两个 Label
    st.subheader("选择两个 Label 计算电荷密度差值")
    labels = st.session_state.table_data["Label"].tolist()
    label1 = st.selectbox("选择第一个 Label", labels, key="label1")
    label2 = st.selectbox("选择第二个 Label", labels, key="label2")

    # 获取对应的电荷密度数据
    index1 = st.session_state.table_data[
        st.session_state.table_data["Label"] == label1
    ].index[0]
    index2 = st.session_state.table_data[
        st.session_state.table_data["Label"] == label2
    ].index[0]
    density1 = st.session_state.table_data.at[index1, "density"]
    density2 = st.session_state.table_data.at[index2, "density"]

    # 检查电荷密度数据是否有效
    if density1 is None or density2 is None:
        st.error("所选 Label 的电荷密度数据无效，请确保已上传有效的电荷密度文件")
        return

    # 计算电荷密度差值
    density_diff = density1 - density2

    # 绘制 2D 平面密度差值图
    st.subheader("2D 平面电荷密度差值图")
    axis_2d = st.selectbox("选择二维平面轴", ["XY", "XZ", "YZ"], key="diff_2d_axis")
    index_2d = st.slider(
        f"选择 {axis_2d} 平面的索引",
        0,
        density_diff.shape[2 if axis_2d == "XY" else 1 if axis_2d == "XZ" else 0] - 1,
        0,
        key="diff_2d_index",
    )

    # 提取平面数据
    plane_data, x, y, z = extract_plane_data(density_diff, lattice, axis_2d, index_2d)

    # 调用 create_3d_plot 绘制 3D 图
    st.subheader("3D 电荷密度差值图")
    create_3d_plot(lattice, position, plane_data, x, y, z, axis_2d)


def plot_all_1d_densities():
    # 检查表格中是否有数据
    if st.session_state.table_data.empty:
        st.warning("表格中没有数据可供绘制")
        return

    # 选择一维直线轴
    st.subheader("绘制所有 1D 电荷密度")
    axis_1d = st.selectbox("选择一维直线轴", ["X", "Y", "Z"], key="all_1d_axis")
    index1_1d = st.slider(
        f"选择 {axis_1d} 轴的固定索引 1",
        0,
        st.session_state.table_data["density"]
        .iloc[0]
        .shape[1 if axis_1d == "X" else 0 if axis_1d == "Y" else 0]
        - 1,
        0,
        key="all_1d_index1",
    )
    index2_1d = st.slider(
        f"选择 {axis_1d} 轴的固定索引 2",
        0,
        st.session_state.table_data["density"]
        .iloc[0]
        .shape[2 if axis_1d == "X" else 2 if axis_1d == "Y" else 1]
        - 1,
        0,
        key="all_1d_index2",
    )

    # 创建绘图
    fig, ax = plt.subplots()
    for _, row in st.session_state.table_data.iterrows():
        label = row["Label"]
        density = row["density"]

        # 提取 1D 数据
        if axis_1d == "X":
            line_data = density[:, index1_1d, index2_1d]
        elif axis_1d == "Y":
            line_data = density[index1_1d, :, index2_1d]
        else:  # Z
            line_data = density[index1_1d, index2_1d, :]

        # 绘制每个 Label 的 1D 数据
        ax.plot(line_data, label=label)

    # 设置图例和标签
    ax.set_xlabel("Position index")
    ax.set_ylabel("Density in a.u.")
    ax.set_title(f"All labeled 1D charge density in ({axis_1d} axis)")
    ax.legend()

    # 显示图形
    st.pyplot(fig)


# 主程序
# 调用动态表格函数
dynamic_table()

st.title("3D 电荷密度可视化")
lattice, position = load_data()
st.subheader("晶格矩阵")
# st.write(lattice)
# st.subheader("晶格 3D 可视化")
# plot_lattice_3d(lattice)
plot_interactive_3d_from_table(lattice, position)

plot_density_difference(lattice, position)
# st.subheader("晶格 3D 可视化（带等高线）")
# plot_interactive_3d(density, lattice, position)
st.subheader("一维直线密度图")
plot_all_1d_densities()
