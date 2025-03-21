import numpy as np


def read_xsf(file: str):
    # 打开并读取XSF文件
    density_data = []
    continue_count = 0
    start_reading_density = False
    read_grid_flag = False

    for line in file:
        if continue_count > 0:
            continue_count -= 1
            continue

        if read_grid_flag:
            grid = list(map(int, line.strip().split()))
            continue_count = 4
            read_grid_flag = False
            continue

        # 检查是否达到电荷密度数据的开始部分
        if "BEGIN_DATAGRID_3D_UNKNOWN" in line:
            start_reading_density = True
            read_grid_flag = True
            continue

        # 如果开始读取电荷密度数据
        if start_reading_density:
            # 忽略用于定义网格的五行
            if "END_DATAGRID_3D" in line:
                break  # 结束电荷密度数据读取
            # if line.strip().isdigit() or 'BEGIN' in line:
            #     continue
            # 读取电荷密度值并添加到列表中
            density_values = line.strip().split()
            density_values = list(map(float, density_values))
            density_data.extend(density_values)

    # 转换列表为NumPy数组，并根据网格大小重塑数组
    density = np.array(density_data).reshape(grid)

    return density[0:-1, 0:-1, 0:-1]
