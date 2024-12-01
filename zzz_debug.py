import numpy as np
import open3d as o3d
import laspy
import os

os.environ['DISPLAY'] = '1'

# 加载分析结果
pts_file = 'C:/Users/zhong/git/mmdetection3d/outputs/cicc/preds/lidar_scanning_20240808.json'
pts_file = 'C:/Users/zhong/git/mmdetection3d/outputs/cicc_bak/lidar_scanning_20240808.json'

import json

# 加载 JSON 文件
with open(pts_file, 'r') as f:
    pred_data = json.load(f)

# 转换为 numpy 数组
pts_semantic_mask = np.array(pred_data['pts_semantic_mask'])
pred_scores = np.array(pred_data['box_type_3d'])

print(pts_semantic_mask.shape)
print(pts_semantic_mask[0: 10])

# 读取 LAS 文件
las_file = "C:/Users/zhong/git/mmdetection3d/data/cicc/lidar_scanning_20240808.las" # 替换为你的 .las 文件路径
las = laspy.read(las_file)

print('加载 las 成功')
# 获取点云坐标 + sementic 后的值
points = np.vstack((las.x, las.y, las.z)).transpose()

points_vector3d = o3d.utility.Vector3dVector(points)


# 创建 Open3D 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = points_vector3d


# 计算 分类识别后的 color
# 定义映射规则
color_map = {
    1: [255, 255, 255],  # 值 1 
    2: [252, 252, 252],  # 值 2 映射到浅灰色
    3: [253, 253, 253]   # 值 3 映射到另一种灰色
}

class_names = (
'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
'bathtub', 'otherfurniture'
)

class_names_cn = (
' 墙 ', ' 地板 ', ' 橱柜 ', ' 床 ', ' 椅子 ', ' 沙发 ', ' 桌子 ',
' 门 ', ' 窗户 ', ' 书架 ', ' 图画 ', ' 柜台 ', ' 书桌 ',
' 窗帘 ', ' 冰箱 ', ' 淋浴帘 ', ' 马桶 ', ' 水槽 ',
' 浴缸 ', ' 其他家具 '
)

palette = [
    [255, 255, 0], [255, 0, 0], [31, 119, 180], [255, 187, 120], [188, 189, 34], [140, 86, 75], [255, 152, 150],
    [214, 39, 40], [197, 176, 213], [148, 103, 189], [196, 156, 148], [23, 190, 207], [247, 182, 210],
    [219, 219, 141], [255, 127, 14], [158, 218, 229], [44, 160, 44], [112, 128, 144],
    [227, 119, 194], [82, 84, 163],
]

palette_default = [200, 200, 200]
palette_mark__1 = [255, 0, 0]
palette_mark__2 = [0, 255, 0]
palette_mark__3 = [0, 0, 255]
palette_mark__4 = [255, 255, 0]
palette_mark__5 = [255, 0, 255]
palette_mark__6 = [0, 255, 255]
palette = [
    palette_mark__1, palette_mark__2, palette_default, palette_default, palette_default, palette_default, palette_default, 
    palette_mark__3, palette_mark__4, palette_default, palette_default, palette_default, palette_default, 
    palette_default, palette_default, palette_default, palette_default, palette_default, 
    palette_default, palette_default, 
]


mapped_colors = np.array([palette[value] for value in pts_semantic_mask])
mapped_colors = mapped_colors / 255
print(mapped_colors.shape)
print(mapped_colors[0:10])
colors = o3d.utility.Vector3dVector(mapped_colors)
colors
print(colors[0: 10])

# 用分类后的颜色
pcd.colors = colors

# 可选：如果 LAS 文件包含颜色信息，也可以设置颜色
if hasattr(las, 'red'):
    # colors = np.vstack((las.red, las.green, las.blue)).transpose() / 65535  # 通常 LAS 的颜色值范围是 0-65535
    # print(colors[0])
    pcd.colors = o3d.utility.Vector3dVector(colors)




# # 可视化点云
# o3d.visualization.draw_geometries([pcd])

# 创建可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window()

# 添加点云
vis.add_geometry(pcd)

# 获取渲染选项并设置
opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])  # 设置黑色背景
opt.point_size = 1  # 设置点的大小
opt.show_coordinate_frame = True  # 显示坐标轴

# 运行可视化
vis.run()

# 清理
vis.destroy_window()


