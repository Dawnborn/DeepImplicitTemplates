import numpy as np
from skimage.measure import marching_cubes
# from plyfile import PlyData, PlyElement
import os
# from scipy.interpolate import RegularGridInterpolator

import open3d as o3d

# import pymesh

def marching_cubes_to_ply(points, output_file):
    # 提取xyz坐标和SDF值
    xyz = points[:, :3]
    sdf_values = points[:, 3]

    # 构建Mesh对象
    mesh = pymesh.form_mesh(xyz, [], sdf_values)

    # 进行Marching Cubes重建
    mesh = pymesh.compute_outer_hull(mesh, 0.0)

    # 保存Mesh为PLY文件
    pymesh.save_mesh(output_file, mesh)
    return mesh

def create_mesh_from_point_cloud(points, sdf_threshold=0.01):
    """
    使用 Open3D 从点云创建网格。
    :param points: 输入点的 (N, 4) 矩阵，包含 x, y, z 坐标和 SDF 值。
    :param sdf_threshold: SDF 阈值，用于确定哪些点包含在表面上。
    :return: 创建的网格。
    """
    # 筛选出 SDF 值接近零的点
    surface_points = points[np.abs(points[:, 3]) < sdf_threshold][:, :3]

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface_points)

    # 使用球面半径估算计算点云的法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 使用泊松表面重建创建网格
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)[0]

    return mesh

# 运行主函数
if __name__ == "__main__":
    # 示例输入数据
    # 注意：此处需要替换为实际的点云数据
    points = np.array([
        # x, y, z, sdf
        [0, 0, 0, -1],
        [0, 0, 1, -0.5],
        # ... 其他点
    ])

    raw_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/data/SdfSamples/ScanARCW/04256520/1a4a8592046253ab5ff61a3a2a0e2484_scene0484_00_ins_1.npz"
    raw = np.load(raw_path)
    file_output = os.path.basename(raw_path)+".ply"
    pos = raw['pos']
    neg = raw['neg']

    all = np.vstack((pos,neg))

    
    
    # mesh = marching_cubes_to_ply(all,file_output)
    mesh = create_mesh_from_point_cloud(all,sdf_threshold=0.02)
    o3d.io.write_triangle_mesh("output4.ply", mesh)
