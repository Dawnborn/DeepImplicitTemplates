# %%
import json
import trimesh
import numpy as np
import quaternion
import os

import plotly.graph_objs as go

# %%
def quaternion_list2rotmat(quant_list: list, format="xyzw"):
    assert len(quant_list) == 4, "Quaternion needs 4 elements"
    if format=="xyzw":
        q = np.quaternion(quant_list[0], quant_list[1], quant_list[2], quant_list[3])
    elif format=="wxyz":
        q = np.quaternion(quant_list[1], quant_list[2], quant_list[3], quant_list[0])
    R = quaternion.as_rotation_matrix(q)
    return R

def mesh_apply_rts(mesh, rotation_mat_c2w, translation_c2w, scale_c2w=None):
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    
    # Apply rotation
    # pcd_meshcoord = (np.linalg.inv(gt_rotation_mat_c2w) @ (pcd_world - gt_translation_c2w[np.newaxis, :]).T).T
    transformed_vertices = (rotation_mat_c2w @ vertices.T).T + translation_c2w[np.newaxis, :]

    x, y, z = transformed_vertices.T  # Transposed for easier unpacking
    i, j, k = faces.T  # Unpack faces

    mesh_transformed = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=0.5,
        color='rgba(244,22,100,0.5)'
    )
    return mesh_transformed

def mesh_load(mesh_path, scale_c2w=None, rotation_quat_wxyz=None, translation_c2w=None):
    # 从文件加载网格数据
    file_suffix = mesh_path.split(".")[-1]
    if file_suffix == "obj":
        with open(mesh_path, 'r') as file:
            lines = file.readlines()

        vertices = []
        faces = []

        for line in lines:
            if line.startswith('v '):
                vertex = line.split()[1:]
                vertices.append([float(vertex[0]), float(vertex[1]), float(vertex[2])])
            elif line.startswith('f '):
                face = line.split()[1:]
                face_indices = [int(idx.split('/')[0]) - 1 for idx in face]
                faces.append(face_indices)

        mesh = go.Mesh3d(x=[v[0] for v in vertices], y=[v[1] for v in vertices], z=[v[2] for v in vertices],
                        i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces], name="gt mesh")
        return mesh

    elif file_suffix == "ply":
        from plyfile import PlyData

        # 从PLY文件加载网格数据
        plydata = PlyData.read(mesh_path)

        # 提取顶点坐标
        vertices = np.array([list(vertex) for vertex in plydata['vertex'].data])

        # 提取面数据
        faces = np.array(plydata['face'].data['vertex_indices'])
        faces = np.array([list(row) for row in faces])

        # 创建网格图形对象
        mesh = go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], name="recon mesh")
    
    if scale_c2w:
        mesh = ap
    
        return mesh

# %%
scene_name = "scene0370_02" # 门和部分桌子有问题
scene_name = "scene0551_00"
data_root = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/DATA"
json_version = "json_files"

LAYOUT_VIS=True

# %%
vis_data = []

pcd_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/DATA/ScanARCW/complete_pcd/{}/{}.txt".format(scene_name,scene_name)
data = np.loadtxt(pcd_path)
N = data.shape[0]
M = 100000
data = data[np.random.choice(N, M, replace=False), :]
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
trace = go.Scatter3d(x=x, y=y, z=z, 
        mode='markers',     
        marker=dict(
            size=0.5,  # Adjust the size of the markers here
            color='rgba(35, 35, 250, 0.8)'  # Set the color you want (e.g., light blue)
        ),
        name="scene points")
vis_data.append(trace)

# json_file = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/DATA/ScanARCW/json_files_v2/{}.json".format(scene_name)
json_file = os.path.join(data_root,"ScanARCW",json_version,"{}.json".format(scene_name))

with open(json_file, 'r') as file:
    raw_json = json.load(file)
for scene_id in raw_json.keys():
    instances_data = raw_json[scene_id]["instances"]
    for instance_id in instances_data.keys():
        instance_data = instances_data[instance_id]
        if not instance_data:
            print("The dictionary is empty.")
            continue
        if instance_data['category_name'] == "layout" and (not LAYOUT_VIS):
            print("layout is skipped!")
            continue
        gt_translation_c2w = np.array(instance_data["gt_translation_c2w"]) # 3,
        mesh_path = os.path.join(data_root,instance_data["gt_scaled_canonical_mesh"])
        if instance_data.get("gt_rotation_quat_xyzw_c2w",False):
            print(mesh_path)
            gt_rotation_mat_c2w = quaternion_list2rotmat(instance_data["gt_rotation_quat_xyzw_c2w"],format="xyzw")
        elif instance_data.get("gt_rotation_quat_wxyz_c2w",False):
            print(mesh_path)
            gt_rotation_mat_c2w = quaternion_list2rotmat(instance_data["gt_rotation_quat_wxyz_c2w"])
        else:
            # import pdb
            # pdb.set_trace()
            print("skipped!")
            continue
        mesh_path = os.path.join(data_root,instance_data["gt_scaled_canonical_mesh"])
        # mesh = mesh_load(mesh_path)
        mesh = trimesh.load(mesh_path)
        mesh = mesh_apply_rts(mesh, gt_rotation_mat_c2w, gt_translation_c2w, scale_c2w=None)
        vis_data.append(mesh)
        # break
        # pcd_meshcoord = (np.linalg.inv(gt_rotation_mat_c2w) @ (pcd_world - gt_translation_c2w[np.newaxis, :]).T).T
        


# %%
layout = go.Layout(scene=dict(
        aspectmode='data',  # Set the aspect ratio to 'cube' for equal scales
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
    ),)

fig = go.Figure(data=vis_data, layout=layout)
fig.show()

# %%



