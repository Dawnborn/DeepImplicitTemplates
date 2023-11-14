import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from mesh_to_sdf import sample_sdf_near_surface
import trimesh
# import pyrender
import numpy as np
import glob
import json
import time
import pdb
from tqdm import tqdm

from multiprocessing import Pool, cpu_count

def write_json(filename, data):
    check_dirname(filename)
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

def read_json(filename):
	with open(filename, 'r') as infile:
		return json.load(infile)

# def visualize(points, sdf):
#     colors = np.zeros(points.shape)
#     colors[sdf < 0, 2] = 1
#     colors[sdf > 0, 0] = 1
#     cloud = pyrender.Mesh.from_points(points, colors=colors)
#     scene = pyrender.Scene()
#     scene.add(cloud)
#     viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

def check_dirname(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

def compute_scale_to_unit_sphere(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    another_trans = mesh.bounding_box.centroid
    vertices = mesh.vertices - another_trans
    distances = np.linalg.norm(vertices, axis=1)
    another_scale = np.max(distances)
    # vertices /= another_scale

    return another_trans, another_scale

def main(mesh_name, format="ScanARCW", json_input_root="DATA/ScanARCW/hjp/json_files_v2/", json_output_root="DATA/ScanARCW/hjp/json_files_v3/", sdf_sample_output_root="DATA/ScanARCW/sdf_samples_canonical_manifoldplus", force_output=False):
    write_json = None
    if format=="ScanARCW":
        category_id, ins_name = mesh_name.split('/')[-3:-1]
        write_json = True
    elif format=="ShapeNet_manifoldplus":
        category_id, ins_name = mesh_name.split('/')[-3:-1]
        write_json = False
    else:
        raise NotImplementedError
    
    # pdb.set_trace()
    if len(category_id) != 8:
        print("wrong category_id:{}".format(category_id))
        raise NotImplementedError
    
    start = time.time()

    if write_json:
        cad_id, scene_name, scene_cnt, _, ins_id = ins_name.split('_')
        scene_id = scene_name + '_' + scene_cnt
        
        json_file = json_input_root + scene_id + '.json'
        if not os.path.isfile(json_file):
            print("json file {} not exists!".format(json_file))
            raise
        json_outname = json_output_root + scene_id + '.json'
        if os.path.isfile(json_outname):
            json_file = json_outname
        scene_data = read_json(json_file)
    
    sdf_outname = os.path.join(sdf_sample_output_root, category_id, ins_name + ".npz")
    if os.path.exists(sdf_outname):
        if not force_output:
            print("sdf {} already exists! Skip".format(sdf_outname))
            return
        else:
            print("sdf {} will be overwritten!".format(sdf_outname))

    mesh = trimesh.load(mesh_name)
    
    another_trans, another_scale = compute_scale_to_unit_sphere(mesh)

    if write_json:
        scene_data[scene_id]["instances"][ins_id]["scale_sdf2mesh"] = [another_scale]
        scene_data[scene_id]["instances"][ins_id]["translation_sdf2mesh"] = [another_trans[0],
                                                                            another_trans[1],
                                                                            another_trans[2]]

    points, sdf = sample_sdf_near_surface(mesh, number_of_points = 500000, 
                                        surface_point_method='scan', sign_method='normal', 
                                        scan_count=100, scan_resolution=400, 
                                        sample_point_count=10000000, normal_sample_count=11, 
                                        min_size=0, return_gradients=False, near_scale=0.02, far_scale=0.05)
    pos_mask = sdf > 0
    pos = np.hstack((points[pos_mask], sdf[pos_mask][:, None]))
    neg_mask = sdf <= 0
    neg = np.hstack((points[neg_mask], sdf[neg_mask][:, None]))
    
    check_dirname(sdf_outname)
    np.savez(sdf_outname, pos = pos, neg = neg, scale_sdf2mesh=another_scale, translation_sdf2mesh=another_trans, mesh_path=mesh_name)

    if write_json:
        scene_data[scene_id]["instances"][ins_id]["gt_sdf"] = sdf_outname


        check_dirname(json_outname)
        write_json(json_outname, scene_data)
    
    end = time.time()
    print(f'Running time for {ins_name}: {end-start} second') 
    print(f'Running time for {ins_name}: {end-start} second', file=log_file) 
    
if __name__ == "__main__":
    # mesh_names = sorted(glob.glob("DATA/ScanARCW/canonical_mesh_manifoldplus/*/*/model_canonical_manifoldplus.obj"))
    # mesh_names = ["DATA/ScanARCW/canonical_mesh_manifoldplus/02818832/1f11b3d9953fabcf8b4396b18c85cf0f_scene0078_01_ins_2/model_canonical_manifoldplus.obj"]
    # mesh_names = ["DATA/ScanARCW/canonical_mesh_manifoldplus/04256520/1a4a8592046253ab5ff61a3a2a0e2484_scene0484_00_ins_1/model_canonical_manifoldplus.obj"]
    mesh_names = ["/data/hdd1/storage/junpeng/ws_dditplus/DeepImplicitTemplates/data/ShapeNet_manifoldplus/02808440/1a74ca07a11f507554d7082b34825ef0/model_normalized_manifoldplus.obj"]
    label2id = {
        "sofa": "04256520",
        # "table": "04379243",
        # "bed": "02818832",
        # "bathtub": "02808440",
        # "chair": "03001627",
        # "cabinet": "02871439",
        # 'plane':'02691156',
        'bottle':'02876657',
    }

    log_file = open(f'data_preprocess/log/generate_sdf.txt', 'a')

    for id in label2id.values():
        mesh_names = sorted(glob.glob("data/ShapeNet_manifoldplus/{}/*/model_normalized_manifoldplus.obj".format(id)))
        # mesh_names = ["data/ShapeNetCore.v2/02691156/1a04e3eab45ca15dd86060f189eb133/models/model_normalized.obj"]
        
        # pdb.set_trace()
        
        for mesh_name in tqdm(mesh_names):
            try:
                main(mesh_name=mesh_name,format="ShapeNet_manifoldplus", sdf_sample_output_root="data/SdfSamples/ShapeNet_manifoldplus")
            except Exception as a:
                print(f"============={mesh_name} encounter error: {a}, continue==============")
                print(f"============={mesh_name} encounter error: {a}, continue==============", file=log_file)
            continue
    
    log_file.close()
        
        
        
        

