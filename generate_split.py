import os
import json

dict_all = {
    "02808440": "bathtub",
    "02818832": "bed",
    "02871439": "bookshelf",
    "02933112": "cabinet",
    "03001627": "chair",
    "04256520": "sofa",
    "04379243": "table"
}

root_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/DATA/ScanARCW/canonical_mesh_manifoldplus"
root_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/DATA/ScanARCW/sdf_samples"
output_root = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/examples/splits"
output_root = "/storage/user/huju/transferred/ws_dditnach/DeepImplicitTemplates/examples/splits"

id = "03001627"
id = "04379243"
id = "02808440"
id = "02818832"
id = "02933112"
id = "02871439"

dir_path = os.path.join(root_path,id)
dir_list = sorted(os.listdir(dir_path))
dir_list = [dir.split(".")[0] for dir in dir_list]

l = len(dir_list)

train_list = dir_list[:int(0.8*l)]
test_list = [dir for dir in dir_list if not dir in train_list]

train_list = dir_list

ans = dict()

ans[id] = dir_list
ans2 = dict()
ans2 = {"canonical_mesh_manifoldplus":ans}

output_file = "scanarcw_{}s_all_manifoldplus.json".format(dict_all[id])
output_path = os.path.join(output_root, output_file)
# 写入JSON文件，使用4个空格进行缩进
with open(output_path, 'w') as f:
    json.dump(ans2, f, indent=4)

output_file = "scanarcw_{}s_train_manifoldplus.json".format(dict_all[id])
output_path = os.path.join(output_root, output_file)
ans[id] = train_list
ans2 = dict()
ans2 = {"canonical_mesh_manifoldplus":ans}
# 写入JSON文件，使用4个空格进行缩进
with open(output_path, 'w') as f:
    json.dump(ans2, f, indent=4)

output_file = "scanarcw_{}s_test_manifoldplus.json".format(dict_all[id])
output_path = os.path.join(output_root, output_file)
ans[id] = test_list
ans2 = dict()
ans2 = {"canonical_mesh_manifoldplus":ans}
# 写入JSON文件，使用4个空格进行缩进
with open(output_path, 'w') as f:
    json.dump(ans2, f, indent=4)
