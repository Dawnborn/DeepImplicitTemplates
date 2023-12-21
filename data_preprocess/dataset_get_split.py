import os
import json
import random

# 设置随机种子，确保每次运行得到相同的随机结果
random_seed = 42
random.seed(random_seed)

# 假设你有一个文件列表
# file_list = os.listdir("/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/data/canonical_manifoldplus/02808440")
file_list = os.listdir("/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/data/ScanARCW/sdf_samples_canonical_manifoldplus/04256520")

# 随机打乱文件列表
random.shuffle(file_list)

# 计算80%的索引位置
split_index = int(0.8 * len(file_list))

# 分割成80%和20%两份
train_files = file_list[:split_index]
test_files = file_list[split_index:]

test_json_file_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/examples/splits/sv2_sofas_test_manifoldplus_scanarcw.json"
train_json_file_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/examples/splits/sv2_sofas_train_manifoldplus_scanarcw.json"
all_json_file_path = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/examples/splits/sv2_sofas_all_manifoldplus_scanarcw.json"

with open(train_json_file_path, "w") as json_file:
    data = dict()
    idata = dict()
    idata["04256520"] = train_files
    data["sdf_samples_canonical_manifoldplus"] = idata
    json.dump(data, json_file, indent=4)

with open(test_json_file_path, "w") as json_file:
    data = dict()
    idata = dict()
    idata["04256520"] = test_files
    data["sdf_samples_canonical_manifoldplus"] = idata
    json.dump(data, json_file, indent=4)