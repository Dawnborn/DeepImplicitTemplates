import os
import json

num2cat = {
    "02808440": "bathtub",
    "02818832": "bed",
    "02871439": "bookshelf",
    "02933112": "cabinet",
    "03001627": "chair",
    "04256520": "sofa",
    "04379243": "table"
}

cat2num = dict()
for key in num2cat.keys():
    cat2num[num2cat[key]] = key

cat = "bathtub"
cat = "bed"
cat = "cabinet"
cat = "bookshelf"
num = cat2num[cat]

size_limit_mb = 5
size_limit_bytes = size_limit_mb * 1024 * 1024  # 转换MB为字节

# large_files = find_large_files(directory_path, size_limit_bytes)
# print("Found large files:", large_files)

sdf_root = "/storage/user/huju/transferred/ws_dditnach/DeepSDF/data/SdfSamples"
sdf_dataset = "canonical_mesh_manifoldplus"

json_file_path = "/storage/user/huju/transferred/ws_dditnach/DeepImplicitTemplates/examples/splits/scanarcw_bathtubs_all_manifoldplus.json"
json_file_path = "/storage/user/huju/transferred/ws_dditnach/DeepImplicitTemplates/examples/splits/scanarcw_chairs_all_manifoldplus_origpreprocess.json"
json_file_path = "/storage/user/huju/transferred/ws_dditnach/DeepImplicitTemplates/examples/splits/scanarcw_{}s_all_manifoldplus.json".format(cat)

basename = os.path.basename(json_file_path)
new_basename = "scanarcw_{}s_all_large_manifoldplus.json".format(cat)
new_json_file_path = json_file_path.replace(basename,new_basename)
print(new_json_file_path)

large_files = []
with open(json_file_path, "r") as json_file:
    data = json.load(json_file)
    sdf_dataset = list(data.keys())[0]
    idata = data[sdf_dataset]
    num = list(idata.keys())[0]
    files = idata[num]
    print("original number: {}".format(len(files)))
    for file in files:
        if file.split(".")[-1] != "npz":
            file = file+".npz"
        file_path = os.path.join(sdf_root,sdf_dataset, num, file)
        if os.path.getsize(file_path) > size_limit_bytes:
            large_files.append(file)
    print("larger than {} Mb: {}".format(size_limit_mb,len(large_files)))

with open(new_json_file_path,"w") as json_file:
    data = dict()
    idata = dict()
    idata[num] = large_files
    data[sdf_dataset] = idata
    json.dump(data, json_file, indent=4)

print(new_json_file_path)

