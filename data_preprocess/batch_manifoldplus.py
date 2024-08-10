import subprocess
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

manifold_plus_bin = "/storage/user/huju/transferred/ws_dditnach/ManifoldPlus/build/manifold"
# shapenet_root="data/ShapeNetCore.v2"
shapenet_root = "../DATA/ScanARCW/canonical_mesh"
shapenet_root = "/storage/user/huju/transferred/ws_dditnach/DATA/ScanARCW/canonical_mesh"

# output_root = "data/ShapeNet_manifoldplus"
output_root = "../DATA/ScanARCW/canonical_mesh_manifoldplus"
output_root = "/storage/user/huju/transferred/ws_dditnach/DATA/ScanARCW/canonical_mesh_manifoldplus"

label2id = {
        # "sofa": "04256520",
        # "table": "04379243",
        # "bed": "02818832",
        # "bathtub": "02808440",
        # "chair": "03001627",
        "cabinet": "02933112",
        # 'plane':'02691156',
        # 'bottle':'02876657',
    }

id2label = {
        "02691156": "airplane",
        "02747177": "trash bin",
        "02773838": "bag",
        "02801938": "basket",
        "02808440": "bathtub",
        "02818832": "bed",
        "02828884": "bench",
        "02843684": "birdhouse",
        "02871439": "bookshelf",
        "02876657": "bottle",
        "02880940": "bowl",
        "02924116": "bus",
        "02933112": "cabinet",
        "02942699": "camera",
        "02946921": "can",
        "02954340": "cap",
        "02958343": "car",
        "02992529": "cellphone",
        "03001627": "chair",
        "03046257": "clock",
        "03085013": "keyboard",
        "03207941": "dishwasher",
        "03211117": "display",
        "03261776": "earphone",
        "03325088": "faucet",
        "03337140": "file cabinet",
        "03467517": "guitar",
        "03513137": "helmet",
        "03593526": "jar",
        "03624134": "knife",
        "03636649": "lamp",
        "03642806": "laptop",
        "03691459": "loudspeaker",
        "03710193": "mailbox",
        "03759954": "microphone",
        "03761084": "microwaves",
        "03790512": "motorbike",
        "03797390": "mug",
        "03928116": "piano",
        "03938244": "pillow",
        "03948459": "pistol",
        "03991062": "flowerpot",
        "04004475": "printer",
        "04074963": "remote",
        "04090263": "rifle",
        "04099429": "rocket",
        "04225987": "skateboard",
        "04256520": "sofa",
        "04330267": "stove",
        "04379243": "table",
        "04401088": "telephone",
        "04460130": "tower",
        "04468005": "train",
        "04530566": "watercraft",
        "04554684": "washer"
    }

if __name__=="__main__":
    print("当前工作目录：", os.getcwd())

    # 设定新的路径
    # new_path = '/path/to/your/directory'

    # 改变当前工作目录
    # os.chdir(new_path)

    # 打印当前工作目录来确认改变
    # print("当前工作目录：", os.getcwd())

    # category_list = label2id.keys()
    key_list = os.listdir(shapenet_root)
    key_list = id2label.keys()
    key_list = ["02933112"]

    for key in key_list:

        category_path = os.path.join(shapenet_root, key)
        category_output_dir = os.path.join(output_root, key)

        if not os.path.exists(category_output_dir):
            os.makedirs(category_output_dir)

        def process_file(file_id):
            # obj_path = os.path.join(category_path, file_id, "models", "model_normalized.obj")
            obj_path = os.path.join(category_path, file_id, "model_canonical.obj")
            if not os.path.exists(obj_path):
                print("path: {} not exists!!! Check the directory structure!!!\n".format(obj_path))
                return
            
            obj_output_dir = os.path.join(category_output_dir,file_id)
            if not os.path.exists(obj_output_dir):
                os.makedirs(obj_output_dir)

            obj_output_path = os.path.join(obj_output_dir, "model_canonical_manifoldplus.obj")
            if not os.path.exists(obj_output_path):
                subprocess.run([manifold_plus_bin, "--input", obj_path, "--output", obj_output_path])
            else:
                print("{} existis and will be skipped!".format(obj_output_path))

        if __name__ == '__main__':
            file_ids = os.listdir(category_path)
            num_processes = os.cpu_count()-2

            # 使用 multiprocessing.Pool 来并行处理文件
            with Pool(processes=num_processes) as pool:
                list(tqdm(pool.imap(process_file, file_ids), total=len(file_ids)))
