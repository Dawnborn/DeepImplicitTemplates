import subprocess
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

manifold_plus_bin = "../ManifoldPlus/build/manifold"
# shapenet_root="../DATA/ShapeNetCore.v2"
shapenet_root = "../DATA/ScanARCW/canonical_mesh"
# output_root = "../DATA/ShapeNet_processed"
output_root = "../DATA/ScanARCW/canonical_mesh_manifoldplus"

label2id = {
        "sofa": "04256520",
        "table": "04379243",
        "bed": "02818832",
        "bathtub": "02808440",
        "chair": "03001627",
        "cabinet": "02871439",
        # 'plane':'02691156'
    }

id2label = {
        "04256520": "sofa",
        "04379243": "table",
        "02818832": "bed",
        "02808440": "bathtub",
        "03001627": "chair",
        "02871439": "cabinet"
    }

if __name__=="__main__":
    print("当前工作目录：", os.getcwd())

    # 设定新的路径
    # new_path = '/path/to/your/directory'

    # 改变当前工作目录
    # os.chdir(new_path)

    # 打印当前工作目录来确认改变
    # print("当前工作目录：", os.getcwd())

    category_list = label2id.keys()

    for key in category_list:

        category_path = os.path.join(shapenet_root, label2id[key])
        category_output_dir = os.path.join(output_root, label2id[key])

        if not os.path.exists(category_output_dir):
            os.makedirs(category_output_dir)

        def process_file(file_id):
            obj_path = os.path.join(category_path, file_id, "model_canonical.obj")
            
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
