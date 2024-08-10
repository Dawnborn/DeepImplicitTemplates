#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

# 用于将存在一起的embeddings拆分成独立的latent vectors然后重建

import argparse
import json
import logging
import os
import random
import time
import torch
import numpy as np

import deep_sdf
import deep_sdf.workspace as ws

from tqdm import tqdm

import pdb


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        # default="examples/sofas_dit_manifoldplus_scanarcw_origpreprocess",
        # default="examples/sofas_dit_manifoldplus_scanarcw_origprep_all",
        # default = "examples/sofas_dit_minggaodata_all",
        # default = "pretrained/sofas_dit",
        # default = "examples/sofas_dit_minggaodata_all_pretrained",
        # default = "examples/sofas_dit_manifoldplus_scanarcw_origprep_all_pretrained",
        # default = "examples/sofas_dit_minggaodata_single",
        # default = "examples/sofas_dit_manifoldplus_scanarcw_origprep_big",
        # default = "examples/sofas_dit_manifoldplus_scanarcw_origprep_single",
        # default = "examples_old/sofas_dit",
        # default = "examples/sofas_dit_manifoldplus_scanarcw_hjppython",
        # default = "examples/sofas_dit_manifoldplus_shapenet",
        # default = "examples/chairs_dit_manifoldplus_scanarcw_origprep_all_mypretrained_b24",
        default = "examples/bathtubs_dit_manifoldplus_scanarcw_origprep_all_b24",
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="2000",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        default="./data",
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        # default="examples/splits/sv2_sofas_all_manifoldplus_scanarcw_origpreprocess.json",
        # default = "examples/splits/sv2_sofas_all_minggaodata.json",
        # default = "examples/splits/sv2_sofas_train.json",
        # default="examples/splits/sv2_sofas_all_minggaodata.json",
        # default="examples/splits/sv2_sofas_all_manifoldplus_scanarcw_origpreprocess.json",
        # default = "examples/splits/sv2_sofas_single_minggaodata.json",
        # default = "examples/splits/sv2_sofas_single_manifoldplus_scanarcw_origpreprocess.json",
        default=None,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=1000,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        default="--skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    arg_parser.add_argument(
        "--seed",
        dest="seed",
        default=10,
        help="random seed",
    )
    arg_parser.add_argument(
        "--resolution",
        dest="resolution",
        type=int,
        default=256,
        help="Marching cube resolution.",
    )
    arg_parser.add_argument(
        "--max_batch",
        dest="max_batch",
        type=int,
        default=2**19,
        help="batch_size for reconstruction.",
    )


    use_octree_group = arg_parser.add_mutually_exclusive_group()
    use_octree_group.add_argument(
        '--octree',
        dest='use_octree',
        action='store_true',
        help='Use octree to accelerate inference. Octree is recommend for most object categories '
             'except those with thin structures like planes'
    )
    use_octree_group.add_argument(
        '--no_octree',
        dest='use_octree',
        action='store_false',
        help='Don\'t use octree to accelerate inference. Octree is recommend for most object categories '
             'except those with thin structures like planes'
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    random.seed(31359)
    torch.random.manual_seed(31359)
    np.random.seed(31359)

    deep_sdf.configure_logging(args)

    def empirical_stat(latent_vecs, indices):
        # 统计latent vector均值和方差，未用到
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    # 'examples/sofas_dit_manifoldplus_scanarcw/specs.json'
    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    # 定义模型：从specs.json文件里定义模型: from networks.deep_implicit_template_decoder import Decoder
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"] # 256

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    # 加载模型: 'examples/sofas_dit_manifoldplus_scanarcw/ModelParameters/2000.pth'
    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"]) # 

    decoder = decoder.module.cuda()

    # 读取数据split：'examples/splits/sv2_sofas_test_manifoldplus_scanarcw.json'
    if args.split_filename is None:
        args.split_filename = specs["TrainSplit"]
        print(args.split_filename)
    with open(args.split_filename, "r") as f:
        split = json.load(f)

    # 读取 data/SdfSamples/ScanARCW下数据， ScanARCW为split中数据集名称
    npz_filenames = deep_sdf.data.get_instance_filenames(args.data_source, split)
    # npz_filenames = split["ShapeNetV2"]["04256520"]

    # 读取整体的latent code embeddings并独立保存
    lat_vecs_file_path = os.path.join(args.experiment_directory,ws.latent_codes_subdir,args.checkpoint+".pth")
    print("latent_vecs_file is stored at {}".format(lat_vecs_file_path))
    if os.path.isfile(lat_vecs_file_path):
        print("=============geting individuell latent vectors")
        lat_vecs = torch.load(lat_vecs_file_path)['latent_codes']['weight']
        assert(lat_vecs.shape[0]==len(npz_filenames))
        for lat_vec, npz_filename in zip(lat_vecs,npz_filenames):
            latent_filename = npz_filename.split(".")[0].split("/")[-1]+".pth"
            latent_dir = os.path.join(args.experiment_directory,ws.latent_codes_subdir,"train",str(args.checkpoint),os.path.dirname(npz_filename))
            if not os.path.exists(latent_dir):
                os.makedirs(latent_dir)
            latent_file_path = os.path.join(latent_dir,latent_filename)
            if os.path.isfile(latent_file_path) and args.skip:
                print("================latent {} already exists and will be skipped!".format(latent_file_path))
                continue
            # pdb.set_trace()
            torch.save(lat_vec,os.path.join(latent_dir,latent_filename))
            

    # random.shuffle(npz_filenames)
    # npz_filenames = sorted(npz_filenames)

    logging.debug(decoder)

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    # 'examples/sofas_dit_manifoldplus_scanarcw/Reconstructions/2000'
    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    # 'examples/sofas_dit_manifoldplus_scanarcw/Reconstructions/2000/Meshes'
    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)
    # reconstruction_codes_dir = latent_dir

    clamping_function = None
    
    # specs["NetworkArch"]="deep_implicit_template_decoder"
    if specs["NetworkArch"] == "deep_sdf_decoder":
        clamping_function = lambda x : torch.clamp(x, -specs["ClampingDistance"], specs["ClampingDistance"])
    elif specs["NetworkArch"] == "deep_implicit_template_decoder":
        # clamping_function = lambda x: x * specs["ClampingDistance"]
        # 该函数将输入裁剪到给定范围
        clamping_function = lambda x : torch.clamp(x, -specs["ClampingDistance"], specs["ClampingDistance"])

    checklist = ["2ee577ae974272dc9faa444e7f5fd692_scene0244_01_ins_3",
                 "3af6ec685a6aa14e2f99eeb199d54554_scene0299_00_ins_4",
                 "6ebad671c5ccfee94c02d21d2fa4d397_scene0601_00_ins_7",
                 "7ab86358957e386d76de5cade2fd5247_scene0030_00_ins_5",
                 "27c82d27a4d38e6f41b160b9adc112fd_scene0192_01_ins_2",
                 "3415f252bd71495649920492438878e5_scene0480_00_ins_5",
                 "8458d6939967ac1bbc7a6acbd8f058b_scene0050_02_ins_4",
                 "8659f0f422096e3d26f6c8b5b75f0ee9_scene0050_01_ins_4",
                 "735578d3cfa39279d098b24937823a09_scene0128_00_ins_8"]
    
    checklist = None

    # filename = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/data_preprocess/small_list.txt"
    # # Open the file for reading ('r' mode)
    # with open(filename, 'r') as file:
    #     # Read lines into a list
    #     lines = file.readlines()
    # # You may want to strip newline characters from each line
    # checklist = [line.rstrip('\n') for line in lines]

    for ii, npz in tqdm(enumerate(npz_filenames)):

        # if "npz" not in npz:
        #     continue
        if len(npz.split("."))==1:
            npz = npz+".npz"
        
        if checklist is not None:
            npz_basename = os.path.basename(npz)
            if npz_basename.split(".")[0] not in checklist:
                continue


        full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)

        logging.debug("loading {}".format(npz))

        # data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename)
        # data_sdf[0] N_neg,4, data_sdf[1] N_pos,4 每个采样不一样

        for k in range(repeat):

            if rerun > 1:
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, npz[:-4] + "-" + str(k + rerun)
                )
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4] + "-" + str(k + rerun) + ".pth"
                )
                
            else:
                mesh_filename = os.path.join(reconstruction_meshes_dir+"_train", npz[:-4])
                # latent_filename = os.path.join(
                #     reconstruction_codes_dir, npz[:-4] + ".pth"
                # )
                latent_filename = os.path.join(latent_dir,os.path.basename(npz)[:-4]+".pth")

            if (
                args.skip
                and os.path.isfile(mesh_filename + ".ply")
                # and os.path.isfile(latent_filename)
            ):
                continue

            logging.info("reconstructing {}".format(npz))

            # 将sdf点云pose和neg部分随机打乱
            # data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
            # data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

            # 如果该data_sdf已经有latent_code，则直接加载，如果没有
            start = time.time()
            # pdb.set_trace()
            if not os.path.isfile(latent_filename): # 'examples/sofas_dit_manifoldplus_scanarcw/Reconstructions/2000/Codes/ScanARCW/04256520/11d5e99e8faa10ff3564590844406360_scene0604_00_ins_5.pth
                print("Pretrained latent code not found!")
                pdb.set_trace()
            else:
                print("========= loading from " + latent_filename)
                logging.info("========= loading from " + latent_filename)
                latent = torch.load(latent_filename).squeeze(0).cuda()

            # decoder不更新权重，decoder纯推理模式
            decoder.eval()

            if not os.path.exists(os.path.dirname(mesh_filename)):
                os.makedirs(os.path.dirname(mesh_filename))

            if not save_latvec_only:
                start = time.time()
                # max_batch = int(2 ** 19) + int(2**17) # 48G
                # max_batch = int(2 ** 19) # 48G
                max_batch = args.max_batch

                # max_batch = int(2**17) # 16G
                # max_batch = int(2**16) # 8G
                with torch.no_grad():
                    if args.use_octree: # False
                        deep_sdf.mesh.create_mesh_octree(
                            decoder, latent, mesh_filename, N=args.resolution, max_batch=max_batch,
                            clamp_func=clamping_function
                        )
                    else:
                        deep_sdf.mesh.create_mesh(
                            decoder, latent, mesh_filename, N=args.resolution, max_batch=max_batch,
                        )
                logging.debug("total time: {}".format(time.time() - start))

            # if not os.path.exists(os.path.dirname(latent_filename)):
            #     os.makedirs(os.path.dirname(latent_filename))
            # # torch.save(latent.unsqueeze(0), latent_filename)