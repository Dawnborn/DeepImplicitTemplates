#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

# 用于将存在一起的embeddings拆分然后重建

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
        # default="examples/sofas_dit_manifoldplus_scanarcw",
        default="examples_mingao/sofas_dit",
        # default="examples/sofas_dit_minggaodata_all",
        # default="examples/sofas_dit_manifoldplus_scanarcw_origprep_all",
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
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
        default="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/examples_mingao/splits/watertight_scale/sv2_sofas_all_big.json",
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

    sdf_decoder = decoder.sdf_decoder

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


    for k in range(repeat):

        # 将sdf点云pose和neg部分随机打乱
        # data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
        # data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

        # decoder不更新权重，decoder纯推理模式
        sdf_decoder.eval()
        mesh_filename = os.path.join(args.experiment_directory, "Q_{}.ply".format(args.checkpoint))

        if not os.path.exists(os.path.dirname(mesh_filename)):
            os.makedirs(os.path.dirname(mesh_filename))

        latent=None

        args.use_octree = True
        if not save_latvec_only:
            max_batch=int(2**19) # 48G
            max_batch=int(2**17)+int(2**16) # 16G
            start = time.time()
            with torch.no_grad():
                if args.use_octree:
                    mesh_filename = os.path.join(args.experiment_directory, "Q_octree_{}.ply".format(args.checkpoint))
                    deep_sdf.mesh.create_mesh_octree(
                        sdf_decoder, latent, mesh_filename, N=args.resolution, max_batch=max_batch,
                        clamp_func=clamping_function
                    )
                else:
                    deep_sdf.mesh.create_mesh(
                        sdf_decoder, latent, mesh_filename, N=args.resolution, max_batch=max_batch,
                    )
            logging.debug("total time: {}".format(time.time() - start))
            print(mesh_filename)
        # if not os.path.exists(os.path.dirname(latent_filename)):
        #     os.makedirs(os.path.dirname(latent_filename))

        # # torch.save(latent.unsqueeze(0), latent_filename)
