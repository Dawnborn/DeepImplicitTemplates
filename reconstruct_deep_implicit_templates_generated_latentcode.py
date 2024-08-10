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
        # default="examples/sofas_dit_manifoldplus_scanarcw_origpreprocess",
        # default="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/examples/sofas_dit_manifoldplus_scanarcw_origprep_all_pretrained",
        # default="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/examples/sofas_dit_manifoldplus_scanarcw_origprep_all_mypretrainedb24_b24",
        # default="/storage/user/huju/transferred/ws_dditnach/DeepImplicitTemplates/examples/chairs_dit_manifoldplus_scanarcw_origprep_all_mypretrained_b24",
        # default="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/examples_NewPretrainedDITonHJPDataOrig/sofas_dit",
        # default="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/examples/sofas_dit_manifoldplus_scanarcw_origprep_all_mypretrainedb24_b24",
        default="/home/wiss/lhao/storage/user/hjp/ws_dditnach/DeepImplicitTemplates/examples/cabinets_dit_manifoldplus_scanarcw_origprep_all_large_pretrainedsofas",
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        # default="1000",
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
        default="examples/splits/sv2_sofas_train_manifoldplus_scanarcw_origpreprocess.json",
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

    # 'examples/sofas_dit_manifoldplus_scanarcw/specs.json'
    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))
    latent_size = specs["CodeLength"] # 256
    
    # 定义模型：从specs.json文件里定义模型: from networks.deep_implicit_template_decoder import Decoder
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
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

    # pdb.set_trace()

    # lat_root = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/lat_test"
    # lat_root = "/home/wiss/lhao/binghui_DONTDELETE_ME/DDIT/DDIT_models_hjp/afterfix_exp_1cl_standard_lr_scheduler_newpretraineddithjpdataorig_diff/test/latest/pred_latent"
    # lat_root = "/home/wiss/lhao/binghui_DONTDELETE_ME/DDIT/DDIT_models_hjp/afterfix_exp_1cl_standard_lr_scheduler_newpretraineddithjpdataorig_diff/test/21/pred_latent"

    # lat_root = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_uncond2_l1/output/epoch=43999"

    # lat_root = "/home/wiss/lhao/binghui_DONTDELETE_ME/DDIT/DDIT_models_hjp/afterfix_exp_1cl_standard_lr_scheduler_newpretraineddithjpdataorig_diff_l1/test/latest/pred_latent"

    # lat_root = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw/output"

    # lat_root = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw7/output/499"

    lat_root = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw_sinl1_pc1024_10times42/output/last"
    lat_root = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw/output/49999"
    lat_root = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw/output/69999"
    lat_root = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond/output/23999"
    lat_root = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw/output/69999"

    lat_root = "/storage/user/huju/transferred/ws_dditnach/DeepImplicitTemplates/examples/chairs_dit_manifoldplus_scanarcw_origprep_all_mypretrained_b24/LatentCodes/train/2000/canonical_mesh_manifoldplus/03001627"
    lat_root = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_sofa_train_neighbor/output/49999/test/lat"
    lat_root = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_sofa_train_noneighbor/output/69999/test/lat"
    lat_root = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_sofa_train_neighbor/output/69999/test/lat"
    lat_root = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_chair_train_neighbor/output/18999/test/lat"
    lat_root = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_chair_train_noneighbor/output/18999/test/lat"

    lat_root = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_sofa_train_noneighbor_pcd128test/output/69999/test/lat"

    # reconstruction_dir = "./test_output"
    # reconstruction_dir = "/home/wiss/lhao/binghui_DONTDELETE_ME/DDIT/DDIT_models_hjp/afterfix_exp_1cl_standard_lr_scheduler_newpretraineddithjpdataorig_diff/test/latest/reconstruction"
    # reconstruction_dir = "/home/wiss/lhao/binghui_DONTDELETE_ME/DDIT/DDIT_models_hjp/afterfix_exp_1cl_standard_lr_scheduler_newpretraineddithjpdataorig_diff/test/21/reconstruction"
    
    # # reconstruction_dir = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_uncond2_l1/output/epoch=43999/reconstruction"
    # reconstruction_dir = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_uncond2_l1/reconstruction/epoch=43999"

    # reconstruction_dir = "/home/wiss/lhao/binghui_DONTDELETE_ME/DDIT/DDIT_models_hjp/afterfix_exp_1cl_standard_lr_scheduler_newpretraineddithjpdataorig_diff_l1/test/latest/recon"

    # reconstruction_dir = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw/recon"

    # reconstruction_dir = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw7/output/499/recon"

    reconstruction_dir = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw_sinl1_pc1024_10times42/recon/last"
    reconstruction_dir = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw/recon/49999"
    reconstruction_dir = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw/recon/69999"
    reconstruction_dir = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond/recon/23999"
    reconstruction_dir = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/stage2_diff_cond_scanarcw/recon/69999"
    reconstruction_dir = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_sofa_train_neighbor/output/49999/test/mesh"
    reconstruction_dir = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_sofa_train_noneighbor/output/69999/test/mesh"
    reconstruction_dir = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_sofa_train_neighbor/output/69999/test/mesh"
    reconstruction_dir = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_chair_train_neighbor/output/18999/test/mesh"
    reconstruction_dir = "/home/wiss/lhao/storage/user/hjp/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_chair_train_noneighbor/output/18999/test/mesh"

    reconstruction_dir = "/storage/user/huju/transferred/ws_dditnach/Diffusion-SDF/config/ddit_stage2_diff_cond_sofa_train_noneighbor_pcd128test/output/69999/test/mesh"

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)
    # 'examples/sofas_dit_manifoldplus_scanarcw/Reconstructions/2000/Meshes'
    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    clamping_function = None
    
    # specs["NetworkArch"]="deep_implicit_template_decoder"
    if specs["NetworkArch"] == "deep_sdf_decoder":
        clamping_function = lambda x : torch.clamp(x, -specs["ClampingDistance"], specs["ClampingDistance"])
    elif specs["NetworkArch"] == "deep_implicit_template_decoder":
        # clamping_function = lambda x: x * specs["ClampingDistance"]
        # 该函数将输入裁剪到给定范围
        clamping_function = lambda x : torch.clamp(x, -specs["ClampingDistance"], specs["ClampingDistance"])

    pth_filenames = os.listdir(lat_root)
    for ii, pth in tqdm(enumerate(pth_filenames)):

        latent_filename = os.path.join(lat_root,pth)

        mesh_filename = os.path.join(reconstruction_meshes_dir, pth.split(".")[0]+".ply")
        # latent_filename = os.path.join(
        #     reconstruction_codes_dir, npz[:-4] + ".pth"
        # )

        logging.info("reconstructing {}".format(pth))

        # 如果该data_sdf已经有latent_code，则直接加载，如果没有
        start = time.time()
        # pdb.set_trace()
        if not os.path.exists(latent_filename): # 'examples/sofas_dit_manifoldplus_scanarcw/Reconstructions/2000/Codes/ScanARCW/04256520/11d5e99e8faa10ff3564590844406360_scene0604_00_ins_5.pth
            print("Pretrained latent code not found!")
            import pdb
            pdb.set_trace()
        else:
            print("========= loading from " + latent_filename)
            logging.info("========= loading from " + latent_filename)
            latent = torch.load(latent_filename).squeeze(0).cuda()

        # decoder不更新权重，decoder纯推理模式
        decoder.eval()

        if os.path.isfile(mesh_filename) and args.skip:
            print("{} already exists and will be skipped!!!!!!")
            continue

        if not os.path.exists(os.path.dirname(mesh_filename)):
            os.makedirs(os.path.dirname(mesh_filename))

        # pdb.set_trace()
        start = time.time()
        max_batch = int(2 ** 18) 
        # max_batch = int(2 ** 9)
        with torch.no_grad():
            if args.use_octree:
                deep_sdf.mesh.create_mesh_octree(
                    decoder, latent, mesh_filename, N=args.resolution, max_batch=max_batch,
                    clamp_func=clamping_function
                )
            else:
                deep_sdf.mesh.create_mesh(
                    decoder, latent, mesh_filename, N=args.resolution, max_batch=max_batch,
                )
        logging.debug("total time: {}".format(time.time() - start))
