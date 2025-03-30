#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math
import imageio
from matplotlib import pyplot as plt
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import threading
import concurrent.futures
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from moviepy.editor import ImageSequenceClip
from sklearn.neighbors import NearestNeighbors

def render_edited(gaussians, viewpoint_camera, mask=None):
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0   

    raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background,
            scale_modifier=1.0,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=False,
        )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    ############################################################################################
    means3D = gaussians.get_xyz
    means2D = screenspace_points
    opacity = gaussians._opacity
    scales = gaussians._scaling
    rotations = gaussians._rotation  
    shs = gaussians.get_features      
    ############################################################################################
    #scene.getTrainCameras()[300].time
    time=torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    means3D_final, scales_final, rotations_final, opacity_final, shs_final = gaussians._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time)
        
    scales_final = gaussians.scaling_activation(scales_final)
    rotations_final = gaussians.rotation_activation(rotations_final)
    opacity_final = gaussians.opacity_activation(opacity_final)
    
    if (mask is None):
        rendered_image, radii, depth = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = None,
            opacities = opacity_final,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = None)
    else:
        #mask = ~mask
        rendered_image, radii, depth = rasterizer(
            means3D = means3D_final[mask],
            means2D = means2D,
            shs = shs_final[mask],
            colors_precomp = None,
            opacities = opacity_final[mask],
            scales = scales_final[mask],
            rotations = rotations_final[mask],
            cov3D_precomp = None)
    
    return rendered_image

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int) #
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--ply_path", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.ply_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    dataset= model.extract(args)
    iteration = args.iteration
    hyperparam = hyperparam.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, hyperparam)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    
    before_xyz = gaussians.get_xyz
    
    print("before edit: ", gaussians.get_xyz.shape)
    
    cam_type=scene.dataset_type
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    gaussians.load_ply(args.ply_path)
    gaussians.load_model(os.path.join(args.model_path, 'point_cloud', 'iteration_14000')) ##########
    
    after_xyz = gaussians.get_xyz
    print("after edit: ", gaussians.get_xyz.shape)

    ## TODO : render and make video for desired camera sequences
    cameras = scene.getTrainCameras()
    imgs = []
    to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
    for idx, viewpoint_camera in enumerate(tqdm(cameras, desc="Rendering progress")):
        if idx < 500 or idx > 549:
            continue
        rendered_img = render_edited(gaussians, viewpoint_camera)
        imgs.append(to8b(rendered_img.detach().cpu()).transpose(1,2,0))

    ## TODO: save_path
    imageio.mimwrite(os.path.join(args.model_path, f"{os.path.splitext(os.path.basename(args.ply_path))[0]}.mp4"), imgs, fps=30)
    print("Video Saved.")
        

    

    
