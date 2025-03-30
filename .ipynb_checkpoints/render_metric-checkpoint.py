import math
import imageio
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
from torch.utils.data import DataLoader, Subset
import concurrent.futures
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from moviepy.editor import ImageSequenceClip
from sklearn.neighbors import NearestNeighbors

# ===== LPIPS, SSIM, PSNR 불러오기 =====
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr

# ===== (1) CLIP 관련 임포트 / 함수: 이미 로드했다고 가정해도 됨 =====
import clip
import torchvision.transforms as T
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
dict_coffee_martini = {
    0: 1,
    300: 2,
    600:4,
    900: 5,
    1200: 6,
    1500:7,
    1800:8,
    2100: 9,
    2400: 10,
    2700: 11,
    3000: 12,
    3300: 13,
    3600: 14,
    3900: 16,
    4200: 18,
    4500: 19,
    4800: 20,
    5100: None
}
def compute_clip_similarity(img_tensor: torch.Tensor, text_prompt: str) -> float:
    pil_image = T.ToPILImage()(img_tensor.detach().cpu())
    image_input = clip_preprocess(pil_image).unsqueeze(0).to(device)
    text_input = clip.tokenize([text_prompt]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (image_features * text_features).sum()
    return similarity.item()


def render_edited(gaussians, viewpoint_camera, mask=None):
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype,
                                          requires_grad=True, device="cuda") + 0

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
    
    means3D = gaussians.get_xyz
    means2D = screenspace_points
    opacity = gaussians._opacity
    scales = gaussians._scaling
    rotations = gaussians._rotation
    shs = gaussians.get_features

    time_t = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    means3D_final, scales_final, rotations_final, opacity_final, shs_final = gaussians._deformation(
        means3D, scales, rotations, opacity, shs, time_t
    )

    scales_final = gaussians.scaling_activation(scales_final)
    rotations_final = gaussians.rotation_activation(rotations_final)
    opacity_final = gaussians.opacity_activation(opacity_final)
    
    if mask is None:
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
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--ply_path", type=str)
    args = get_combined_args(parser)

    print("Rendering ", args.ply_path)

    # configs 로딩
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    # 데이터셋 / 모델 불러오기
    dataset = model.extract(args)
    iteration = args.iteration
    hyperparam = hyperparam.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, hyperparam)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    
    before_xyz = gaussians.get_xyz
    print("before edit: ", gaussians.get_xyz.shape)

    cam_type = scene.dataset_type
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Ply & Model Load
    gaussians.load_ply(args.ply_path)
    gaussians.load_model(os.path.join(args.model_path, 'point_cloud_3dedit', 'iteration_500'))

    after_xyz = gaussians.get_xyz
    print("after edit: ", gaussians.get_xyz.shape)
    
    # 카메라 Subset
    cameras = scene.getTrainCameras()
    t_idxs = list(range(0, len(cameras), scene.maxtime))
    cameras = Subset(cameras, t_idxs)

    # === Metric 리스트 초기화 ===
    ssims = []
    psnrs = []
    lpipss = []
    lpipsa = []
    clip_scores = []   # ★ CLIP 유사도 리스트 추가

    # 텍스트 프롬프트
    clip_prompt = "Make the person a wood sculpture"
    edited_images_path = "./data/dynerf/coffee_martini/wood_sculpture"

    to8b = lambda x: (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

    for idx, viewpoint_camera in enumerate(tqdm(cameras, desc="Rendering progress")):
        with torch.no_grad():
            # (1) 렌더링
            rendered_img = render_edited(gaussians, viewpoint_camera)
    
            # (2) Ground-truth (또는 original) 이미지 GPU로 이동
            #gt_img = viewpoint_camera.original_image.to(device)
            image = Image.open(os.path.join(edited_images_path, "edited_sculpture_render_time0_{:d}.png".format(dict_coffee_martini[int(viewpoint_camera.image_name)])))
            transform = transforms.ToTensor()
            gt_img = transform(image).cuda()
    
            if gt_img.shape != rendered_img.shape:
                # gt_img: [3, H1, W1]
                # rendered_img: [3, H2, W2]
                
                # interpolate는 (N,C,H,W) 4D 텐서 입력이므로 unsqueeze(0)로 배치 차원을 만든다
                gt_img_4d = gt_img.unsqueeze(0)  # [1,3,H1,W1]
                
                # rendered_img의 높이/너비를 얻어서 size=(H2, W2)로 설정
                new_size = rendered_img.shape[-2:]  # (H2, W2)
                
                # bilinear 보간으로 리사이즈
                gt_img_resized = F.interpolate(gt_img_4d, size=new_size, mode='bilinear', align_corners=False)
                
                # 다시 [3,H2,W2]로
                gt_img = gt_img_resized.squeeze(0)
            # (3) SSIM, PSNR, LPIPS (VGG / Alex)
            ssim_score = ssim(rendered_img, gt_img)
            psnr_score = psnr(rendered_img, gt_img)
            lpips_vgg = lpips(rendered_img, gt_img, net_type='vgg')
            lpips_alex = lpips(rendered_img, gt_img, net_type='alex')
    
            ssims.append(ssim_score)
            psnrs.append(psnr_score)
            lpipss.append(lpips_vgg)
            lpipsa.append(lpips_alex)
    
            # (4) CLIP similarity
            clip_score = compute_clip_similarity(rendered_img, clip_prompt)
            clip_scores.append(clip_score)
    
            # 중간점검
            print("SSIM: ", ssim_score, "CLIP: ", clip_score)

    print("Metrics Completed.")
    
    # === (5) 전체 평균 계산 후 출력 ===
    avg_ssim       = torch.stack(ssims).mean().item()
    avg_psnr       = torch.stack(psnrs).mean().item()
    avg_lpips_vgg  = torch.stack(lpipss).mean().item()
    avg_lpips_alex = torch.stack(lpipsa).mean().item()
    avg_clip       = float(np.mean(clip_scores))


    print("\n=== Final Averages ===")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"PSNR: {avg_psnr:.4f}")
    print(f"LPIPS(VGG): {avg_lpips_vgg:.4f}")
    print(f"LPIPS(Alex): {avg_lpips_alex:.4f}")
    print(f"CLIP similarity with prompt '{clip_prompt}': {avg_clip:.4f}")
