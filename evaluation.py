#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate edited videos with CLIP, perceptual, sharpness, frequency, flow, and distortion metrics.

Example:
    python evaluation.py \
        --original_video path/to/original.mp4 \
        --edited_video path/to/edited.mp4 \
        --prompt "Make it look like a fauvism painting"

The script also accepts directories of frames for --original_video and
--edited_video. Frame files are matched by sorted order.

"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import random


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def require_package(import_name: str, install_hint: str):
    try:
        return __import__(import_name)
    except ImportError as exc:
        raise ImportError(
            f"Missing dependency '{import_name}'. Install it with: {install_hint}"
        ) from exc


def validate_torch() -> None:
    missing = [name for name in ("device", "from_numpy", "no_grad", "cat") if not hasattr(torch, name)]
    if missing:
        raise ImportError(
            "PyTorch is not imported correctly in this environment "
            f"(missing: {', '.join(missing)}). Please activate/install a working "
            "PyTorch environment before running evaluation.py."
        )


def default_device() -> str:
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sorted_frame_paths(frame_dir: Path) -> List[Path]:
    paths = [
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort()
    if not paths:
        raise ValueError(f"No image frames found in directory: {frame_dir}")
    return paths


def load_rgb_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def iter_video_frames(path: Path) -> Iterable[np.ndarray]:
    reader = imageio.get_reader(str(path))
    try:
        for frame in reader:
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            yield frame.astype(np.uint8)
    finally:
        reader.close()


def load_frames(
    source: str,
    max_frames: Optional[int],
    stride: int,
    resize: Optional[Tuple[int, int]],
) -> List[np.ndarray]:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {source}")

    if stride < 1:
        raise ValueError("--stride must be >= 1")

    frames: List[np.ndarray] = []
    if path.is_dir():
        frame_iter = (load_rgb_image(frame_path) for frame_path in sorted_frame_paths(path))
    else:
        frame_iter = iter_video_frames(path)

    for idx, frame in enumerate(frame_iter):
        if idx % stride != 0:
            continue
        image = Image.fromarray(frame).convert("RGB")
        if resize is not None:
            image = image.resize(resize, Image.BICUBIC)
        frames.append(np.asarray(image, dtype=np.uint8))
        if max_frames is not None and len(frames) >= max_frames:
            break

    if not frames:
        raise ValueError(f"No frames were loaded from: {source}")
    return frames


def align_frame_counts(
    original_frames: Sequence[np.ndarray], edited_frames: Sequence[np.ndarray]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    count = min(len(original_frames), len(edited_frames))
    if count == 0:
        raise ValueError("Both videos must contain at least one frame.")
    return list(original_frames[:count]), list(edited_frames[:count])


def frames_to_float_tensor(frames: Sequence[np.ndarray], device: torch.device) -> torch.Tensor:
    array = np.stack(frames, axis=0)
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float() / 255.0
    return tensor.to(device)


def compute_psnr(
    original_frames: Sequence[np.ndarray],
    edited_frames: Sequence[np.ndarray],
    batch_size: int,
) -> float:
    scores = []
    for start in tqdm(range(0, len(original_frames), batch_size), desc="PSNR"):
        original = np.stack(original_frames[start : start + batch_size], axis=0).astype(np.float32)
        edited = np.stack(edited_frames[start : start + batch_size], axis=0).astype(np.float32)
        mse = np.mean((original - edited) ** 2, axis=(1, 2, 3))
        batch_scores = np.full_like(mse, np.inf, dtype=np.float32)
        nonzero = mse > 0.0
        batch_scores[nonzero] = 20.0 * np.log10(255.0) - 10.0 * np.log10(mse[nonzero])
        scores.append(batch_scores)
    return float(np.concatenate(scores, axis=0).mean())


def gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    gaussian = torch.exp(-(coords**2) / (2.0 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    window_2d = torch.outer(gaussian, gaussian)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim_batch(
    original: torch.Tensor,
    edited: torch.Tensor,
    window: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    channels = original.shape[1]
    padding = window.shape[-1] // 2

    mu_original = torch.nn.functional.conv2d(original, window, padding=padding, groups=channels)
    mu_edited = torch.nn.functional.conv2d(edited, window, padding=padding, groups=channels)

    mu_original_sq = mu_original.pow(2)
    mu_edited_sq = mu_edited.pow(2)
    mu_original_edited = mu_original * mu_edited

    sigma_original_sq = (
        torch.nn.functional.conv2d(original * original, window, padding=padding, groups=channels)
        - mu_original_sq
    )
    sigma_edited_sq = (
        torch.nn.functional.conv2d(edited * edited, window, padding=padding, groups=channels)
        - mu_edited_sq
    )
    sigma_original_edited = (
        torch.nn.functional.conv2d(original * edited, window, padding=padding, groups=channels)
        - mu_original_edited
    )

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = (
        (2.0 * mu_original_edited + c1)
        * (2.0 * sigma_original_edited + c2)
        / ((mu_original_sq + mu_edited_sq + c1) * (sigma_original_sq + sigma_edited_sq + c2))
    )
    return ssim_map.mean(dim=(1, 2, 3))


def compute_ssim(
    original_frames: Sequence[np.ndarray],
    edited_frames: Sequence[np.ndarray],
    device: torch.device,
    batch_size: int,
) -> float:
    window = gaussian_window(window_size=11, sigma=1.5, channels=3, device=device)
    scores = []
    with torch.no_grad():
        for start in tqdm(range(0, len(original_frames), batch_size), desc="SSIM"):
            original = frames_to_float_tensor(original_frames[start : start + batch_size], device)
            edited = frames_to_float_tensor(edited_frames[start : start + batch_size], device)
            scores.append(ssim_batch(original, edited, window).detach().cpu())
    return torch.cat(scores).mean().item()


def compute_lpips(
    original_frames: Sequence[np.ndarray],
    edited_frames: Sequence[np.ndarray],
    device: torch.device,
    net: str,
    batch_size: int,
) -> float:
    lpips = require_package("lpips", "pip install lpips")
    model = lpips.LPIPS(net=net).to(device).eval()

    original = frames_to_float_tensor(original_frames, device) * 2.0 - 1.0
    edited = frames_to_float_tensor(edited_frames, device) * 2.0 - 1.0

    scores = []
    with torch.no_grad():
        for start in tqdm(range(0, original.shape[0], batch_size), desc="LPIPS"):
            end = start + batch_size
            score = model(original[start:end], edited[start:end])
            scores.append(score.detach().flatten().cpu())
    return torch.cat(scores).mean().item()


def frame_to_gray(frame: np.ndarray, cv2_module) -> np.ndarray:
    cv2 = cv2_module
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def estimate_flow(prev_frame: np.ndarray, next_frame: np.ndarray, cv2_module) -> np.ndarray:
    cv2 = cv2_module
    prev_gray = frame_to_gray(prev_frame, cv2)
    next_gray = frame_to_gray(next_frame, cv2)
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    ).astype(np.float32, copy=False)


def backward_warp_batch(
    images: torch.Tensor,
    flows: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, _, height, width = images.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=images.device, dtype=torch.float32),
        torch.arange(width, device=images.device, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
    sample_grid = grid + flows

    valid = (
        (sample_grid[..., 0] >= 0.0)
        & (sample_grid[..., 0] <= width - 1)
        & (sample_grid[..., 1] >= 0.0)
        & (sample_grid[..., 1] <= height - 1)
    )

    if width > 1:
        sample_grid[..., 0] = sample_grid[..., 0] / (width - 1) * 2.0 - 1.0
    else:
        sample_grid[..., 0] = 0.0
    if height > 1:
        sample_grid[..., 1] = sample_grid[..., 1] / (height - 1) * 2.0 - 1.0
    else:
        sample_grid[..., 1] = 0.0

    warped = torch.nn.functional.grid_sample(
        images,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped, valid.unsqueeze(1).float()


def compute_variance_of_laplacian(
    frames: Sequence[np.ndarray],
    batch_size: int,
) -> float:
    cv2 = require_package("cv2", "pip install opencv-python")
    scores = []
    for start in tqdm(range(0, len(frames), batch_size), desc="Variance of Laplacian"):
        for frame in frames[start : start + batch_size]:
            gray = frame_to_gray(frame, cv2)
            scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(np.mean(scores))


def compute_high_frequency_energy(
    frames: Sequence[np.ndarray],
    batch_size: int,
    cutoff: float,
) -> float:
    cv2 = require_package("cv2", "pip install opencv-python")
    if not 0.0 <= cutoff < 1.0:
        raise ValueError("--high_freq_cutoff must be in [0, 1).")

    scores = []
    for start in tqdm(range(0, len(frames), batch_size), desc="High-Frequency Energy"):
        for frame in frames[start : start + batch_size]:
            gray = frame_to_gray(frame, cv2).astype(np.float32) / 255.0
            spectrum = np.fft.fftshift(np.fft.fft2(gray))
            power = np.abs(spectrum) ** 2
            height, width = gray.shape
            yy, xx = np.ogrid[:height, :width]
            cy, cx = height // 2, width // 2
            radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            max_radius = np.sqrt(cy**2 + cx**2)
            high_freq_mask = radius >= cutoff * max_radius
            scores.append(power[high_freq_mask].sum() / np.maximum(power.sum(), 1e-8))
    return float(np.mean(scores))


def compute_warp_lpips(
    edited_frames: Sequence[np.ndarray],
    device: torch.device,
    net: str,
    batch_size: int,
) -> float:
    if len(edited_frames) < 2:
        raise ValueError("Warp LPIPS requires at least two frames.")

    cv2 = require_package("cv2", "pip install opencv-python")
    lpips = require_package("lpips", "pip install lpips")
    model = lpips.LPIPS(net=net).to(device).eval()
    pair_count = len(edited_frames) - 1
    scores = []

    with torch.no_grad():
        for start in tqdm(range(0, pair_count, batch_size), desc="Warp LPIPS"):
            end = min(start + batch_size, pair_count)
            prev_edited = frames_to_float_tensor(edited_frames[start:end], device)
            next_edited = frames_to_float_tensor(edited_frames[start + 1 : end + 1], device)
            backward_flows = [
                estimate_flow(edited_frames[idx + 1], edited_frames[idx], cv2)
                for idx in range(start, end)
            ]
            flows = torch.from_numpy(np.stack(backward_flows, axis=0)).to(device)
            warped_prev, valid = backward_warp_batch(prev_edited, flows)
            warped_prev = warped_prev * valid + next_edited * (1.0 - valid)
            score = model(warped_prev * 2.0 - 1.0, next_edited * 2.0 - 1.0)
            scores.append(score.detach().flatten().cpu())

    return torch.cat(scores).mean().item()


def compute_flow_consistency(
    original_frames: Sequence[np.ndarray],
    edited_frames: Sequence[np.ndarray],
    batch_size: int,
) -> float:
    if len(original_frames) < 2:
        raise ValueError("Flow Consistency requires at least two frames.")

    cv2 = require_package("cv2", "pip install opencv-python")
    pair_count = len(original_frames) - 1
    scores = []

    for start in tqdm(range(0, pair_count, batch_size), desc="Flow Consistency"):
        end = min(start + batch_size, pair_count)
        for idx in range(start, end):
            original_flow = estimate_flow(original_frames[idx], original_frames[idx + 1], cv2)
            edited_flow = estimate_flow(edited_frames[idx], edited_frames[idx + 1], cv2)
            endpoint_error = np.linalg.norm(original_flow - edited_flow, axis=-1)
            scores.append(endpoint_error.mean())

    return float(np.mean(scores))


def compute_clipscore_openai_clip(
    edited_frames: Sequence[np.ndarray],
    prompt: str,
    device: torch.device,
    clip_model_name: str,
    batch_size: int,
) -> float:
    clip = require_package(
        "clip",
        "pip install git+https://github.com/openai/CLIP.git",
    )
    model, preprocess = clip.load(clip_model_name, device=device)
    model.eval()

    text_tokens = clip.tokenize([prompt], truncate=True).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    scores = []
    with torch.no_grad():
        for start in tqdm(range(0, len(edited_frames), batch_size), desc="CLIPScore"):
            batch_frames = edited_frames[start : start + batch_size]
            images = [
                preprocess(Image.fromarray(frame).convert("RGB"))
                for frame in batch_frames
            ]
            image_input = torch.stack(images, dim=0).to(device)
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            score = (image_features @ text_features.T).squeeze(1)
            scores.append(score.detach().float().cpu())
    return torch.cat(scores).mean().item()


def compute_clipscore_torchmetrics(
    edited_frames: Sequence[np.ndarray],
    prompt: str,
    device: torch.device,
    clip_model_name: str,
    batch_size: int,
) -> float:
    try:
        from torchmetrics.multimodal.clip_score import CLIPScore
    except ImportError as exc:
        raise ImportError(
            "Missing torchmetrics CLIPScore dependencies. Install with: "
            "pip install torchmetrics transformers"
        ) from exc

    metric = CLIPScore(model_name_or_path=clip_model_name).to(device)
    with torch.no_grad():
        for start in tqdm(range(0, len(edited_frames), batch_size), desc="CLIPScore"):
            batch_frames = edited_frames[start : start + batch_size]
            images = torch.from_numpy(np.stack(batch_frames, axis=0)).permute(0, 3, 1, 2)
            images = images.to(device=device, dtype=torch.uint8)
            prompts = [prompt] * images.shape[0]
            metric.update(images, prompts)
    return metric.compute().detach().float().cpu().item()


def compute_clipscore(
    edited_frames: Sequence[np.ndarray],
    prompt: str,
    device: torch.device,
    backend: str,
    clip_model_name: str,
    batch_size: int,
) -> float:
    if backend == "openai-clip":
        return compute_clipscore_openai_clip(
            edited_frames, prompt, device, clip_model_name, batch_size
        )
    if backend == "torchmetrics":
        return compute_clipscore_torchmetrics(
            edited_frames, prompt, device, clip_model_name, batch_size
        )
    raise ValueError(f"Unknown CLIP backend: {backend}")


def compute_clip_directional_similarity(
    original_frames: Sequence[np.ndarray],
    edited_frames: Sequence[np.ndarray],
    source_prompt: str,
    target_prompt: str,
    device: torch.device,
    clip_model_name: str,
    batch_size: int,
) -> float:
    clip = require_package(
        "clip",
        "pip install git+https://github.com/openai/CLIP.git",
    )
    model, preprocess = clip.load(clip_model_name, device=device)
    model.eval()

    with torch.no_grad():
        text_tokens = clip.tokenize(
            [source_prompt, target_prompt],
            truncate=True,
        ).to(device)
        text_features = model.encode_text(text_tokens).float()
        text_direction = text_features[1] - text_features[0]
        text_direction = text_direction / text_direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        scores = []
        for start in tqdm(
            range(0, len(original_frames), batch_size),
            desc="CLIP Directional Similarity",
        ):
            original_batch = original_frames[start : start + batch_size]
            edited_batch = edited_frames[start : start + batch_size]
            original_images = [
                preprocess(Image.fromarray(frame).convert("RGB"))
                for frame in original_batch
            ]
            edited_images = [
                preprocess(Image.fromarray(frame).convert("RGB"))
                for frame in edited_batch
            ]
            image_input = torch.stack(original_images + edited_images, dim=0).to(device)
            image_features = model.encode_image(image_input).float()
            original_features, edited_features = image_features.chunk(2, dim=0)
            image_direction = edited_features - original_features
            image_direction = image_direction / image_direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            score = image_direction @ text_direction.unsqueeze(1)
            scores.append(score.squeeze(1).detach().cpu())

    return torch.cat(scores).mean().item()


def parse_resize(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if "x" not in value.lower():
        raise argparse.ArgumentTypeError("Resize must be formatted as WIDTHxHEIGHT.")
    width_str, height_str = value.lower().split("x", 1)
    width, height = int(width_str), int(height_str)
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Resize width and height must be positive.")
    return width, height

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    validate_torch()
    device = torch.device(args.device)
    resize = parse_resize(args.resize)
    clipscore_model = args.clip_model
    if args.clip_backend == "torchmetrics" and clipscore_model == "ViT-B/32":
        clipscore_model = "openai/clip-vit-base-patch32"

    original_frames = load_frames(args.original_video, args.max_frames, args.stride, resize)
    edited_frames = load_frames(args.edited_video, args.max_frames, args.stride, resize)
    original_frames, edited_frames = align_frame_counts(original_frames, edited_frames)

    results = {
        "num_frames": len(original_frames),
        "PSNR": compute_psnr(
            original_frames,
            edited_frames,
            args.batch_size,
        ),
        "SSIM": compute_ssim(
            original_frames,
            edited_frames,
            device,
            args.batch_size,
        ),
        "Variance of Laplacian": compute_variance_of_laplacian(
            edited_frames,
            args.batch_size,
        ),
        "High-Frequency Energy": compute_high_frequency_energy(
            edited_frames,
            args.batch_size,
            args.high_freq_cutoff,
        ),
        "Warp LPIPS": compute_warp_lpips(
            edited_frames,
            device,
            args.lpips_net,
            args.batch_size,
        ),
        "Flow Consistency": compute_flow_consistency(
            original_frames,
            edited_frames,
            args.batch_size,
        ),
        "CLIP Directional Similarity": compute_clip_directional_similarity(
            original_frames,
            edited_frames,
            args.source_prompt,
            args.prompt,
            device,
            args.clip_directional_model,
            args.batch_size,
        ),
        "CLIPScore": compute_clipscore(
            edited_frames,
            args.prompt,
            device,
            args.clip_backend,
            clipscore_model,
            args.batch_size,
        ),
        "LPIPS": compute_lpips(
            original_frames,
            edited_frames,
            device,
            args.lpips_net,
            args.batch_size,
        ),
    }
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate original/edited videos with CLIP, perceptual, sharpness, frequency, flow, and distortion metrics."
    )
    parser.add_argument("--original_video", required=True, help="Original video path or frame directory.")
    parser.add_argument("--edited_video", required=True, help="Edited video path or frame directory.")
    parser.add_argument("--prompt", required=True, help="Target text prompt used for CLIPScore and directional similarity.")
    parser.add_argument(
        "--source_prompt",
        default="a video of a man cooking spinach",
        help="Source text prompt used for CLIP Directional Similarity.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--max_frames", type=int, default=None, help="Optional frame limit.")
    parser.add_argument("--stride", type=int, default=1, help="Read every Nth frame.")
    parser.add_argument("--resize", default="1024x1024", help="Resize frames as WIDTHxHEIGHT. Use none to disable.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lpips_net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument(
        "--high_freq_cutoff",
        type=float,
        default=0.25,
        help="Normalized radial FFT cutoff for High-Frequency Energy.",
    )
    parser.add_argument(
        "--clip_backend",
        choices=["openai-clip", "torchmetrics"],
        default="openai-clip",
    )
    parser.add_argument(
        "--clip_model",
        default="ViT-B/32",
        help=(
            "For openai-clip use names like ViT-B/32. "
            "For torchmetrics use HuggingFace names like openai/clip-vit-base-patch32."
        ),
    )
    parser.add_argument(
        "--clip_directional_model",
        default="ViT-B/32",
        help="OpenAI CLIP model name used for CLIP Directional Similarity.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.resize and args.resize.lower() == "none":
        args.resize = None

    results = evaluate(args)

    print("\n=== Evaluation Results ===")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
