#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate a rendered video against a ground-truth video with PSNR, SSIM, and LPIPS.

Example:
    python evaluate_video_quality.py \
        --render_video path/to/render.mp4 \
        --gt_video path/to/ground_truth.mp4
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch


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
            "PyTorch environment before running this script."
        )


def default_device() -> str:
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


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
        raise FileNotFoundError(f"Input video does not exist: {source}")
    if path.is_dir():
        raise ValueError(f"Expected a video file, got a directory: {source}")
    if stride < 1:
        raise ValueError("--stride must be >= 1")

    frames: List[np.ndarray] = []
    for idx, frame in enumerate(iter_video_frames(path)):
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
    render_frames: Sequence[np.ndarray],
    gt_frames: Sequence[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    count = min(len(render_frames), len(gt_frames))
    if count == 0:
        raise ValueError("Both videos must contain at least one frame.")
    return list(render_frames[:count]), list(gt_frames[:count])


def frames_to_float_tensor(frames: Sequence[np.ndarray], device: torch.device) -> torch.Tensor:
    array = np.stack(frames, axis=0)
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float() / 255.0
    return tensor.to(device)


def compute_psnr(
    render_frames: Sequence[np.ndarray],
    gt_frames: Sequence[np.ndarray],
    batch_size: int,
) -> float:
    scores = []
    for start in tqdm(range(0, len(render_frames), batch_size), desc="PSNR"):
        render = np.stack(render_frames[start : start + batch_size], axis=0).astype(np.float32)
        gt = np.stack(gt_frames[start : start + batch_size], axis=0).astype(np.float32)
        mse = np.mean((render - gt) ** 2, axis=(1, 2, 3))
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
    render: torch.Tensor,
    gt: torch.Tensor,
    window: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    channels = render.shape[1]
    padding = window.shape[-1] // 2

    mu_render = torch.nn.functional.conv2d(render, window, padding=padding, groups=channels)
    mu_gt = torch.nn.functional.conv2d(gt, window, padding=padding, groups=channels)

    mu_render_sq = mu_render.pow(2)
    mu_gt_sq = mu_gt.pow(2)
    mu_render_gt = mu_render * mu_gt

    sigma_render_sq = (
        torch.nn.functional.conv2d(render * render, window, padding=padding, groups=channels)
        - mu_render_sq
    )
    sigma_gt_sq = (
        torch.nn.functional.conv2d(gt * gt, window, padding=padding, groups=channels)
        - mu_gt_sq
    )
    sigma_render_gt = (
        torch.nn.functional.conv2d(render * gt, window, padding=padding, groups=channels)
        - mu_render_gt
    )

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = (
        (2.0 * mu_render_gt + c1)
        * (2.0 * sigma_render_gt + c2)
        / ((mu_render_sq + mu_gt_sq + c1) * (sigma_render_sq + sigma_gt_sq + c2))
    )
    return ssim_map.mean(dim=(1, 2, 3))


def compute_ssim(
    render_frames: Sequence[np.ndarray],
    gt_frames: Sequence[np.ndarray],
    device: torch.device,
    batch_size: int,
) -> float:
    window = gaussian_window(window_size=11, sigma=1.5, channels=3, device=device)
    scores = []
    with torch.no_grad():
        for start in tqdm(range(0, len(render_frames), batch_size), desc="SSIM"):
            render = frames_to_float_tensor(render_frames[start : start + batch_size], device)
            gt = frames_to_float_tensor(gt_frames[start : start + batch_size], device)
            scores.append(ssim_batch(render, gt, window).detach().cpu())
    return torch.cat(scores).mean().item()


def compute_lpips(
    render_frames: Sequence[np.ndarray],
    gt_frames: Sequence[np.ndarray],
    device: torch.device,
    net: str,
    batch_size: int,
) -> float:
    lpips = require_package("lpips", "pip install lpips")
    model = lpips.LPIPS(net=net).to(device).eval()

    render = frames_to_float_tensor(render_frames, device) * 2.0 - 1.0
    gt = frames_to_float_tensor(gt_frames, device) * 2.0 - 1.0

    scores = []
    with torch.no_grad():
        for start in tqdm(range(0, render.shape[0], batch_size), desc="LPIPS"):
            end = start + batch_size
            score = model(render[start:end], gt[start:end])
            scores.append(score.detach().flatten().cpu())
    return torch.cat(scores).mean().item()


def parse_resize(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if value.lower() == "none":
        return None
    if "x" not in value.lower():
        raise argparse.ArgumentTypeError("Resize must be formatted as WIDTHxHEIGHT.")
    width_str, height_str = value.lower().split("x", 1)
    width, height = int(width_str), int(height_str)
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Resize width and height must be positive.")
    return width, height


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    validate_torch()
    device = torch.device(args.device)

    render_frames = load_frames(args.render_video, args.max_frames, args.stride, args.resize)
    gt_frames = load_frames(args.gt_video, args.max_frames, args.stride, args.resize)
    render_frames, gt_frames = align_frame_counts(render_frames, gt_frames)

    return {
        "num_frames": len(render_frames),
        "PSNR": compute_psnr(render_frames, gt_frames, args.batch_size),
        "SSIM": compute_ssim(render_frames, gt_frames, device, args.batch_size),
        "LPIPS": compute_lpips(render_frames, gt_frames, device, args.lpips_net, args.batch_size),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate rendered and ground-truth videos with PSNR, SSIM, and LPIPS."
    )
    parser.add_argument("--render_video", required=True, help="Rendered video path.")
    parser.add_argument("--gt_video", required=True, help="Ground-truth video path.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--max_frames", type=int, default=None, help="Optional frame limit.")
    parser.add_argument("--stride", type=int, default=1, help="Read every Nth frame.")
    parser.add_argument(
        "--resize",
        type=parse_resize,
        default=parse_resize("1024x1024"),
        help="Resize frames as WIDTHxHEIGHT. Use none to disable.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lpips_net", choices=["alex", "vgg", "squeeze"], default="alex")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
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
    main()
