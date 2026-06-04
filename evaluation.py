#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate edited videos with FVD, CLIPScore, and LPIPS.

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


def frames_to_uint8_video_array(
    frames: Sequence[np.ndarray], clip_length: int
) -> np.ndarray:
    if clip_length < 1:
        raise ValueError("--fvd_clip_length must be >= 1")

    if len(frames) < clip_length:
        clips = [np.stack(frames, axis=0)]
    else:
        clips = [
            np.stack(frames[start : start + clip_length], axis=0)
            for start in range(0, len(frames) - clip_length + 1, clip_length)
        ]

    array = np.stack(clips, axis=0)
    return array.astype(np.uint8, copy=False)


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


def compute_fvd(
    original_frames: Sequence[np.ndarray],
    edited_frames: Sequence[np.ndarray],
    device: torch.device,
    model_name: str,
    clip_length: int,
) -> float:
    try:
        from cdfvd import fvd
    except ImportError as exc:
        raise ImportError(
            "Missing cd-fvd FVD implementation. Install with: pip install cd-fvd"
        ) from exc

    original = frames_to_uint8_video_array(original_frames, clip_length)
    edited = frames_to_uint8_video_array(edited_frames, clip_length)

    evaluator = fvd.cdfvd(
        model=model_name,
        n_real="full",
        n_fake="full",
        device=str(device),
    )
    with torch.no_grad():
        score = evaluator.compute_fvd(original, edited)
    return float(score)


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


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    validate_torch()
    device = torch.device(args.device)
    resize = parse_resize(args.resize)
    clip_model = args.clip_model
    if args.clip_backend == "torchmetrics" and clip_model == "ViT-B/32":
        clip_model = "openai/clip-vit-base-patch32"

    original_frames = load_frames(args.original_video, args.max_frames, args.stride, resize)
    edited_frames = load_frames(args.edited_video, args.max_frames, args.stride, resize)
    original_frames, edited_frames = align_frame_counts(original_frames, edited_frames)

    results = {
        "num_frames": len(original_frames),
        "FVD": compute_fvd(
            original_frames,
            edited_frames,
            device,
            args.fvd_model,
            args.fvd_clip_length,
        ),
        "CLIPScore": compute_clipscore(
            edited_frames,
            args.prompt,
            device,
            args.clip_backend,
            clip_model,
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
        description="Evaluate original/edited videos with FVD, CLIPScore, and LPIPS."
    )
    parser.add_argument("--original_video", required=True, help="Original video path or frame directory.")
    parser.add_argument("--edited_video", required=True, help="Edited video path or frame directory.")
    parser.add_argument("--prompt", required=True, help="Text prompt used for CLIPScore.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--max_frames", type=int, default=None, help="Optional frame limit.")
    parser.add_argument("--stride", type=int, default=1, help="Read every Nth frame.")
    parser.add_argument("--resize", default="224x224", help="Resize frames as WIDTHxHEIGHT. Use none to disable.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lpips_net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--fvd_model", choices=["i3d", "videomae"], default="i3d")
    parser.add_argument("--fvd_clip_length", type=int, default=16)
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
