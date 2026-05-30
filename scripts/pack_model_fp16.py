import argparse
import json
import os
from pathlib import Path
import tempfile
import zipfile

import torch


def cast_fp16(obj):
    if torch.is_tensor(obj):
        if obj.dtype in (torch.float32, torch.float64):
            return obj.half()
        return obj
    if isinstance(obj, dict):
        return {k: cast_fp16(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cast_fp16(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cast_fp16(v) for v in obj)
    return obj


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def pack(iteration_dir: Path):
    required = [
        "point_cloud.ply",
        "deformation.pth",
        "deformation_table.pth",
        "deformation_accum.pth",
    ]
    optional = ["lite_color_predictor.pth", "point_cloud_meta.json"]

    for name in required:
        if not (iteration_dir / name).exists():
            raise FileNotFoundError(f"Missing required file: {iteration_dir / name}")

    pack_dict = {}
    pack_dict["deformation"] = cast_fp16(torch.load(iteration_dir / "deformation.pth", map_location="cpu"))
    pack_dict["deformation_table"] = torch.load(iteration_dir / "deformation_table.pth", map_location="cpu")
    pack_dict["deformation_accum"] = cast_fp16(torch.load(iteration_dir / "deformation_accum.pth", map_location="cpu"))
    if (iteration_dir / "lite_color_predictor.pth").exists():
        pack_dict["lite_color_predictor"] = cast_fp16(torch.load(iteration_dir / "lite_color_predictor.pth", map_location="cpu"))
    if (iteration_dir / "point_cloud_meta.json").exists():
        with open(iteration_dir / "point_cloud_meta.json", "r", encoding="utf-8") as f:
            pack_dict["point_cloud_meta"] = json.load(f)

    packed_pt = iteration_dir / "packed_fp16.pt"
    packed_zip = iteration_dir / "packed_fp16.zip"
    torch.save(pack_dict, packed_pt)

    with zipfile.ZipFile(packed_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(iteration_dir / "point_cloud.ply", arcname="point_cloud.ply")
        zf.write(packed_pt, arcname="packed_fp16.pt")
    print(f"Packed -> {packed_zip}")


def unpack(zip_path: Path, output_dir: Path):
    ensure_parent(output_dir / "dummy")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(td_path)

        for fname in ("point_cloud.ply", "packed_fp16.pt"):
            src = td_path / fname
            if not src.exists():
                raise FileNotFoundError(f"Corrupt package. Missing: {fname}")

        packed = torch.load(td_path / "packed_fp16.pt", map_location="cpu")
        (output_dir / "point_cloud.ply").write_bytes((td_path / "point_cloud.ply").read_bytes())

        torch.save(packed["deformation"], output_dir / "deformation.pth")
        torch.save(packed["deformation_table"], output_dir / "deformation_table.pth")
        torch.save(packed["deformation_accum"], output_dir / "deformation_accum.pth")
        if "lite_color_predictor" in packed:
            torch.save(packed["lite_color_predictor"], output_dir / "lite_color_predictor.pth")
        if "point_cloud_meta" in packed:
            with open(output_dir / "point_cloud_meta.json", "w", encoding="utf-8") as f:
                json.dump(packed["point_cloud_meta"], f, indent=2)

    print(f"Unpacked -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack/unpack Instruct-4DGS checkpoints.")
    parser.add_argument("mode", choices=["pack", "unpack"])
    parser.add_argument("--path", required=True, help="Iteration directory for pack, zip file path for unpack.")
    parser.add_argument("--output_dir", default="", help="Output directory for unpack.")
    args = parser.parse_args()

    if args.mode == "pack":
        pack(Path(args.path))
    else:
        if not args.output_dir:
            raise ValueError("--output_dir is required in unpack mode.")
        unpack(Path(args.path), Path(args.output_dir))
