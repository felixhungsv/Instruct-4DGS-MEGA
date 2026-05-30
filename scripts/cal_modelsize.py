import argparse
import json
import os
from pathlib import Path


def mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def summarize_dir(target: Path):
    keys = [
        "point_cloud.ply",
        "deformation.pth",
        "deformation_table.pth",
        "deformation_accum.pth",
        "lite_color_predictor.pth",
        "point_cloud_meta.json",
        "packed_fp16.pt",
        "packed_fp16.zip",
    ]
    out = {k: 0.0 for k in keys}
    total = 0.0
    for p in target.rglob("*"):
        if not p.is_file():
            continue
        size = mb(p)
        total += size
        if p.name in out:
            out[p.name] += size
    out["total_mb"] = total
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize model artifact sizes.")
    parser.add_argument("--model_root", required=True, type=str)
    parser.add_argument("--report_path", default="", type=str)
    args = parser.parse_args()

    summary = summarize_dir(Path(args.model_root))
    print(json.dumps(summary, indent=2))
    if args.report_path:
        with open(args.report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
