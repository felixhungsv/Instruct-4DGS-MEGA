import argparse
import json
from pathlib import Path
import re


PROMPTS = [
    "Make it look like a fauvism painting",
    "Make it look like a sculpture",
    "Turn the man into a woman",
]


def parse_log(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    mode_match = re.search(r"color_mode=(\w+), entropy=([0-9.]+)", text)
    mode = mode_match.group(1) if mode_match else "unknown"
    entropy = mode_match.group(2) if mode_match else "unknown"
    runtimes = [int(x) for x in re.findall(r"Prompt runtime \(sec\): (\d+)", text)]
    return {"mode": mode, "entropy": entropy, "runtimes": runtimes}


def summarize(log_dir: Path):
    rows = []
    for log_path in sorted(log_dir.glob("*.log")):
        parsed = parse_log(log_path)
        ts_match = re.search(r"_(\d{8}_\d{6})\.log$", log_path.name)
        ts = ts_match.group(1) if ts_match else ""
        size_report = log_dir / f"modelsize_{parsed['mode']}_{ts}.json"
        size = {}
        if size_report.exists():
            size = json.loads(size_report.read_text())
        for i, prompt in enumerate(PROMPTS):
            rows.append(
                {
                    "prompt": prompt,
                    "mode": parsed["mode"],
                    "entropy": parsed["entropy"],
                    "runtime_sec": parsed["runtimes"][i] if i < len(parsed["runtimes"]) else None,
                    "total_mb": size.get("total_mb"),
                    "packed_zip_mb": size.get("packed_fp16.zip"),
                    "log": log_path.name,
                }
            )
    return rows


def to_markdown(rows):
    header = "| prompt | mode | entropy | runtime_sec | total_mb | packed_zip_mb | log |\n|---|---:|---:|---:|---:|---:|---|"
    body = []
    for r in rows:
        body.append(
            f"| {r['prompt']} | {r['mode']} | {r['entropy']} | {r['runtime_sec']} | "
            f"{None if r['total_mb'] is None else round(r['total_mb'], 2)} | "
            f"{None if r['packed_zip_mb'] is None else round(r['packed_zip_mb'], 2)} | {r['log']} |"
        )
    return "\n".join([header] + body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize hybrid benchmark logs.")
    parser.add_argument("--log_dir", required=True, type=str)
    parser.add_argument("--output_md", default="", type=str)
    args = parser.parse_args()

    rows = summarize(Path(args.log_dir))
    md = to_markdown(rows)
    print(md)
    if args.output_md:
        Path(args.output_md).write_text(md, encoding="utf-8")
