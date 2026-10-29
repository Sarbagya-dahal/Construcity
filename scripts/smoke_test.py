"""Run the existing CLI on the local demonstration images."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description="PPE Detection CLI smoke test")
    parser.add_argument("--output-dir", default="outputs/smoke_test")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for input_path in (root / "demo_images" / "no_helmet.jpg", root / "demo_images" / "no_vest.jpg"):
        output_path = output_dir / f"{input_path.stem}_result.jpg"
        command = [
            sys.executable,
            "-m",
            "src.inference",
            "--source",
            "image",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--no-save-violations",
        ]
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
        image = cv2.imread(str(output_path)) if output_path.exists() else None
        result = {
            "input": str(input_path),
            "output": str(output_path),
            "exit_code": completed.returncode,
            "output_exists": output_path.exists(),
            "output_readable": image is not None,
        }
        results.append(result)
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)

    report_path = output_dir / "smoke_test_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0 if all(item["exit_code"] == 0 and item["output_readable"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
