"""
Command-line inference script for PPE Detection
FINAL VERSION – Object-based PPE violation system
"""

import argparse
import sys
from pathlib import Path
import cv2

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.processor import FrameProcessor
from src.utils import load_config, ensure_dir


def main():
    parser = argparse.ArgumentParser(
        description="PPE Compliance Detection – Inference"
    )

    parser.add_argument(
        "--source",
        required=True,
        choices=["image", "video", "webcam"],
        help="Input source type",
    )
    parser.add_argument("--input", help="Input file or folder")
    parser.add_argument("--output", help="Output file or folder")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", help="Override model path")
    parser.add_argument("--skip-frames", type=int, default=0)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--no-save-violations", action="store_true")

    args = parser.parse_args()

    # Validate args
    if args.source in ["image", "video"] and not args.input:
        parser.error("--input is required for image/video")

    # Resolve paths (Windows safe)
    if args.input:
        args.input = str(Path(args.input).resolve())
    if args.output:
        args.output = str(Path(args.output).resolve())
    args.config = str(Path(args.config).resolve())

    # Load configuration
    config = load_config(args.config)

    if args.model:
        config["model"]["weights_path"] = str(Path(args.model).resolve())
    if args.no_save_violations:
        config["storage"]["save_violations"] = False

    model_path = config["model"]["weights_path"]
    if not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    print("Initializing PPE Detection System...")
    processor = FrameProcessor(config, model_path)

    try:
        if args.source == "image":
            process_images(processor, args)
        elif args.source == "video":
            process_video(processor, args)
        elif args.source == "webcam":
            processor.process_webcam(args.camera)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


def process_images(processor, args):
    input_path = Path(args.input)

    # Folder or single image
    if input_path.is_dir():
        images = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        )
        output_dir = Path(args.output or f"{input_path}_results")
        ensure_dir(str(output_dir))
    else:
        images = [input_path]
        output_dir = None

    total_violations = 0

    for i, img in enumerate(images, 1):
        print(f"[{i}/{len(images)}] Processing: {img.name}")

        out_path = (
            output_dir / f"result_{img.name}"
            if output_dir
            else args.output
        )

        result = processor.process_image(
            str(img),
            str(out_path) if out_path else None
        )

        violations = result["violations"]["violation_count"]
        total_violations += violations

        print(f"  - Violations detected: {violations}")
        unverified = result["violations"].get("unverified_candidates", [])
        if unverified:
            print(f"  - Unverified PPE candidates: {len(unverified)}")

        if args.display:
            cv2.imshow("PPE Detection", result["annotated_frame"])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    print("\n" + "=" * 50)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Images processed: {len(images)}")
    print(f"Total violations: {total_violations}")
    print(f"Results saved to: {output_dir or args.output}")
    print("=" * 50)

    if args.display:
        cv2.destroyAllWindows()


def process_video(processor, args):
    stats = processor.process_video(
        args.input,
        output_path=args.output,
        display=args.display,
        skip_frames=args.skip_frames,
    )

    print("\n" + "=" * 50)
    print("VIDEO PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Frames processed: {stats['frames_processed']}")
    print(f"Total violations: {stats['total_violations']}")
    print(f"Frames with violations: {stats['frames_with_violations']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
