"""Quick diagnostic: run all demo images and print raw/filtered detections."""
from __future__ import annotations
import sys, cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.processor import FrameProcessor
from src.utils import load_config

config = load_config(str(ROOT / "config" / "config.yaml"))
config["storage"]["save_violations"] = False
processor = FrameProcessor(config, str(ROOT / "models/trained/best.pt"))

demo_dir = ROOT / "demo_images"
for img_path in sorted(demo_dir.glob("*.jpg")):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"SKIP {img_path.name}: unreadable")
        continue
    result = processor.process_frame(img, save_violation=False, source_name=img_path.name)
    viol = result["violations"]
    print(f"--- {img_path.name} ({img.shape[1]}x{img.shape[0]}) ---")
    print(f"  persons={viol['person_count']}  helmets={viol['helmet_count']}  vests={viol['vest_count']}")
    print(f"  status={viol['status']}  violations={viol['violation_count']}")
    for pr in viol.get("person_results", []):
        print(f"  person {pr['person_id']}: has_helmet={pr['has_helmet']}  "
              f"has_vest={pr['has_vest']}  violations={pr['violations']}  "
              f"uncertain={pr['uncertain']}  review={pr['review_reasons']}")
    diag = processor.detector.last_diagnostics
    raw_ppe = diag.get("ppe", {}).get("raw", {})
    for cls, items in raw_ppe.items():
        for item in items:
            det = item["detection"]
            rej = item["reasons"]
            print(f"  RAW {cls}: conf={det['confidence']:.3f} "
                  f"bbox={[round(x,1) for x in det['bbox']]} rejected={rej}")
    for rejection in diag.get("ppe", {}).get("helmet_post_filter_rejections", []):
        det = rejection["detection"]
        rej = rejection["reasons"]
        print(f"  POST-REJECT helmet: conf={det['confidence']:.3f} "
              f"bbox={[round(x,1) for x in det['bbox']]} reasons={rej}")
    print()

