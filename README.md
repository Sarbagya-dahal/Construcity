# PPE Detection App

PPE (Personal Protective Equipment) Compliance Detection system built with YOLOv8, OpenCV, and Streamlit.

## What This Project Does
- Detects PPE classes from images/video/webcam:
  - `helmet`
  - `vest`
  - `without_helmet`
  - `without_vest`
- Flags violations when `without_helmet` or `without_vest` are detected.
- Saves annotated violation frames to `violations/`.
- Supports:
  - CLI inference (`image`, `video`, `webcam`)
  - Model training with Ultralytics YOLOv8
  - Streamlit dashboard UI (see notes below)

## Project Structure
```text
ppe_detection_app/
|-- app/
|   `-- dashboard.py
|-- config/
|   `-- config.yaml
|-- data/
|   `-- Construction Site.v1i.yolov8/
|       |-- data.yaml
|       |-- train/
|       |-- valid/
|       `-- test/
|-- models/
|   `-- trained/
|       `-- best.pt
|-- src/
|   |-- __main__.py
|   |-- detector.py
|   |-- inference.py
|   |-- processor.py
|   |-- train.py
|   |-- utils.py
|   `-- violation_logic.py
|-- requirements.txt
|-- pyproject.toml
|-- run_inference.bat
`-- README.md
```

## Requirements
- Python 3.8+
- `pip`
- Optional GPU: CUDA-compatible PyTorch

## Installation
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Configuration
Main config file: `config/config.yaml`

Key settings:
- `model.weights_path`: model file path (default: `models/trained/best.pt`)
- `model.confidence_threshold`: detection confidence threshold
- `model.iou_threshold`: YOLO NMS IoU threshold
- `model.device`: `cpu` or `cuda`
- `storage.violations_dir`: where violation images are stored
- `storage.save_violations`: enable/disable saving violation frames

## Run Inference (CLI)
The main entry point is `src/inference.py`.

### 1) Image (single file)
```bash
python -m src.inference --source image --input path/to/image.jpg --output path/to/result.jpg
```

### 2) Image folder (batch)
```bash
python -m src.inference --source image --input path/to/images --output path/to/output_folder
```

### 3) Video
```bash
python -m src.inference --source video --input path/to/video.mp4 --output path/to/output.mp4
```

### 4) Webcam
```bash
python -m src.inference --source webcam --camera 0
```

Useful flags:
```bash
--display
--skip-frames 5
--model models/trained/best.pt
--config config/config.yaml
--no-save-violations
```

### Windows shortcut
```bat
run_inference.bat image "path/to/images" "path/to/output"
run_inference.bat video "input.mp4" "output.mp4"
run_inference.bat webcam
```

## Train the Model
Training script: `src/train.py`

Validate dataset only:
```bash
python src/train.py --validate-only
```

Train:
```bash
python src/train.py \
  --data "data/Construction Site.v1i.yolov8/data.yaml" \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device cpu \
  --project models \
  --name ppe_detection
```

After training, copy best weights to:
- `models/trained/best.pt`

## Dashboard (Streamlit)
Run:
```bash
streamlit run app/dashboard.py
```

The dashboard uses the same `FrameProcessor` pipeline as the CLI and provides
Overview, Image Analysis, Batch Testing, Evaluation, and How It Works pages.
Uploads are processed locally. Run it from the project root after activating
`.venv`.

## Tests and Evaluation (Windows PowerShell)
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\smoke_test.py
.\.venv\Scripts\python.exe scripts\evaluate_dataset.py
```

The evaluation command reads the polygon labels in
`data/Construction Site.v1i.yolov8/test/`, converts them to enclosing boxes,
and writes `outputs/evaluation_report.json`. It reports raw PPE-model metrics,
filtered pipeline PPE metrics, and a separate end-to-end violation summary.

## Dataset
This repo includes a YOLO-format dataset in:
- `data/Construction Site.v1i.yolov8/`

Classes from `data.yaml`:
- `helmet`
- `vest`
- `without_helmet`
- `without_vest`

### Helmet data collection and isolated experiments

The production checkpoint is a four-class YOLO **detection** model:
`models/trained/best.pt`. The source labels use YOLO polygon coordinates;
Ultralytics converts those polygons to object boxes for this detection model.
Keep the existing class IDs unchanged:

| ID | Class | Annotation target |
|---:|---|---|
| 0 | `helmet` | The physical safety-helmet object itself, whether worn, held, or resting nearby. Do not label hair or an ordinary cap as a helmet. |
| 1 | `vest` | The visible reflective/safety vest object. Label it even when the image also contains a missing-helmet example. |
| 2 | `without_helmet` | The visible head/head region of a person who is not wearing a safety helmet. Do not surround the whole person or a helmet held nearby. |
| 3 | `without_vest` | The visible torso/person region representing the dataset's existing missing-vest convention. Review against nearby labels before adding examples. |

For helmet work, collect and independently review examples of bare heads with
different hair colours and bald scalps, caps/beanies/headphones, helmets held
in hands or resting nearby, and genuine worn helmets across colour, angle,
distance, lighting, and occlusion. Label every visible relevant object so an
unlabelled vest or helmet is not treated as background. Keep reviewed images
outside the official validation and test splits.

The repeatable workflow is manifest-gated. A human must change an entry in
`data/hard_negatives/review_manifest.json` from `manual_review_required` to
`reviewed` only after inspecting the image and label. The preparation command
requires at least 12 reviewed images and all required scene categories; it
also rejects exact and perceptual near-duplicates of the official splits.
The current single fixture intentionally fails this sufficiency check.

Run these commands in Windows PowerShell:

```powershell
Set-Location 'C:\Users\Sarbagya\Downloads\ppe_detection'
& 'C:\Users\Sarbagya\Downloads\ppe_detection\.venv\Scripts\python.exe' `
  'C:\Users\Sarbagya\Downloads\ppe_detection\scripts\validate_hard_negatives.py'

# After manual review, update review_manifest.json and rerun preparation.
& 'C:\Users\Sarbagya\Downloads\ppe_detection\.venv\Scripts\python.exe' `
  'C:\Users\Sarbagya\Downloads\ppe_detection\scripts\prepare_helmet_experiment.py' `
  --dataset 'C:\Users\Sarbagya\Downloads\ppe_detection\data\Construction Site.v1i.yolov8' `
  --hard-negatives 'C:\Users\Sarbagya\Downloads\ppe_detection\data\hard_negatives' `
  --manifest 'C:\Users\Sarbagya\Downloads\ppe_detection\data\hard_negatives\review_manifest.json' `
  --output 'C:\Users\Sarbagya\Downloads\ppe_detection\data\helmet_reviewed_experiment'

# Train only the isolated experiment; this never overwrites best.pt.
& 'C:\Users\Sarbagya\Downloads\ppe_detection\.venv\Scripts\python.exe' `
  'C:\Users\Sarbagya\Downloads\ppe_detection\scripts\train_helmet_candidate.py' `
  --data 'C:\Users\Sarbagya\Downloads\ppe_detection\data\helmet_reviewed_experiment\data.yaml' `
  --weights 'C:\Users\Sarbagya\Downloads\ppe_detection\models\trained\best.pt' `
  --project 'C:\Users\Sarbagya\Downloads\ppe_detection\runs' `
  --name 'helmet_reviewed_candidate' --epochs 50 --imgsz 640 --batch 8 --device cpu

# Select a candidate using validation only.
& 'C:\Users\Sarbagya\Downloads\ppe_detection\.venv\Scripts\python.exe' `
  'C:\Users\Sarbagya\Downloads\ppe_detection\scripts\evaluate_dataset.py' `
  --dataset 'C:\Users\Sarbagya\Downloads\ppe_detection\data\helmet_reviewed_experiment' `
  --split valid `
  --weights 'C:\Users\Sarbagya\Downloads\ppe_detection\runs\helmet_reviewed_candidate\weights\best.pt' `
  --output 'C:\Users\Sarbagya\Downloads\ppe_detection\outputs\helmet_candidate_valid.json'

# Only after candidate selection, evaluate the untouched test split.
& 'C:\Users\Sarbagya\Downloads\ppe_detection\.venv\Scripts\python.exe' `
  'C:\Users\Sarbagya\Downloads\ppe_detection\scripts\evaluate_dataset.py' `
  --dataset 'C:\Users\Sarbagya\Downloads\ppe_detection\data\helmet_reviewed_experiment' `
  --split test `
  --weights 'C:\Users\Sarbagya\Downloads\ppe_detection\runs\helmet_reviewed_candidate\weights\best.pt' `
  --output 'C:\Users\Sarbagya\Downloads\ppe_detection\outputs\helmet_candidate_test.json'
```

Do not promote a candidate based only on a personal demonstration image or a
passing unit test. Compare raw and filtered per-class precision, recall, F1,
mAP, false-positive helmet detections, and missed genuine helmets on the same
validation split before looking at the held-out test results.

### Optional head-region helmet classifier

The project also contains a separate two-class head-crop classifier pipeline.
It is disabled by default with `head_classifier.enabled: false`, so the
production PPE detector and dashboard fallback remain unchanged. It uses
`helmet_worn` and `no_helmet` ImageFolder classes and a face-anchored head
localizer. If a face cannot be localized, the integration returns
`needs_review`; it does not silently assume that the top of a person box is a
head. A held or nearby helmet is outside the head crop and cannot establish
that the person is wearing one.

The installed environment has CPU PyTorch and torchvision. The current
repository does not yet contain enough manually reviewed head crops to train a
candidate. Check the empty split folders and manifest with:

```powershell
Set-Location 'C:\Users\Sarbagya\Downloads\ppe_detection'
& 'C:\Users\Sarbagya\Downloads\ppe_detection\.venv\Scripts\python.exe' `
  'C:\Users\Sarbagya\Downloads\ppe_detection\scripts\validate_head_classifier_dataset.py' `
  --data 'C:\Users\Sarbagya\Downloads\ppe_detection\data\head_classifier' `
  --min-per-class 1
```

Add manually reviewed head crops to `data/head_classifier/manifest.csv` and
set `review_status` to `reviewed` only after visual inspection. Use separate
source scenes for `train`, `val`, and `test`; do not copy the same person or
near-duplicate frame into multiple splits. Then prepare the folders:

```powershell
& 'C:\Users\Sarbagya\Downloads\ppe_detection\.venv\Scripts\python.exe' `
  'C:\Users\Sarbagya\Downloads\ppe_detection\scripts\prepare_head_classifier_dataset.py' `
  --manifest 'C:\Users\Sarbagya\Downloads\ppe_detection\data\head_classifier\manifest.csv' `
  --output 'C:\Users\Sarbagya\Downloads\ppe_detection\data\head_classifier_reviewed' `
  --min-train-per-class 20 --min-eval-per-class 10
```

Train only after preparation succeeds:

```powershell
& 'C:\Users\Sarbagya\Downloads\ppe_detection\.venv\Scripts\python.exe' `
  'C:\Users\Sarbagya\Downloads\ppe_detection\scripts\train_head_classifier.py' `
  --data 'C:\Users\Sarbagya\Downloads\ppe_detection\data\head_classifier_reviewed' `
  --output 'C:\Users\Sarbagya\Downloads\ppe_detection\runs\head_classifier_candidate' `
  --epochs 20 --batch 16 --image-size 224 --device cpu
```

Only after held-out evaluation supports promotion should you set
`head_classifier.enabled: true` and point `checkpoint_path` to the candidate
checkpoint. The existing `models/trained/best.pt` remains the fallback.

## Packaging Commands (optional)
Using `pyproject.toml`:
```bash
pip install -e .
```

Then available console scripts:
```bash
ppe-detect
ppe-train
ppe-dashboard
```

## Troubleshooting
- Model not found:
  - Ensure `models/trained/best.pt` exists.
  - Or pass `--model` explicitly.
- Camera not opening:
  - Check webcam index (`--camera 0`, `--camera 1`, etc.).
- Slow inference:
  - Increase `--skip-frames` for video.
  - Use `model.device: "cuda"` if GPU is available.
- OpenCV window not appearing:
  - Run from local desktop session (not headless terminal).

## License
Dataset metadata indicates Roboflow export with `CC BY 4.0`.
Add your project license details here if needed.
