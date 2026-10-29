# Head Classifier Dataset Collection & Review Guide

This guide establishes the standard workflow for collecting, labelling, and reviewing head-region image crops to train a dedicated two-class head classifier (`helmet_worn` vs. `no_helmet`).

---

## 1. Ground Truth Definitions

### A. `helmet_worn` (Positive Class)
A genuine industrial safety helmet / hard hat is physically worn on the worker's head.
- **Colours**: White, yellow, blue, orange, red, green, black, grey, hi-vis, etc. Safety helmets come in diverse colours. **Never reject or label based on colour alone.**
- **Styles**: Standard hard hats, climbing helmets, bump caps with hard shells, helmets with chin straps, face shields, or earmuffs attached.
- **Angles & Viewpoints**: Frontal, profile/side, 3/4 view, rear/back-of-head, tilted, looking down, looking up.
- **Conditions**: Bright sunlight, shadows, indoor lighting, partial occlusion by machinery or scaffolding.

### B. `no_helmet` (Negative Class)
The worker's head does NOT have a protective safety helmet being worn.
- **Bare Scalp / Bald**: Fully or partially bald heads.
- **Hair**: All natural hair colours and textures (black, dark brown, light brown, blonde, red, grey, white, curly, straight, bald fade).
- **Non-Safety Headwear**: Ordinary baseball caps, beanies, winter hats, turbans, headscarves/hijabs, hoods, headbands, or headphones.
- **Held Helmets**: If a worker holds a helmet in their hand, carries it under their arm, or has it on the ground nearby, **their head crop is labelled `no_helmet`**.

### C. `needs_review` / Excluded
- The head is >70% occluded by obstacles or cut off by the camera boundary.
- Severe motion blur or resolution too low to distinguish hair vs. helmet shell.
- Ambiguous headwear where a hard shell cannot be determined.

---

## 2. Leakage Prevention Across Splits

- **Source Scene Isolation**: Head crops extracted from the **same original image or video frame sequence must be assigned to the SAME split** (`train`, `val`, or `test`).
- **Identity / Person Isolation**: If the same worker appears in multiple frames or angles, keep all crops of that worker in the same split.

---

## 3. Manifest Specification (`manifest.csv`)

Every candidate crop is tracked in `data/head_classifier/manifest.csv`:

| Column | Type | Allowed Values | Description |
|---|---|---|---|
| `crop_path` | string | e.g. `data/head_classifier/crops/xyz.jpg` | Path to extracted head crop image |
| `label` | string | `helmet_worn`, `no_helmet` | Verified class label |
| `split` | string | `train`, `val`, `test` | Target dataset split |
| `review_status` | string | `manual_review_required`, `reviewed`, `rejected` | Review lifecycle state |
| `source_image` | string | e.g. `demo_images/vest_nohelmet.jpg` | Full-scene origin |
| `reviewer` | string | e.g. `sarbagya` | Reviewer identity |
| `notes` | string | free text | Notes (e.g. `bald`, `dark_hair`, `held_helmet`) |

Only rows with `review_status=reviewed` will be ingested by the dataset preparation script.

---

## 4. Execution Workflow

### Step 1: Automatic Extraction
Extract head crops using person detection and head-region localization:
```powershell
python scripts/extract_head_crops.py --inputs "demo_images" "data/Construction Site.v1i.yolov8/train/images"
```

### Step 2: Human Inspection and Labelling
Open `data/head_classifier/manifest.csv` and review crops in `data/head_classifier/crops/`:
1. Check the image crop visually.
2. Fill `label`: `helmet_worn` or `no_helmet`.
3. Assign `split`: `train`, `val`, or `test` (maintain 70/15/15 ratio; ensure scene isolation).
4. Update `review_status` to `reviewed` (or `rejected` if crop is unusable).
5. Add your name under `reviewer` and observations under `notes`.

### Step 3: Materialize Reviewed Dataset
Copy verified crops into ImageFolder structure:
```powershell
python scripts/prepare_head_classifier_dataset.py --manifest data/head_classifier/manifest.csv --output data/head_classifier_reviewed --min-train-per-class 20 --min-eval-per-class 10
```

### Step 4: Validate Integrity and Split Leakage
```powershell
python scripts/validate_head_classifier_dataset.py --data data/head_classifier_reviewed
```

### Step 5: Train Candidate Model
Once validated with sufficient samples:
```powershell
python scripts/train_head_classifier.py --data data/head_classifier_reviewed --epochs 30 --batch 16 --output models/head_classifier/experiment_YYYYMMDD
```

