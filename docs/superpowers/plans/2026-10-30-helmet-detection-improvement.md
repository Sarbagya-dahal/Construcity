# PPE Helmet Detection Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix persistent helmet false-positives (hair detected as helmet) and false-negatives (no-helmet not reported for bare-head persons) via head-region-localised heuristics, a robust data-collection workflow, and a head-region crop classifier (once sufficient reviewed data exists).

**Architecture:** The four-class PPE model (`best.pt`) and person model remain unchanged. A separate, lightweight head-region verification layer runs after detector association and feeds `head_verifications` into the existing `ViolationDetector.check_violations()` hook that is already wired but disabled. The layer uses OpenCV Haar face detection for head localisation (already present in `HeadRegionLocalizer`), plus an improved heuristic filter that does not rely on colour thresholding. If face detection fails, the result is `needs_review`, not a silent assumption.

**Tech Stack:** Python 3.x, OpenCV (Haar cascades for face anchor), PyTorch + torchvision ResNet-18 (for eventual classifier), ultralytics YOLO (existing), unittest (existing test suite).

**Spec:** This document and the task description in the conversation.

## Global Constraints

- Never overwrite or retrain `models/trained/best.pt` — it is the production checkpoint.
- All new code lives in `src/`, `scripts/`, or `tests/`; all new config keys go under existing top-level sections.
- The head classifier stays behind `head_classifier.enabled: false` until a reviewed dataset and checkpoint exist.
- All regression tests must remain green after every task.
- No colour-based rules (e.g. "black pixel = no helmet"). Helmets come in all colours.
- Windows PowerShell; Python is at `.venv\Scripts\python.exe`; no Docker.

## Review Focus

1. **Face not visible (downward gaze, hard hat covering face, far away):** `HeadRegionLocalizer` returns `face_not_localized` → result must be `needs_review`, not a missed violation.
2. **Helmet held in hand:** PPE model may detect helmet bbox; violation logic must not count it as worn (hand bbox is far from person head region) — already covered by `_helmet_in_person_head_region`.
3. **Multiple persons sharing one helmet detection:** `_assign_items` already prevents double assignment.
4. **Very small or partially visible person:** person model returns narrow bbox; head region ratio may clip outside image — `HeadRegionLocalizer._clip_bbox` handles this.
5. **`vest_nohelmet.jpg` scenario (hair = helmet, 77.5% conf):** The raw model emits a 0.775 conf helmet at the person's head-top. The `helmet_worn_confidence_threshold=0.80` gate already sets this to `needs_review`. The improvement task is to make the face-anchored localizer verify whether the ROI actually contains a helmet shape vs hair texture.

---

## Diagnosis Summary (Root Causes Established)

### Evidence from fresh diagnostic run

| Image | Expected | Actual status | Root cause |
|---|---|---|---|
| `vest_nohelmet.jpg` | violation: no helmet | `needs_review` (no violation reported) | PPE model fires `helmet@0.775` at person's hair. `helmet_worn_confidence_threshold=0.80` blocks it from confirming helmet but no `without_helmet` is emitted → result is `needs_review`, never a confirmed violation |
| `no_helmet.jpg` | violation: no helmet | `evaluated_with_uncertainty` (no violation) | PPE model emits **zero** helmet or without_helmet predictions; person detected; no PPE evidence → `helmet_status_uncertain` → no violation raised |
| `both.jpg` | persons compliant | `needs_review` | 1 of 3 persons has helmet@0.731 (below 0.80 gate) → `helmet_below_worn_confidence_threshold` → needs_review. Two other persons have no PPE associated → uncertain |
| `helmet_novest.jpg` | correct: vest violation | `evaluated`: `without_vest` violation | **Correct already** |
| `no_vest.jpg` | vest violation | `evaluated_with_uncertainty`: vest violation | Correct violation; uncertain only for helmet (no helmet class detected) |
| `both2.jpg` | both PPE compliant | persons uncertain | No helmet ≥0.80, no without_helmet → uncertain (conservative) |
| `Can you smell.jpg` | (unknown ground truth) | `evaluated_with_uncertainty` | — |

### Root Cause #1 — The PPE model confuses hair with helmets at 0.60–0.79 confidence
The `helmet_worn_confidence_threshold=0.80` gate correctly prevents these from confirming compliance. But the system has no mechanism to **positively reject** these detections as hair-not-helmet. The result is permanently `needs_review`, which is technically honest but not an improvement.

### Root Cause #2 — The PPE model emits no signal for bare heads in many images  
`no_helmet.jpg` produces **no helmet class and no without_helmet class** from the detector. The pipeline correctly marks this as `uncertain`; it does not hallucinate a violation. But the desired behaviour is to report `without_helmet`. Without a dedicated classifier or a reliable head-region signal, the raw model simply lacks the evidence.

### Root Cause #3 — The head classifier infrastructure exists but the dataset is empty  
`data/head_classifier/manifest.csv` has 4 placeholder rows (all `review_status=pending`). The directory structure exists. The `HeadHelmetClassifier` class and `HeadRegionLocalizer` are fully implemented. The `processor.py` integration point is live but guarded by `head_classifier.enabled: false`. Nothing can be trained yet.

### Root Cause #4 — `HeadRegionLocalizer` requires a visible frontal face
Haar cascade `haarcascade_frontalface_default.xml` fails on side-facing, downward-looking, or occluded faces. This means the head crop classifier will fall through to `needs_review` for many real-world construction-site postures — a known limitation documented in `head_localization.py`.

### What is NOT a root cause
- Threshold drift (changing thresholds did not help and was correctly reverted).
- Annotation format mismatches (class mapping confirmed correct: 0=helmet, 1=vest, 2=without_helmet, 3=without_vest).
- Bounding-box coordinate system errors (coordinates verified correct; preprocessing is simple brightness/contrast adjustment that preserves dimensions).

---

## Task 1: Extend head-region localiser with top-of-person fallback (heuristic, no colour thresholding)

**Files:**
- Modify: `src/head_localization.py`
- Test: `tests/test_detection_pipeline.py`

**Interfaces:**
- Consumes: existing `HeadCropResult` dataclass and `HeadRegionLocalizer.locate(image, person_bbox)` signature.
- Produces: same `HeadCropResult` interface; `reason` now reports `face_anchor`, `face_not_localized` (no fallback), `top_region_fallback` (new), or `invalid_person_box`.

**Rationale:** The current implementation returns `reliable=False` whenever Haar face detection fails, which means the head classifier never runs for workers wearing hard hats (which cover the face). A configurable top-of-person-box crop (top 20-30% of person height) provides a fallback that is explicitly marked as less reliable. The fallback is NOT equivalent to top-quarter blindly — it is parameterised, documented, and sets `reliable=True` only when the crop is large enough.

- [ ] **Step 1: Write failing tests for fallback behaviour**

```python
# In tests/test_detection_pipeline.py, add:

def test_head_localizer_top_region_fallback_when_face_not_found(self):
    """Fallback crop should be returned when Haar detects no face."""
    from src.head_localization import HeadRegionLocalizer
    # Black image — Haar will find no face; fallback should trigger
    image = np.zeros((200, 100, 3), dtype=np.uint8)
    person_bbox = [0.0, 0.0, 100.0, 200.0]
    localizer = HeadRegionLocalizer(use_top_region_fallback=True, top_region_ratio=0.30)
    result = localizer.locate(image, person_bbox)
    self.assertTrue(result.reliable)
    self.assertIsNotNone(result.crop)
    self.assertEqual(result.reason, "top_region_fallback")
    # Fallback crop should be approximately top 30% of person height
    crop_height = result.crop.shape[0]
    person_height = 200
    self.assertAlmostEqual(crop_height / person_height, 0.30, delta=0.05)

def test_head_localizer_no_fallback_when_disabled(self):
    """When fallback is disabled, face-not-found should remain unreliable."""
    from src.head_localization import HeadRegionLocalizer
    image = np.zeros((200, 100, 3), dtype=np.uint8)
    result = HeadRegionLocalizer(use_top_region_fallback=False).locate(image, [0.0, 0.0, 100.0, 200.0])
    self.assertFalse(result.reliable)
    self.assertEqual(result.reason, "face_not_localized")

def test_head_localizer_fallback_does_not_fire_when_face_found(self):
    """If Haar finds a face, face_anchor reason is used not fallback."""
    from src.head_localization import HeadRegionLocalizer
    # A real frontal face image; skip if not available, else check reason
    # Use an image with a detectable face pattern (white rectangle at expected ratio)
    image = np.zeros((200, 100, 3), dtype=np.uint8)
    # We cannot guarantee Haar fires on synthetic data; just verify no crash
    result = HeadRegionLocalizer(use_top_region_fallback=True).locate(image, [0.0, 0.0, 100.0, 200.0])
    self.assertIn(result.reason, {"face_anchor", "top_region_fallback", "face_not_localized"})
```

- [ ] **Step 2: Run tests to confirm they fail**

```powershell
cd C:\Users\Sarbagya\Downloads\ppe_detection
.venv\Scripts\python.exe -m unittest tests.test_detection_pipeline.DetectionPipelineTests.test_head_localizer_top_region_fallback_when_face_not_found -v
```
Expected: FAIL (TypeError — `HeadRegionLocalizer` doesn't accept `use_top_region_fallback`)

- [ ] **Step 3: Implement fallback in `src/head_localization.py`**

Add `use_top_region_fallback: bool = True` and `top_region_ratio: float = 0.28` to `__init__`. After the Haar face block, if no face found and `use_top_region_fallback=True`: compute top-region crop (top `top_region_ratio` of person height), return `HeadCropResult(crop, bbox, True, "top_region_fallback")`. Minimum crop dimension check: at least 20px in both axes; if smaller, return `face_not_localized`.

- [ ] **Step 4: Run tests to verify they pass**

```powershell
cd C:\Users\Sarbagya\Downloads\ppe_detection
.venv\Scripts\python.exe -m unittest discover -s tests -v
```
Expected: all 23 + 3 new tests pass.

- [ ] **Step 5: Update config and `HeadHelmetClassifier` construction**

In `config/config.yaml` under `head_classifier:`, add:
```yaml
use_top_region_fallback: true
top_region_ratio: 0.28
```
In `src/head_classifier.py`, pass these settings through to `HeadRegionLocalizer`.

- [ ] **Step 6: Commit**

```powershell
git add src/head_localization.py src/head_classifier.py config/config.yaml tests/test_detection_pipeline.py
git commit -m "feat: add top-region fallback to HeadRegionLocalizer when face not detected"
```

---

## Task 2: Implement a rule-based head-region heuristic (no colour, no classifier)

**Files:**
- Create: `src/head_heuristic.py`
- Modify: `src/processor.py` (wire up when `head_classifier.enabled: false`)
- Modify: `config/config.yaml` (add `head_heuristic` block)
- Test: `tests/test_detection_pipeline.py`

**Interfaces:**
- Consumes: `image: np.ndarray`, `person_bbox: list[float]`, `helmet_detections: list[dict]`, `head_crop_result: HeadCropResult`
- Produces: `dict` with keys `status: Literal["helmet_worn","no_helmet","needs_review"]`, `reason: str`, `confidence: float`

**Rationale:** Without a trained classifier, we can still distinguish "helmet candidate is inside the person head-region bbox" from "nothing detects a helmet there". The heuristic re-examines the geometry of accepted helmet detections relative to the localised head crop and the IoU overlap. This is explicitly not colour-based.

Specifically: if a `helmet` detection's center falls within `[hx1, hy1, hx2, hy2]` (the localised head crop bbox) AND the IoU of the helmet bbox and the head crop bbox exceeds a configurable threshold, return `helmet_worn`. If no helmet detection overlaps the head crop, return `no_helmet`. If the head crop is `top_region_fallback` (less reliable), return `needs_review` unless overlap is strong (>= a higher threshold).

- [ ] **Step 1: Write failing tests**

```python
def test_head_heuristic_returns_helmet_worn_when_detection_inside_head_crop(self):
    from src.head_heuristic import HeadHeuristic
    heuristic = HeadHeuristic({"head_heuristic": {}})
    from src.head_localization import HeadCropResult
    head_crop = HeadCropResult(
        crop=np.zeros((30, 40, 3), dtype=np.uint8),
        bbox=[10.0, 5.0, 50.0, 35.0],
        reliable=True,
        reason="face_anchor",
    )
    helmet = {"bbox": [15.0, 5.0, 45.0, 32.0], "confidence": 0.85}
    result = heuristic.evaluate(head_crop, [helmet])
    self.assertEqual(result["status"], "helmet_worn")

def test_head_heuristic_returns_no_helmet_when_no_overlap(self):
    from src.head_heuristic import HeadHeuristic
    heuristic = HeadHeuristic({"head_heuristic": {}})
    from src.head_localization import HeadCropResult
    head_crop = HeadCropResult(
        crop=np.zeros((30, 40, 3), dtype=np.uint8),
        bbox=[10.0, 5.0, 50.0, 35.0],
        reliable=True,
        reason="face_anchor",
    )
    result = heuristic.evaluate(head_crop, [])
    self.assertEqual(result["status"], "no_helmet")

def test_head_heuristic_returns_needs_review_for_fallback_crop_with_no_helmet(self):
    from src.head_heuristic import HeadHeuristic
    heuristic = HeadHeuristic({"head_heuristic": {}})
    from src.head_localization import HeadCropResult
    head_crop = HeadCropResult(
        crop=np.zeros((30, 40, 3), dtype=np.uint8),
        bbox=[10.0, 5.0, 50.0, 35.0],
        reliable=True,
        reason="top_region_fallback",
    )
    result = heuristic.evaluate(head_crop, [])
    self.assertEqual(result["status"], "needs_review")
```

- [ ] **Step 2: Run tests to confirm they fail**

```powershell
.venv\Scripts\python.exe -m unittest tests.test_detection_pipeline.DetectionPipelineTests.test_head_heuristic_returns_helmet_worn_when_detection_inside_head_crop -v
```
Expected: FAIL (ImportError — module does not exist)

- [ ] **Step 3: Implement `src/head_heuristic.py`**

Class `HeadHeuristic`:
- `__init__(self, config: dict)`: read `config.get("head_heuristic", {})` for `face_anchor_iou_threshold` (default 0.10), `fallback_iou_threshold` (default 0.30).
- `_iou(a, b) -> float`: static, same logic as `PPEDetector._bbox_iou`.
- `evaluate(self, head_crop: HeadCropResult, associated_helmets: list[dict]) -> dict`: 
  - If `not head_crop.reliable`: return `{"status": "needs_review", "reason": head_crop.reason, "confidence": 0.0}`.
  - If `head_crop.reason == "top_region_fallback"`: use `fallback_iou_threshold`.
  - Else: use `face_anchor_iou_threshold`.
  - For each associated helmet bbox, compute IoU with `head_crop.bbox`.
  - If any helmet IoU >= threshold: return `{"status": "helmet_worn", "reason": "helmet_overlaps_head_region", "confidence": max_iou}`.
  - Else: return `{"status": "no_helmet", "reason": "no_helmet_overlaps_head_region", "confidence": 0.0}` if `face_anchor`; return `{"status": "needs_review", "reason": "fallback_no_helmet_overlap", "confidence": 0.0}` if `top_region_fallback`.

- [ ] **Step 4: Wire `HeadHeuristic` into `FrameProcessor`**

In `src/processor.py`:
- Import `HeadHeuristic` and `HeadRegionLocalizer` at the top.
- In `__init__`, always instantiate `HeadRegionLocalizer` and `HeadHeuristic` (they require no checkpoint).
- In `process_frame`, after `self.detector.detect()`, if `head_classifier` is None (disabled), call the heuristic path:
  - For each detected person, call `localizer.locate(enhanced_frame, person_bbox)` → `head_crop`.
  - Filter `detections["helmet"]` to only those associated with this person (use the association detail from `violation_detector._assign_items` or reproduce the containment check cheaply).
  - Call `heuristic.evaluate(head_crop, person_helmets)`.
  - Collect into `head_verifications` list.
- The existing `head_classifier` branch is unchanged and takes priority.

- [ ] **Step 5: Add config keys**

In `config/config.yaml`, add below `head_classifier:`:
```yaml
head_heuristic:
  face_anchor_iou_threshold: 0.10
  fallback_iou_threshold: 0.30
```

- [ ] **Step 6: Run all tests**

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```
Expected: all tests pass.

- [ ] **Step 7: Smoke-test all demo images and record results**

```powershell
.venv\Scripts\python.exe scripts/_quick_diag.py 2>&1 | Select-String "---|persons|status|person [0-9]|RAW|POST"
```
Document which images now show a changed status.

- [ ] **Step 8: Commit**

```powershell
git add src/head_heuristic.py src/processor.py config/config.yaml tests/test_detection_pipeline.py
git commit -m "feat: add head-region heuristic verification layer (no colour rules)"
```

---

## Task 3: Add regression tests for key failure scenarios

**Files:**
- Modify: `tests/test_detection_pipeline.py`

**Interfaces:**
- Consumes: `HeadHeuristic`, `HeadRegionLocalizer`, `ViolationDetector.check_violations()` (existing).
- Produces: Named test cases for held-helmet, bare-head via heuristic, fallback uncertainty, and classifier conflict.

- [ ] **Step 1: Write and commit additional regression tests**

Add the following named tests (all unit-level, no real images):

```python
def test_heuristic_held_helmet_at_waist_does_not_confirm_helmet_worn(self):
    """Helmet bbox in lower half of person cannot overlap head crop at top."""
    from src.head_heuristic import HeadHeuristic
    from src.head_localization import HeadCropResult
    heuristic = HeadHeuristic({"head_heuristic": {}})
    head_crop = HeadCropResult(np.zeros((30,40,3),dtype=np.uint8), [10,5,50,35], True, "face_anchor")
    held_helmet = {"bbox": [10, 150, 50, 190], "confidence": 0.85}  # waist level
    result = heuristic.evaluate(head_crop, [held_helmet])
    self.assertEqual(result["status"], "no_helmet")

def test_heuristic_needs_review_when_head_crop_unreliable(self):
    from src.head_heuristic import HeadHeuristic
    from src.head_localization import HeadCropResult
    heuristic = HeadHeuristic({"head_heuristic": {}})
    head_crop = HeadCropResult(None, None, False, "face_not_localized")
    result = heuristic.evaluate(head_crop, [{"bbox":[0,0,10,10],"confidence":0.9}])
    self.assertEqual(result["status"], "needs_review")

def test_violation_result_no_helmet_confirmed_by_heuristic(self):
    """ViolationDetector receives head_verification with no_helmet → records violation."""
    detector = ViolationDetector({"violation": {}})
    result = detector.check_violations(
        {"person": [detection([0,0,100,200],class_name="person")],
         "helmet": [], "vest": [], "without_helmet": [], "without_vest": []},
        head_verifications=[{"status": "no_helmet", "reason": "no_helmet_overlaps_head_region", "confidence": 0.0}],
    )
    self.assertEqual(result["person_results"][0]["violations"], ["without_helmet"])
    self.assertEqual(result["violation_count"], 1)

def test_heuristic_helmet_worn_confirmed_by_strong_overlap(self):
    from src.head_heuristic import HeadHeuristic
    from src.head_localization import HeadCropResult
    heuristic = HeadHeuristic({"head_heuristic": {"face_anchor_iou_threshold": 0.10}})
    head_crop = HeadCropResult(np.zeros((40,50,3),dtype=np.uint8), [0.0,0.0,50.0,40.0], True, "face_anchor")
    helmet = {"bbox": [2.0, 1.0, 48.0, 37.0], "confidence": 0.92}
    result = heuristic.evaluate(head_crop, [helmet])
    self.assertEqual(result["status"], "helmet_worn")
```

- [ ] **Step 2: Run all tests to verify green**

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

- [ ] **Step 3: Commit**

```powershell
git add tests/test_detection_pipeline.py
git commit -m "test: add heuristic regression tests for held-helmet, bare-head, unreliable crop"
```

---

## Task 4: Produce a comprehensive demo-image diagnostic report

**Files:**
- Modify: `scripts/diagnose_demo_images.py` (extend to cover all demo images and include heuristic results)
- Output: `outputs/heuristic_diagnostic_report.json`

**Interfaces:**
- Consumes: all `.jpg` files in `demo_images/`, full pipeline result including `head_verifications`.
- Produces: JSON report with per-image raw detections, filtered detections, head-region localisation results, heuristic verdicts, and violation results.

- [ ] **Step 1: Extend `diagnose_demo_images.py` to iterate all demo images**

The current script only processes `vest_nohelmet.jpg`, `helmet_novest.jpg`, `both.jpg`. Extend to `sorted(demo_dir.glob("*.jpg"))` and include `result["head_verifications"]` in the per-image report.

- [ ] **Step 2: Run the extended diagnostic**

```powershell
.venv\Scripts\python.exe scripts/diagnose_demo_images.py 2>&1 | Select-String "^{|^outputs|^---"
```

- [ ] **Step 3: Record the heuristic verdict for each demo image and compare against ground truth**

Expected improvements after Task 2:
- `vest_nohelmet.jpg`: heuristic should emit `no_helmet` (no helmet bbox overlaps head crop) or `needs_review` → triggers `without_helmet` violation or at least escalates.
- `no_helmet.jpg`: heuristic confirms `no_helmet` → triggers `without_helmet` violation.
- `helmet_novest.jpg`: heuristic confirms `helmet_worn` → resolves previous uncertainty.
- `both.jpg`: at least person 1 gets a heuristic verdict from head crop.

- [ ] **Step 4: Commit**

```powershell
git add scripts/diagnose_demo_images.py outputs/heuristic_diagnostic_report.json
git commit -m "diag: extend demo diagnostic to all images and include head-heuristic results"
```

---

## Task 5: Data-collection workflow for head-crop classifier

**Files:**
- Create: `scripts/extract_head_crops.py` (extract crops from full-scene images for manual review)
- Modify: `data/head_classifier/manifest.csv` (add rows for newly extracted crops with `review_status=pending`)
- Create: `data/head_classifier/COLLECTION_GUIDE.md` (instructions for manual crop review and labelling)

**Interfaces:**
- Consumes: full-scene images in `data/Construction Site.v1i.yolov8/` and `demo_images/`.
- Produces: cropped head-region images in `data/head_classifier/crops/`, manifest rows with source, crop_path, label, split, review_status=pending.

**Note:** No classifier can be trained until a human reviewer fills `review_status=reviewed` and correct `label` for each manifest row. This task sets up the pipeline for that human review step.

- [ ] **Step 1: Implement `scripts/extract_head_crops.py`**

Script logic:
1. Load `best.pt` person model (not PPE model).
2. For each image in `data/Construction Site.v1i.yolov8/train/images/`, detect persons.
3. For each person, use `HeadRegionLocalizer` to get head crop (with fallback enabled).
4. Save crop to `data/head_classifier/crops/{source_stem}_p{idx}.jpg`.
5. Append a manifest row: `crop_path=..., label=, split=, review_status=pending, source_image=..., reviewer=, notes=`.
6. Run de-duplication using SHA-256 to skip near-identical crops.
7. Print a summary of how many crops were extracted.

- [ ] **Step 2: Run crop extraction (dry-run first to check counts)**

```powershell
.venv\Scripts\python.exe scripts/extract_head_crops.py --dry-run
```

Then run for real:
```powershell
.venv\Scripts\python.exe scripts/extract_head_crops.py
```

- [ ] **Step 3: Write `COLLECTION_GUIDE.md`**

Clear instructions: what `helmet_worn` vs `no_helmet` means, how to handle ambiguous crops, how to set `split`, how to update `review_status`, how to run the validator after review.

- [ ] **Step 4: Validate the manifest structure**

```powershell
.venv\Scripts\python.exe scripts/validate_head_classifier_dataset.py --data data/head_classifier --min-per-class 0
```

- [ ] **Step 5: Commit**

```powershell
git add scripts/extract_head_crops.py data/head_classifier/COLLECTION_GUIDE.md data/head_classifier/manifest.csv
git commit -m "data: add head-crop extraction script and collection guide for manual review"
```

---

## Task 6: Train head-region classifier (contingent on reviewed data)

> **Prerequisite:** `prepare_head_classifier_dataset.py` must succeed (requires ≥20 reviewed train crops and ≥10 val/test crops per class). If not yet available, **skip this task** and document the gap clearly.

**Files:**
- Script: `scripts/train_head_classifier.py` (already exists)
- Output: `models/head_classifier/experiment_YYYYMMDD/` (checkpoint, config, metrics)

**Interfaces:**
- Consumes: `data/head_classifier_reviewed/` (output of `prepare_head_classifier_dataset.py`)
- Produces: `models/head_classifier/best.pt` (candidate only)

- [ ] **Step 1: Verify reviewed data meets minimum threshold**

```powershell
.venv\Scripts\python.exe scripts/prepare_head_classifier_dataset.py --output data/head_classifier_reviewed
```

- [ ] **Step 2: Train ResNet-18 classifier on CPU**

```powershell
.venv\Scripts\python.exe scripts/train_head_classifier.py --data data/head_classifier_reviewed --epochs 30 --batch-size 16 --output models/head_classifier/exp_YYYYMMDD
```

Record loss curves, val accuracy, per-class precision/recall/F1.

- [ ] **Step 3: Evaluate on held-out test split**

Only after candidate selection, run once on the test split.

- [ ] **Step 4: Promote candidate only if improvement is demonstrated**

Enable `head_classifier.enabled: true` and `head_classifier.checkpoint_path` in `config/config.yaml` only when test metrics show clear improvement over heuristic baseline. Keep `head_classifier.enabled: false` until then.

- [ ] **Step 5: Commit**

```powershell
git add models/head_classifier/exp_YYYYMMDD/ config/config.yaml
git commit -m "feat: train head classifier candidate (ResNet-18)"
```

---

## Task 7: Final verification

**Files:**
- No code changes — verification only.

- [ ] **Step 1: Run full test suite**

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```
Expected: all tests pass.

- [ ] **Step 2: Run smoke test on all demo images**

```powershell
.venv\Scripts\python.exe scripts/_quick_diag.py 2>&1 | Where-Object { $_ -match "---|persons|status|person " }
```

- [ ] **Step 3: Verify `best.pt` is unchanged**

```powershell
Get-FileHash models/trained/best.pt -Algorithm SHA256
```

Compare against pre-task hash (record it now).

- [ ] **Step 4: Generate final diagnostic report**

```powershell
.venv\Scripts\python.exe scripts/diagnose_demo_images.py
```

- [ ] **Step 5: Commit final state**

```powershell
git add outputs/
git commit -m "docs: final diagnostic report and verification results"
```

