# Blood Detection — Current Status and Blockers

**Document date:** 2026-10-30  
**Project:** PPE Safety Monitor  
**Status:** ❌ Reliable automated blood detection is NOT currently available.

---

## 1. What was investigated

### BloodCCTV-9500 / bloodNet_model
- **Paper:** [Enhanced CCTV Blood Detection with Multiscale Attention and InceptionV3](https://www.nature.com/articles/s41598-025-14941-w) (Nature Scientific Reports, 2025)
- **Code repo:** https://github.com/irshadkhalil23/bloodNet_model

### Findings
| Property | Finding |
|----------|---------|
| Pre-trained weights | **Not released** — no GitHub releases, no downloadable `.h5` or `.pt` file |
| Model type | Image-level classifier (InceptionV3 backbone) — detects whole-image "blood present/absent", not spatial bounding boxes |
| Framework | TensorFlow 2.9.0 — not installed in this project's `.venv` (PyTorch/Ultralytics environment) |
| Dataset access | Google Drive link (requires manual download, no API) |
| Dataset format | Binary image classification (blood / no-blood) — not spatial detection |
| License | None specified in repo |
| Confidence in public deployment | Low — 0 stars, 0 forks, no README verification of published accuracy claims |

---

## 2. Why the previous heuristic was disabled

The previous color-based red-region heuristic was removed from the production pipeline because:

1. It produced **false positive boxes** on hazard tape, red tools, red clothing, and backgrounds.
2. It labeled raw pixel-score results (e.g., `RED_REGION_CANDIDATE 66.0%`) **as if they were blood probabilities**, which is medically misleading.
3. It triggered PDF reports and urgent alerts based on background colors rather than validated injury evidence.

The heuristic is preserved behind `incident_detection.enable_experimental_heuristic: false` in config for debugging only.

---

## 3. What would be needed for a real blood detector

To deploy a genuine blood detection feature, the following are required:

| Requirement | Notes |
|------------|-------|
| Labeled dataset with **spatial annotations** (bounding boxes or segmentation masks) of blood pools | Image-level labels alone cannot train a spatial detector |
| Dataset must include **hard negatives**: red paint, red tools, hazard tape, ketchup, rust, red fabric | Prevents false positives on common red objects |
| Minimum dataset size | At least several thousand images for reliable results |
| YOLO-compatible format (`.txt` label files with class + normalized bbox) or equivalent | To train alongside the existing PPE model |
| Independent evaluation on held-out test set with documented precision/recall | Required before any "BLOOD DETECTED" alert |
| Human expert ground-truth verification | Critical for a safety-critical medical claim |

---

## 4. Current behavior

When `incident_detection.enabled: false` (default):

- The system returns `NOT_EVALUATED_UNAVAILABLE` for blood/injury status.
- PDF reports include an explicit disclosure: *"This scan was not evaluated for blood or injuries."*
- **Absence of a blood alert does NOT mean the scene is blood-free.** The system has not examined the image for blood.
- No urgent notifications are triggered based on color heuristics.

---

## 5. Recommended path forward

1. **Collect a labeled dataset** (internal CCTV footage of simulated spill scenarios, or license a research dataset with appropriate terms).
2. **Annotate with bounding boxes** (not image-level labels) to enable spatial localization.
3. **Include hard negatives** systematically.
4. **Fine-tune a YOLO model** (same architecture as the PPE model) on the blood dataset.
5. **Evaluate independently** — publish precision/recall on a held-out test set before enabling any alerts.
6. **Enable via config** under `incident_detection.blood_model_path` once validated.

---

*This document was generated automatically during investigation of the BloodCCTV-9500 research lead.*

