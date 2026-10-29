# Head-region helmet classifier dataset

This dataset is intentionally separate from the four-class PPE detector. It
uses ImageFolder-compatible folders:

```text
data/head_classifier/
  train/helmet_worn/
  train/no_helmet/
  val/helmet_worn/
  val/no_helmet/
  test/helmet_worn/
  test/no_helmet/
```

Each file must be a manually reviewed head crop, not a full scene. A crop may
include the person's face, hair/scalp, helmet brim/top, and a small amount of
context needed to decide whether the helmet is worn. A helmet held in a hand
must not be included as a positive worn-helmet crop. Occluded, blurry, or
ambiguous crops should be excluded from the training classes and retained in a
separate review set.

`manifest.csv` is the intake queue. A row may enter the dataset only when
`crop_path`, `label`, `split`, and `review_status=reviewed` are filled by a
human reviewer. The same source scene must never appear in multiple splits.
Do not copy labels from the existing PPE detector predictions.

Recommended reviewed coverage before training:

- At least 20 training crops per class and at least 10 validation/test crops
  per class.
- Bare heads with dark, light, grey, and no hair.
- Ordinary caps, beanies, headphones, and uncertain/occluded heads.
- Worn helmets in white, black, yellow, orange, and other legitimate colours.
- Worn helmets at different viewpoints, distances, lighting, and image edges.
- Helmets held in hands or resting nearby, labelled as `no_helmet` for the
  person's head crop when the person is not wearing one.

Use a visual tool such as CVAT, Label Studio, or an image editor to create and
review crops. Keep the original full image and record it in `source_image`.
Run the validator and preparation script before training.
