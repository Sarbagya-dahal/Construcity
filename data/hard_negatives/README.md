# PPE hard-negative fixtures

These images are kept outside the `train`, `valid`, and `test` splits. They
are for diagnosis and future annotation review only; they must not be used as
evaluation evidence without an independent labelled split.

`vest_nohelmet.jpg` is manually annotated using the dataset's polygon-label
format:

- class `0` (`helmet`): the white helmet held in the worker's hand.
- class `2` (`without_helmet`): the worker's uncovered head.
- class `1` (`vest`): the visible reflective vest.

The rectangles are conservative manual polygons around the visible objects.
All three visible PPE-relevant objects are annotated so the fixture does not
turn the vest into an unlabelled background region during training.

## Review workflow

1. Copy a reviewed image into `data/hard_negatives/images/`.
2. Add a same-stem label file under `data/hard_negatives/labels/`.
3. Use the existing polygon-label format: `class_id x1 y1 x2 y2 ...`, with
   normalized coordinates in `[0, 1]`.
4. Label every visible relevant object, including positive PPE objects, so
   unlabelled objects are not treated as background during training.
5. Record the scene category in the review notes: bare head, bald scalp, cap,
   beanie, headphones, held helmet, nearby helmet, or worn helmet.
6. Run `python scripts/validate_hard_negatives.py` before adding a fixture to
   an experiment dataset. Never copy a fixture into the official validation
   or test split.
