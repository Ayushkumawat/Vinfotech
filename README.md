# Vinfotech CV Engineer — Case Study 1

Assembly verification for the gear inspection station using the YOLO **instance segmentation** model (`defect final.pt`). All logic lives in **`inspect.py`** (single entry point).

## Setup

```bash
cd candidate_pack
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Single image:

```bash
python inspect.py --image test_images/test_01.png
```

All test images (summary table + annotated outputs in `output/`):

```bash
python inspect.py --folder test_images
```

Options:

- `--model "defect final.pt"` — segmentation weights (default)
- `--reference-dir reference_good` — builds slot templates from OK references
- `--output-dir output` — annotated PNGs with green (OK) / red (fail) masks
- `--device cpu` — default; no GPU required

## Approach (short)

1. Run YOLO-seg on each image to get instance masks and class labels.
2. Align to a canonical frame using the **base plate** mask (centroid, PCA orientation, scale).
3. Learn expected part locations from the three images in `reference_good/` (k-means slots for multi-instance parts).
4. Merge duplicate detections on the same part, then match to reference slots (Hungarian assignment). Report **missing** when required slots are not filled (too few detections or wrong location). Spurious duplicate boxes are merged; leftover detections do not add extra failure labels.
5. Save an overlay image with colored masks and missing-slot markers.

## Assumptions

- The base plate is always visible; alignment uses its segmentation mask.
- Part locations are stable enough to compare in base-normalized coordinates (±5° rotation / small shift).
- Wrong class on a slot is treated as misplaced when counts or slot assignment fail.
- Detection confidence threshold is 0.25 unless changed with `--conf`.

## Limitations

- Similar-looking parts (e.g. extra circlip vs compound circlip) rely on the model’s class labels.
- Severe occlusion or very poor segmentation may cause false missing/misplaced flags.
- Templates are built from three reference photos only; more references would tighten tolerances.
