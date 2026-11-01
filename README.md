# Vinfotech CV Engineer — Case Studies

Case Study 1 is assembly verification (`inspect.py`). Case Study 2 is traffic intersection analytics (`traffic.py`).

# Case Study 1

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

# Case Study 2 — Traffic intersection analytics

Counts vehicles through the Coldwater downtown camera (the screencast in the parent folder). Top of the frame is North. Logic is in `traffic.py`.

## Run

From `Vinfotech/`, with the same virtualenv as Case Study 1:

```bash
python traffic.py --video "../Screencast from 2026-10-31 15-35-56.mp4"
```

GPU is used when CUDA is available. Force the device with `--device 0` or `--device cpu`.

Outputs in `output_traffic/`:

- `annotated.mp4` — boxes, track ids, arm zones, running entry counts and queues
- `results.json` and `results.csv` — movements, waits, speeds
- `summary.txt` — the printed table
- `trajectories.jpg` and a few `frame_XXXs.jpg` stills

Other flags: `--model path/to/yolo11x.pt`, `--fps 10`, `--analyze-only` (reuse `detections.pkl` and skip the video).

## Why 10 fps

The recording is 29.7 fps. The program keeps every 3rd frame, about 10 fps. A vehicle in town moves a short distance between those samples, including while it is stopped, which is what ByteTrack needs. Reading all 30 fps would not change who entered or who turned.

## Assumptions

- Top of the frame is North. Arms are the roads toward the top, bottom, right and left.
- The stop line is the boundary between an arm polygon and the intersection polygon, drawn along the crosswalk.
- A vehicle is counted only if the track is seen in an entry arm and later in an exit arm, and the path goes through the intersection (or within 35 px of it). Parked vehicles and anyone who never crosses are not counted.
- Queue: the box moves slower than 25 px/s for at least 0.5 s inside that arm's queue polygon. The north queue polygon stops part-way up the block so curb parking further away is outside it.
- Wait starts when the vehicle stops in that queue and ends when it crosses the stop line. A creep under 2.5 s does not break the stop. Rolling through without a 0.5 s stop has wait 0 and is included in the average. Vehicles still waiting at the end of the clip are not in the average.
- Classes are car, motorcycle, bus and truck (YOLO11x + ByteTrack). People and bicycles are not counted. The ground point is the bottom centre of the box.
- One movement per track id. A track that starts already inside the intersection is not counted. A U-turn stays on the diagonal of the table.
- A box that stays inside a 22 px span for 5 s or more is parked or a static false detection.
- Lost tracks are kept for about 5 s so a signal pole does not split one vehicle.
- Speed is a homography on the south through lane: 3.35 m wide (11 ft), and 10 m from the crosswalk to near the bottom of the frame. Numbers are kept only where one pixel is 1.5–22 cm. Expect about ±30%. East-west speeds are more trustworthy than speeds toward the horizon.
- This clip is 2 min 40 s, shorter than the 10–15 min asked for, so it may cover fewer than three signal cycles.
- The CBPU logo and the park above the east arm are masked out.

## Approach (about 200 words)

YOLO11x finds cars, motorcycles, buses and trucks. ByteTrack keeps an id while a vehicle is stopped, which is what makes queues and turns measurable. The Coldwater view is split by hand into four arm polygons, a queue polygon behind each crosswalk, and an intersection polygon. The top of the frame is North.

A track is the bottom centre of its box. It is counted only when it is seen in one arm, passes through the intersection, and shows up in an exit arm, so parked cars and vehicles that never cross are left out. Queue length is how many tracks sit in the queue polygon moving slower than 25 pixels per second. Wait time runs from the start of that stop until the vehicle leaves the arm. The clip is read at 10 fps, because town traffic barely moves between samples and 30 fps would not change the counts.

Speed is a bonus. A homography on the south through lane uses an 11-foot width and a 10 m length, then converts ground distance over time to km/h. The scale is about ±30% and is not used near the horizon. A signal pole can still split an id.

## Limitations

- Zones are manual. Another intersection needs new polygons.
- The supplied clip is 2 min 40 s, not 10–15 min.
- Occlusion by the lamp post and signal poles can break a track.
- The southeast right-turn lane sits on the boundary of the south and east polygons.
- There is no signal-color detector. Wait comes from motion, not from the light.
- Far up the north arm, speed is not reported. Static windows are sometimes detected as cars; tracks that never move are dropped from the queue, but a long red that lasts the whole clip would look the same as parking.
- YouTube controls occasionally cover the bottom edge of the screencast.

## With more time

- Measure the lane on a map or a site survey and refit the homography.
- Read the signal heads so red time is separate from a voluntary stop.
- Repair broken ids when two tracks hand off at the same spot.
- Use a longer recording that covers at least three full cycles.
