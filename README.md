# Vinfotech CV Engineer — Case Studies

Case Study 1 is assembly verification (`inspection.py`). Case Study 2 is traffic intersection analytics (`draw_fov.py`, then `traffic.py`).

# Case Study 1

Assembly verification for the gear inspection using the YOLO **instance segmentation** model (`defect final.pt`). All logic lives in `inspection.py`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Single image:

```bash
python inspection.py --image test_images/test_01.png
```

All test images (summary table + annotated outputs in `output/`):

```bash
python inspection.py --folder test_images
```



## Approach

1. Run YOLO-seg (`defect final.pt`) and keep each part's box centre.
2. The three photos in `reference_good/` are the layouts. The base-plate centre is the origin. The assembly is lined up with a shift and a rotation of up to ±5°, and the layout whose typical error is smallest is used.
3. A part is present when its centre is within 80 px of a slot of the same name. That mask is green. A slot with no detection is missing: a red circle is drawn there and the name is listed. A detection farther than 80 px from every slot of that name is misplaced only when a slot of that name is still empty. If every slot is already filled, the extra box is ignored.
4. The image prints **OK** only when nothing is missing or misplaced. Otherwise it prints **NOT OK** and the names. A folder run prints one summary table after the images.



## Assumptions

- The base plate is visible, and its box centre is the origin. If it is missing, other parts are counted but not checked for place.
- The whole assembly may shift and rotate by up to ±5°. A gear spinning in place does not move its centre.
- In place means within 80 px of the slot. The three good photos agree within about 25 px, and the nearest two screws are about 210 px apart.
-The three photos in `reference_good/` are the layouts.



## Limitations

- Without the base plate, missing parts are listed but not marked on the image, and place is not checked.
- Occlusion or a bad mask can mark a part missing or misplaced. An extra box is ignored once every slot of that part is filled, so a wrong class on top of another part is not called misplaced.
- Only three reference photos. A turn larger than ±5° will not line up.



## With more time

- Train the model on more images and different scenarios to make the model more confident in detecting defects.



# Case Study 2 — Traffic intersection analytics

Counts vehicles through a fixed camera of a signalised 4-way intersection. Top of the frame is North. Two scripts - For FoV setup: `draw_fov.py`, Run inference on video: `traffic.py`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Download the raw video clip from the google drive - [https://drive.google.com/drive/folders/1hYbV1kiuQggNsCxmqqtlGqHb5TgY9SFP?usp=sharing](https://drive.google.com/drive/folders/1hYbV1kiuQggNsCxmqqtlGqHb5TgY9SFP?usp=sharing)

## Run

1. Draw the four road zones on one frame. Press `n` `s` `e` `w` to pick the arm, click the corners, `p` to print, `q` to quit and print.

```bash
python draw_fov.py --video Traffic_Video.mp4
```

Draw each arm along the driving lanes up to the stop line. Leave the middle of the crossing empty, and keep curb parking outside the polygon.

1. Paste the printed `ZONES` block over the one at the top of `traffic.py`.
2. Run the clip:

```bash
python traffic.py --video Traffic_Video.mp4
```

The live window uses `yolo26n.pt` at 720 px and reads every frame. Drawing and the saved video are the 960 px preview, not the full 1080p frame, so the laptop is not encoding a large picture on top of the model. A fast car moves too far if frames are skipped, so no frames are dropped.

Outputs in `output_traffic/`:

- `annotated.mp4` — zones, boxes, track id, km/h, how many vehicles are inside each arm, the queue and the turn counts
- `results.csv` — totals per arm, the 4×4 entry and exit table, then max queue, max wait and average wait

Press `q` in the video window to stop early. The summary still saves.

## Frame rate

The clip is about 30 fps. Skipping frames drops high-speed cars, because they move too far for the tracker to keep the same id. `traffic.py` runs `yolo26n` at 720 px on every frame, so a fast car is not lost between frames. The saved video is the smaller preview, not the full 1080p frame.

## Assumptions

- U-turns are not counted. The diagonal of the table stays zero.
- Classes are car, motorcycle, bus and truck. People and bicycles are ignored.
- Queue means: inside that arm, not parked, slower than 5 km/h.
- Parked is separate from the turn. Speed under 5 km/h for 60 seconds is parked. Speed under 5 km/h for less than that is waiting.
- Wait is recorded only for a vehicle that dropped under 5 km/h and later reached a different arm. A vehicle that drives through without stopping is not included, so its wait is not stored as 0. A vehicle that stays under 5 km/h for 60 seconds is parked and is not included.
- Speed uses the box width as the real vehicle width (car 1.8 m, motorcycle 0.8 m, bus 2.6 m, truck 2.4 m) and the movement of the box centre over about half a second. Expect about ±40%. Trucks and side-on cars are the weak cases, because the box width is not always the vehicle width.



## Approach

`draw_fov.py` shows one frame. You mark the north, south, east and west roads and paste the printed points into `traffic.py`. That second script runs YOLOv26n with BoT-SORT for tracking and draws the zones and a box on every vehicle. The centre of the box is tested against the four polygons. The first arm it is found in is where it entered, the next different arm is where it exited, and that pair fills the 4×4 table. While a vehicle is inside its arm, under 5 km/h, and not parked, it counts toward that queue. Wait is kept only if it did slow under 5 km/h and then reached another arm. A drive-through is not stored as wait 0. If it stays under 5 km/h for 60 seconds it is parked and leaves the queue. The km/h on the box is box-width converted to metres, then distance of the box centre over time. It is a rough figure, not a surveyed scale. A pole in front of a vehicle can still split one id into two, and a red longer than 60 seconds is called parked.

## Limitations

- Zones are hand drawn. A new camera needs new points.
- Speed is a rough conversion from box width, about ±40%.
- A stop longer than 60 seconds is treated as parked, so a very long red can leave the queue.
- A signal pole or the lamp post can break a track id.



## With more time

- Measure one lane on a map and replace the box-width scale.
- Read the signal heads so a red is not confused with parking.

