import math
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ZONES = {
    "N": [(396, 214), (1090, 578), (492, 636), (228, 216)],
    "S": [(680, 892), (1578, 783), (1911, 976), (1918, 1076), (817, 1075)],
    "E": [(1291, 538), (1651, 721), (1911, 685), (1918, 507)],
    "W": [(1, 682), (493, 613), (652, 890), (1, 922)]}
ARMS = ["N", "S", "E", "W"]
VEHICLE_IDS = [2, 3, 5, 7]
WIDTH_M = {2: 1.8, 3: 0.8, 5: 2.6, 7: 2.4}
COLORS = {"N": (255, 80, 0), "S": (0, 180, 0), "E": (0, 140, 255), "W": (200, 0, 200)}
PARK_SEC = 60
POLYS = {arm: np.array(pts, np.int32) for arm, pts in ZONES.items()}

video = sys.argv[-1]
root = Path(__file__).resolve().parent
model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
cap.release()

tracks = {}
moves = {a: {b: 0 for b in ARMS} for a in ARMS}
waits = {a: [] for a in ARMS}
max_queue = {a: 0 for a in ARMS}
writer = None
scale = 1.0
draw_polys = {}
out_path = root / "output_traffic" / "annotated.mp4"
out_path.parent.mkdir(exist_ok=True)
frame_i = 0

for result in model.track(source=video, stream=True, persist=True, classes=VEHICLE_IDS, conf=0.4, tracker=str(root / "botsort_traffic.yaml"), imgsz=720, verbose=False):
    frame_i += 1
    t = frame_i / fps
    src = result.orig_img
    if writer is None:
        h, w = src.shape[:2]
        scale = min(1.0, 960 / w, 540 / h)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(w * scale), int(h * scale)))
        draw_polys = {arm: np.round(poly * scale).astype(np.int32) for arm, poly in POLYS.items()}
    frame = cv2.resize(src, (int(src.shape[1] * scale), int(src.shape[0] * scale)))
    over = frame.copy()
    for arm, poly in draw_polys.items():
        cv2.fillPoly(over, [poly], COLORS[arm])
    frame = cv2.addWeighted(over, 0.22, frame, 0.78, 0)
    for arm, poly in draw_polys.items():
        cv2.polylines(frame, [poly], True, COLORS[arm], 2)
        cv2.putText(frame, arm, (int(poly[:, 0].mean()), int(poly[:, 1].mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[arm], 2)

    queue = {a: 0 for a in ARMS}
    inside = {a: 0 for a in ARMS}
    boxes = result.boxes
    if boxes is not None and boxes.id is not None:
        for (x1, y1, x2, y2), tid, cls in zip(boxes.xyxy.cpu().numpy(), boxes.id.int().tolist(), boxes.cls.int().tolist()):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw = max(x2 - x1, 1)
            zone = None
            for arm, poly in POLYS.items():
                if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                    zone = arm
                    break

            tr = tracks.get(tid)
            if tr is None:
                tr = {"pts": [], "frm": None, "counted": False, "slow_since": None, "waited": None, "parked": False, "speed": 0.0}
                tracks[tid] = tr
            tr["pts"].append((t, cx, cy))
            if len(tr["pts"]) > 10:
                tr["pts"].pop(0)
            t0, x0, y0 = tr["pts"][0]
            if t - t0 > 0.2:
                dist = math.hypot(cx - x0, cy - y0)
                tr["speed"] = 0.0 if dist < 4 else min(130.0, dist * (WIDTH_M[cls] / bw) / (t - t0) * 3.6)

            known = t - tr["pts"][0][0] > 0.2
            if known and tr["speed"] < 5:
                if tr["slow_since"] is None:
                    tr["slow_since"] = t
                if t - tr["slow_since"] >= PARK_SEC:
                    tr["parked"] = True
            elif known and tr["slow_since"] is not None and not tr["parked"]:
                tr["waited"] = t - tr["slow_since"]
                tr["slow_since"] = None

            if zone is not None:
                if tr["frm"] is None:
                    tr["frm"] = zone
                elif zone != tr["frm"] and not tr["counted"]:
                    moves[tr["frm"]][zone] += 1
                    tr["counted"] = True
                    if not tr["parked"] and tr["waited"] is not None:
                        waits[tr["frm"]].append(tr["waited"])
                    elif not tr["parked"] and tr["slow_since"] is not None:
                        waits[tr["frm"]].append(t - tr["slow_since"])
            if zone:
                inside[zone] += 1
                if known and tr["speed"] < 5 and not tr["parked"]:
                    queue[zone] += 1

            color = (0, 255, 255) if tr["speed"] < 5 else (0, 220, 0)
            cv2.rectangle(frame, (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale)), color, 2)
            cv2.putText(frame, f"{tid} {tr['frm'] or '-'} {tr['speed']:.0f} km/h", (int(x1 * scale), int(max(16, y1 * scale - 6))), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    for arm in ARMS:
        max_queue[arm] = max(max_queue[arm], queue[arm])
    y = 22
    cv2.putText(frame, f"in N{inside['N']} S{inside['S']} E{inside['E']} W{inside['W']}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 18
    cv2.putText(frame, f"wait N{queue['N']} S{queue['S']} E{queue['E']} W{queue['W']}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    for arm in ARMS:
        y += 18
        m = moves[arm]
        cv2.putText(frame, f"{arm} -> N{m['N']} S{m['S']} E{m['E']} W{m['W']}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imshow("traffic", frame)
    writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

if writer is not None:
    writer.release()
cv2.destroyAllWindows()

lines = ["arm,total"]
for a in ARMS:
    lines.append(f"{a},{sum(moves[a].values())}")
lines += ["", "entry," + ",".join(ARMS)]
for a in ARMS:
    lines.append(a + "," + ",".join(str(moves[a][b]) for b in ARMS))
lines += ["", "arm,max_queue,max_wait_s,avg_wait_s"]
for a in ARMS:
    ws = waits[a]
    lines.append(f"{a},{max_queue[a]},{max(ws) if ws else 0:.1f},{(sum(ws) / len(ws) if ws else 0):.1f}")
text = "\n".join(lines)
print(text)
(out_path.parent / "results.csv").write_text(text + "\n", encoding="utf-8")