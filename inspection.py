import argparse
import math
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Required parts and their quantities
REQUIRED_PARTS = {
    "Base plate": 1,
    "Corner mounting bolts with washers": 4,
    "Large brass output gear": 1,
    "Washer and nut on the output gear": 1,
    "Middle idler gear": 1,
    "Circlip on the idler gear": 1,
    "Compound gear": 1,
    "Circlip on the compound gear": 1,
    "Bearing blocks": 2,
    "Screws on the bearing blocks": 4,
    "Motor on its mounting plate": 1,
    "Screws on the motor mounting plate": 2,
    "Pinion gear on the motor": 1,
    "Cable connector plugged into the motor socket": 1}

MODEL_PATH = "defect final.pt"

GREEN = (0, 255, 0)
RED = (0, 0, 255)
# Good photos agree within about 25 px. The nearest two screws are about 210 px apart.
PLACE_PX = 80

# Load YOLO model
model = YOLO(MODEL_PATH, task="segment")

# Each good photo: part name -> list of (x, y) centers. These are the possible layouts.
REF_LAYOUTS = []
for ref_path in sorted(Path("reference_good").glob("*.png")):
    ref = model.predict(str(ref_path), conf=0.13, verbose=False)[0]
    layout = {}
    if ref.boxes is not None:
        for i in range(len(ref.boxes)):
            name = model.names[int(ref.boxes.cls[i])]
            x1, y1, x2, y2 = ref.boxes.xyxy[i]
            layout.setdefault(name, []).append(((float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2))
    REF_LAYOUTS.append(layout)


# Move a reference layout onto this image: base-plate center, then a rotation of deg.
def place_layout(layout, bx, by, deg):
    rbx, rby = layout["Base plate"][0]
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    moved = {}
    for name, pts in layout.items():
        moved[name] = []
        for x, y in pts:
            dx, dy = x - rbx, y - rby
            moved[name].append((bx + c * dx - s * dy, by + s * dx + c * dy))
    return moved


# Pick the reference photo and rotation (within ±5°) that best matches the parts we found.
def best_layout(centers):
    bx, by = centers["Base plate"][0]
    best = None
    for layout in REF_LAYOUTS:
        if "Base plate" not in layout:
            continue
        for deg in range(-5, 6):
            moved = place_layout(layout, bx, by, deg)
            dists = []
            for name, pts in centers.items():
                if name == "Base plate" or name not in moved:
                    continue
                for fx, fy in pts:
                    dists.append(min(math.hypot(fx - x, fy - y) for x, y in moved[name]))
            if not dists:
                continue
            dists.sort()
            err = dists[len(dists) // 2]
            if best is None or err < best[0]:
                best = (err, moved)
    if best is not None:
        return best[1]
    for layout in REF_LAYOUTS:
        if "Base plate" in layout:
            return place_layout(layout, bx, by, 0)
    return None


# Inspect one image. Returns the row used by the folder summary.
def inspect_image(image_path):
    image = cv2.imread(str(image_path))
    result = model.predict(image, conf=0.13, verbose=False)[0]
    dets = []

    if result.masks is not None:
        for i in range(len(result.masks.data)):
            name = model.names[int(result.boxes.cls[i])]
            if name not in REQUIRED_PARTS:
                continue
            x1, y1, x2, y2 = result.boxes.xyxy[i]
            mask = result.masks.data[i].cpu().numpy()
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            dets.append({
                "name": name,
                "cx": (float(x1) + float(x2)) / 2,
                "cy": (float(y1) + float(y2)) / 2,
                "x1": int(x1),
                "y1": int(y1),
                "mask": mask > 0.5})

    centers = {}
    for d in dets:
        centers.setdefault(d["name"], []).append((d["cx"], d["cy"]))

    moved = best_layout(centers) if "Base plate" in centers else None
    status = ["ok"] * len(dets)
    missing = []

    if moved is None:
        # No base plate, so there is no origin to judge place. Count only.
        for name, need in REQUIRED_PARTS.items():
            for _ in range(need - len(centers.get(name, []))):
                missing.append((name, None, None))
    else:
        for name, need in REQUIRED_PARTS.items():
            slots = moved.get(name, [])
            idxs = [i for i, d in enumerate(dets) if d["name"] == name]
            pairs = []
            for i in idxs:
                for s, (x, y) in enumerate(slots):
                    pairs.append((math.hypot(dets[i]["cx"] - x, dets[i]["cy"] - y), i, s))
            pairs.sort()
            used_i, used_s = set(), set()
            for dist, i, s in pairs:
                if dist > PLACE_PX or i in used_i or s in used_s:
                    continue
                used_i.add(i)
                used_s.add(s)
            for s, (x, y) in enumerate(slots):
                if s not in used_s:
                    missing.append((name, x, y))
            for _ in range(max(0, need - len(slots))):
                missing.append((name, None, None))
            # A spare box is not a misplaced part once every slot of that name is filled.
            for i in idxs:
                if i in used_i:
                    continue
                near = any(math.hypot(dets[i]["cx"] - x, dets[i]["cy"] - y) <= PLACE_PX for x, y in slots)
                if not near and len(used_s) < len(slots):
                    status[i] = "misplaced"
                else:
                    status[i] = "ignore"

    misplaced = [dets[i]["name"] for i, st in enumerate(status) if st == "misplaced"]
    verdict = "OK" if not missing and not misplaced else "NOT OK"

    output = image.copy()
    labels = []
    order = sorted(range(len(dets)), key=lambda i: dets[i]["name"] == "Base plate", reverse=True)
    for i in order:
        if status[i] == "ignore":
            continue
        d = dets[i]
        color = RED if status[i] == "misplaced" else GREEN
        alpha = 0.15 if d["name"] == "Base plate" else 0.45
        tint = np.array(color, dtype=np.float32)
        output[d["mask"]] = (output[d["mask"]].astype(np.float32) * (1 - alpha) + tint * alpha).astype(np.uint8)
        text = ("misplaced: " if status[i] == "misplaced" else "") + d["name"]
        labels.append((text, d["x1"], d["y1"], color))

    for name, x, y in missing:
        if x is None:
            continue
        px, py = int(x), int(y)
        cv2.circle(output, (px, py), 16, RED, 2)
        labels.append(("missing: " + name, px - 30, py - 20, RED))

    # Place names so their boxes do not cover each other
    used = []
    pad = 8
    for text, x, y, color in labels:
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        th = th + base
        x = max(0, min(int(x), image.shape[1] - tw))
        y = max(th + pad, min(int(y), image.shape[0] - pad))
        for _ in range(40):
            box = (x - pad, y - th - pad, tw + pad * 2, th + pad * 2)
            hit = False
            for ox, oy, ow, oh in used:
                if box[0] < ox + ow and box[0] + box[2] > ox and box[1] < oy + oh and box[1] + box[3] > oy:
                    hit = True
                    break
            if not hit:
                break
            y += th + pad
            if y >= image.shape[0] - pad:
                y = th + pad
                x = min(x + 30, image.shape[1] - tw)
        used.append((x - pad, y - th - pad, tw + pad * 2, th + pad * 2))
        cv2.putText(output, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / (Path(image_path).stem + "_annotated.png")
    cv2.imwrite(str(output_path), output)

    print("\nImage:", image_path)
    print(verdict)
    if missing:
        print("Missing:")
        for name, _, _ in missing:
            print("-", name)
    if misplaced:
        print("Misplaced:")
        for name in misplaced:
            print("-", name)
    print("Output saved:", output_path)

    return {
        "image": Path(image_path).name,
        "verdict": verdict,
        "missing": [name for name, _, _ in missing],
        "misplaced": misplaced}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    parser.add_argument("--folder", type=str)
    args = parser.parse_args()

    if args.image:
        inspect_image(args.image)
    elif args.folder:
        rows = []
        for image_path in sorted(Path(args.folder).iterdir()):
            if image_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                rows.append(inspect_image(image_path))
        print("\nimage,verdict,missing,misplaced")
        for row in rows:
            missing = "; ".join(row["missing"]) if row["missing"] else "-"
            misplaced = "; ".join(row["misplaced"]) if row["misplaced"] else "-"
            print(f"{row['image']},{row['verdict']},{missing},{misplaced}")


if __name__ == "__main__":
    main()
