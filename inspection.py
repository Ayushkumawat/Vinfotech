import argparse
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


# Load YOLO model

model = YOLO(MODEL_PATH, task="segment")

# Each good photo: part name -> list of (x, y) centers. These are the possible layouts.
REF_LAYOUTS = []
for ref_path in sorted(Path("reference_good").glob("*.png")):
    ref = model.predict(str(ref_path), conf=0.15, verbose=False)[0]
    layout = {}
    if ref.boxes is not None:
        for i in range(len(ref.boxes)):
            name = model.names[int(ref.boxes.cls[i])]
            x1, y1, x2, y2 = ref.boxes.xyxy[i]
            layout.setdefault(name, []).append(((float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2))
    REF_LAYOUTS.append(layout)

# Inspect one image

def inspect_image(image_path):

    image = cv2.imread(str(image_path))
    result = model.predict(image,conf=0.15,verbose=False)[0]
    detected_counts = {}
    centers = {}
    labels = []

    output = image.copy()

    # Process detected objects
    if result.masks is not None:
        # Draw the base plate first so smaller parts stay on top of it
        order = sorted(range(len(result.masks.data)),key=lambda i: model.names[int(result.boxes.cls[i])] == "Base plate",reverse=True)

        for i in order:
            mask = result.masks.data[i]
            class_id = int(result.boxes.cls[i])
            class_name = model.names[class_id]

            # Count detected parts
            detected_counts[class_name] = (detected_counts.get(class_name, 0) + 1)

            # Resize mask to original image size
            mask = mask.cpu().numpy()
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST)

            mask = mask > 0.5

            # Light tint for the big base plate; stronger tint for other parts
            alpha = 0.15 if class_name == "Base plate" else 0.45
            color = np.array(GREEN, dtype=np.float32)
            output[mask] = (output[mask].astype(np.float32) * (1 - alpha) + color * alpha).astype(np.uint8)

            # Add class name
            x1, y1, x2, y2 = result.boxes.xyxy[i]
            cx = (float(x1) + float(x2)) / 2
            cy = (float(y1) + float(y2)) / 2
            centers.setdefault(class_name, []).append((cx, cy))

            labels.append((class_name, int(x1), int(y1), GREEN))

    # Check for missing parts
    missing_parts = []

    for name, required_count in REQUIRED_PARTS.items():
        found_count = detected_counts.get(name, 0)
        if found_count < required_count:
            missing_parts.append(f"{name} (Required: {required_count}, Found: {found_count})")

    # Print missing parts
    print("\nImage:", image_path)

    if missing_parts:
        print("Missing parts:")
        for part in missing_parts:
            print("-", part)
    else:
        print("All required parts detected!")

    # Mark where a missing part should be, using the reference photo that
    # best matches the parts we did find (shift only, from the base plate).
    if "Base plate" in centers:
        bx, by = centers["Base plate"][0]
        best_layout = None
        best_err = 1e18
        for layout in REF_LAYOUTS:
            if "Base plate" not in layout:
                continue
            rbx, rby = layout["Base plate"][0]
            errs = []
            for name, pts in centers.items():
                if name == "Base plate" or name not in layout:
                    continue
                dx = pts[0][0] - bx
                dy = pts[0][1] - by
                rdx = layout[name][0][0] - rbx
                rdy = layout[name][0][1] - rby
                errs.append((dx - rdx) ** 2 + (dy - rdy) ** 2)
            err = sum(errs) / len(errs) if errs else 1e18
            if err < best_err:
                best_err = err
                best_layout = layout
        if best_layout is not None:
            rbx, rby = best_layout["Base plate"][0]
            for name, required_count in REQUIRED_PARTS.items():
                found = centers.get(name, [])
                if len(found) >= required_count:
                    continue
                for rx, ry in best_layout.get(name, []):
                    px = int(bx + (rx - rbx))
                    py = int(by + (ry - rby))
                    already_there = False
                    for fx, fy in found:
                        if (fx - px) ** 2 + (fy - py) ** 2 < 80 ** 2:
                            already_there = True
                    if already_there:
                        continue
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

    # Save output image

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / (Path(image_path).stem + "_annotated.png")

    cv2.imwrite(str(output_path), output)

    print("Output saved:", output_path)


# Main function

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str)
    parser.add_argument("--folder", type=str)

    args = parser.parse_args()

    if args.image:

        inspect_image(args.image)

    elif args.folder:

        for image_path in Path(args.folder).iterdir():

            if image_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:

                inspect_image(image_path)


if __name__ == "__main__":
    main()
