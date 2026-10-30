#!/usr/bin/env python3
"""
Gear assembly check (Case Study 1).

What this program does:
  1. Run a YOLO segmentation model to find parts on a photo.
  2. Compare to "good" reference photos to see if every part is present and in place.
  3. Print OK / NOT OK and save an annotated image.

Usage:
    python inspect.py --image test_images/test_01.png
    python inspect.py --folder test_images
"""

import importlib.util
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Special case: this file is named inspect.py (required by the case study).
# Python also has a built-in module called "inspect". When other libraries
# do "import inspect", they must get the built-in one, not this script.
# ---------------------------------------------------------------------------
if __name__ == "inspect":
    stdlib = Path(sys.base_prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "inspect.py"
    if not stdlib.is_file():
        stdlib = Path(sys.executable).resolve().parent / "lib" / "inspect.py"
    spec = importlib.util.spec_from_file_location("_stdlib_inspect", stdlib)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[__name__] = mod
    globals().update(mod.__dict__)

else:
    # -----------------------------------------------------------------------
    # Normal run: python inspect.py ...
    # -----------------------------------------------------------------------
    import argparse

    import cv2
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from ultralytics import YOLO

    # --- Settings ----------------------------------------------------------

    # Checklist from the PDF: each part name and how many must exist.
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
        "Cable connector plugged into the motor socket": 1,
    }

    DEFAULT_MODEL = "defect final.pt"  # YOLO weights file in this folder

    # After aligning to the base plate, a part must be within this distance
    # of its reference slot to count as "in the right place".
    SLOT_DISTANCE_OK = 0.045

    # If two detections of the same part are closer than this, treat as one
    # physical part (e.g. one screw split into two masks because a gear covers it).
    DUPLICATE_DISTANCE = 0.03

    # BGR colors for OpenCV drawing
    COLOR_OK = (40, 200, 40)       # green
    COLOR_BAD = (40, 40, 220)      # red
    COLOR_MISSING = (0, 0, 220)    # red for missing-slot markers


    # --- Simple data containers --------------------------------------------

    class Part:
        """One part detected on an image (one row from the neural network)."""

        def __init__(self, name, cx, cy, angle, scale, confidence, mask):
            self.name = name              # e.g. "Compound gear"
            self.cx = cx                  # center x in pixels
            self.cy = cy                  # center y in pixels
            self.angle = angle            # rotation of base plate (from mask shape)
            self.scale = scale            # size of mask (used when aligning)
            self.confidence = confidence  # model score 0..1
            self.mask = mask              # binary segmentation, same size as image

            # Filled in later during checking:
            self.norm_x = 0.0             # position in "base plate coordinates"
            self.norm_y = 0.0
            self.fits_slot = False        # True if matched a reference slot
            self.hide = False             # True if duplicate detection (do not draw)


    class InspectionResult:
        """Everything we report for one input image."""

        def __init__(self, image_path, verdict):
            self.image_path = image_path
            self.verdict = verdict          # "OK" or "NOT OK"
            self.missing = []               # part names that failed check
            self.misplaced = []             # reserved (extra/misplaced parts)
            self.issue_labels = []          # strings like "not found:..." / "missing:..."
            self.slot_markers = []          # where to draw circles on the image
            self.annotated_path = None      # path to saved PNG, if any


    # --- Main inspector class ----------------------------------------------

    class AssemblyInspector:
        """
        Loads the model once, learns slot positions from reference_good/,
        then can inspect any number of test images.
        """

        def __init__(self, model_path, reference_dir, conf=0.25, device="cpu"):
            # Load YOLO instance segmentation model
            self.model = YOLO(str(model_path), task="segment")
            self.conf = conf          # minimum detection confidence
            self.device = device      # "cpu" or "cuda"
            self.good_slots = {}      # part name -> array of (x,y) slot positions

            # Build template from reference photos (runs once at startup)
            self._learn_slots_from_reference(reference_dir)

        def _learn_slots_from_reference(self, reference_dir):
            """
            Read all good reference images and record where each part sits
            relative to the base plate. Cluster into one slot per required instance.
            """
            ref_list = sorted(reference_dir.glob("*.png")) + sorted(reference_dir.glob("*.jpg"))
            if len(ref_list) == 0:
                raise FileNotFoundError("No reference images in " + str(reference_dir))

            # all_positions[part_name] = list of (norm_x, norm_y) from every ref photo
            all_positions = {}
            for ref_path in ref_list:
                parts = self.run_model(ref_path)
                base = self.get_base_plate(parts)
                if base is None:
                    continue
                self.align_to_base(parts, base)
                for part in parts:
                    if part.name not in all_positions:
                        all_positions[part.name] = []
                    all_positions[part.name].append((part.norm_x, part.norm_y))

            # For each required part type, merge many points into N slot centers
            for part_name in REQUIRED_PARTS:
                need = REQUIRED_PARTS[part_name]
                if part_name not in all_positions:
                    continue
                points = np.array(all_positions[part_name])
                self.good_slots[part_name] = self._cluster_points(points, need)

        def _cluster_points(self, points, slot_count):
            """
            Simple k-means: group reference points into slot_count clusters.
            Example: 12 bolt points from 3 photos -> 4 corner slot positions.
            """
            if len(points) == 0:
                return np.zeros((slot_count, 2))

            centers = points[:slot_count].copy()
            for _ in range(30):
                # Assign each point to nearest center
                labels = []
                for pt in points:
                    best_j = 0
                    best_d = 1e9
                    for j in range(slot_count):
                        dx = pt[0] - centers[j][0]
                        dy = pt[1] - centers[j][1]
                        d = dx * dx + dy * dy
                        if d < best_d:
                            best_d = d
                            best_j = j
                    labels.append(best_j)

                # Move each center to average of its points
                for j in range(slot_count):
                    xs = []
                    ys = []
                    for i in range(len(points)):
                        if labels[i] == j:
                            xs.append(points[i][0])
                            ys.append(points[i][1])
                    if len(xs) > 0:
                        centers[j][0] = sum(xs) / len(xs)
                        centers[j][1] = sum(ys) / len(ys)
            return centers

        def get_base_plate(self, parts):
            """Return the base plate detection with highest confidence, or None."""
            best = None
            for part in parts:
                if part.name != "Base plate":
                    continue
                if best is None or part.confidence > best.confidence:
                    best = part
            return best

        def align_to_base(self, parts, base):
            """
            Rewrite each part's (norm_x, norm_y) so positions are relative to
            the base plate. Same physical layout on a shifted/rotated photo
            still maps to similar (norm_x, norm_y).
            """
            rad = np.radians(base.angle)
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)
            scale = 1.0 / max(base.scale, 1.0)
            for part in parts:
                dx = part.cx - base.cx
                dy = part.cy - base.cy
                # Rotate into base-plate frame
                rx = cos_a * dx + sin_a * dy
                ry = -sin_a * dx + cos_a * dy
                part.norm_x = rx * scale
                part.norm_y = ry * scale

        def slot_to_pixel(self, norm_x, norm_y, base):
            """Convert template slot (norm_x, norm_y) back to image pixel coordinates."""
            rad = np.radians(base.angle)
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)
            lx = norm_x * base.scale
            ly = norm_y * base.scale
            px = cos_a * lx - sin_a * ly + base.cx
            py = sin_a * lx + cos_a * ly + base.cy
            return int(round(px)), int(round(py))

        def run_model(self, image_path):
            """
            Run YOLO on one image. Returns a list of Part objects with masks.
            """
            result = self.model.predict(str(image_path), conf=self.conf, device=self.device, verbose=False)[0]
            if result.masks is None or result.boxes is None:
                return []

            height, width = result.orig_shape
            parts = []
            for i in range(len(result.boxes)):
                name = self.model.names[int(result.boxes.cls[i])]
                conf = float(result.boxes.conf[i])

                # Resize mask from model size to full image size
                small_mask = result.masks.data[i].cpu().numpy()
                mask = cv2.resize(small_mask, (width, height), interpolation=cv2.INTER_NEAREST)

                # Center and orientation from mask pixels (fallback: box center)
                ys, xs = np.where(mask > 0.5)
                if len(xs) >= 10:
                    cx = float(xs.mean())
                    cy = float(ys.mean())
                    centered = np.stack([xs - cx, ys - cy], axis=1, dtype=np.float64)
                    axis = np.linalg.eigh(np.cov(centered.T))[1][:, 1]
                    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
                    scale = float(np.sqrt(len(xs)))
                else:
                    cx = float(result.boxes.xywh[i][0])
                    cy = float(result.boxes.xywh[i][1])
                    angle = 0.0
                    scale = 1.0

                parts.append(Part(name, cx, cy, angle, scale, conf, mask))
            return parts

        def is_duplicate(self, part, kept_list):
            """
            True if this detection is probably the same physical part as one
            we already kept (double segmentation).
            """
            for other in kept_list:
                dx = part.norm_x - other.norm_x
                dy = part.norm_y - other.norm_y
                if (dx * dx + dy * dy) ** 0.5 < DUPLICATE_DISTANCE:
                    return True
                # Or masks overlap a lot
                m1 = part.mask > 0.5
                m2 = other.mask > 0.5
                overlap = np.logical_and(m1, m2).sum()
                if overlap == 0:
                    continue
                union = np.logical_or(m1, m2).sum()
                if overlap / max(int(union), 1) >= 0.15:
                    return True
            return False

        def make_label(self, part_name, found_count):
            """Text shown on image and in console."""
            if found_count == 0:
                return "not found:" + part_name
            return "missing:" + part_name

        def check_parts(self, parts):
            """
            Compare detections to REQUIRED_PARTS and good_slots.
            Sets part.fits_slot / part.hide and builds labels + markers.
            """
            missing_names = []
            misplaced = []
            labels = []
            markers = []  # each entry: (part_name, slot_index, label_text)

            base = self.get_base_plate(parts)
            if base is None:
                missing_names.append("Base plate")
                labels.append("not found:Base plate")
                return missing_names, misplaced, labels, markers

            self.align_to_base(parts, base)

            # Group detections by part name
            parts_by_name = {}
            for part in parts:
                if part.name not in parts_by_name:
                    parts_by_name[part.name] = []
                parts_by_name[part.name].append(part)

            # Check every part type in the checklist
            for part_name in REQUIRED_PARTS:
                need_count = REQUIRED_PARTS[part_name]
                group = parts_by_name.get(part_name, [])
                group.sort(key=lambda p: p.confidence, reverse=True)

                # Remove duplicate masks on the same physical part
                kept = []
                for part in group:
                    part.hide = False
                    part.fits_slot = False
                    if self.is_duplicate(part, kept):
                        part.hide = True
                    else:
                        kept.append(part)

                text = self.make_label(part_name, len(kept))
                slots = self.good_slots.get(part_name)

                # No template slots learned for this name (should be rare)
                if slots is None:
                    if len(kept) < need_count:
                        missing_names.append(part_name)
                        labels.append(text)
                    else:
                        for part in kept:
                            part.fits_slot = True
                    continue

                # Match kept detections to slot positions (Hungarian algorithm)
                filled_slots = set()
                if len(kept) > 0:
                    det_xy = []
                    for part in kept:
                        det_xy.append([part.norm_x, part.norm_y])
                    det_xy = np.array(det_xy)
                    dist = np.linalg.norm(det_xy[:, None, :] - slots[None, :, :], axis=2)
                    n_det = dist.shape[0]
                    n_slot = dist.shape[1]
                    size = max(n_det, n_slot)
                    cost = np.full((size, size), 1e6)
                    cost[0:n_det, 0:n_slot] = dist
                    rows, cols = linear_sum_assignment(cost)
                    for row, col in zip(rows, cols):
                        if row < n_det and col < n_slot and dist[row, col] <= SLOT_DISTANCE_OK:
                            kept[row].fits_slot = True
                            filled_slots.add(int(col))

                # Not enough slots filled -> assembly problem
                if len(filled_slots) < need_count:
                    missing_names.append(part_name)
                    if text not in labels:
                        labels.append(text)
                    # Mark empty slots on the output image
                    for slot_i in range(len(slots)):
                        if slot_i not in filled_slots:
                            markers.append((part_name, slot_i, text))

            return missing_names, misplaced, labels, markers

        def draw_result(self, image_path, parts, labels, markers, misplaced, save_path):
            """Draw colored masks, OK/FAIL dots, missing circles, verdict text."""
            image = cv2.imread(str(image_path))
            base = self.get_base_plate(parts)
            overlay = image.copy()

            # Layer 1: color each part mask green (OK) or red (problem)
            for part in parts:
                if part.hide:
                    continue
                if part.name == "Base plate" or part.fits_slot:
                    color = COLOR_OK
                else:
                    color = COLOR_BAD
                mask = (part.mask > 0.5).astype(np.uint8)
                tint = np.zeros_like(image)
                tint[:, :] = color
                overlay = np.where(mask[:, :, None] > 0, tint, overlay)

            out = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)

            # Layer 2: dot + OK/FAIL label at each part center
            for part in parts:
                if part.hide:
                    continue
                if part.name == "Base plate" or part.fits_slot:
                    ok = True
                    color = COLOR_OK
                else:
                    ok = False
                    color = COLOR_BAD
                x = int(part.cx)
                y = int(part.cy)
                cv2.circle(out, (x, y), 6, color, -1)
                tag = "OK" if ok else "FAIL"
                cv2.putText(out, tag, (x + 8, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            # Layer 3: circles where a required slot was not filled
            if base is not None:
                for part_name, slot_i, text in markers:
                    sx = self.good_slots[part_name][slot_i][0]
                    sy = self.good_slots[part_name][slot_i][1]
                    px, py = self.slot_to_pixel(float(sx), float(sy), base)
                    cv2.circle(out, (px, py), 14, COLOR_MISSING, 2)
                    cv2.putText(out, text, (px - 40, py - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_MISSING, 1, cv2.LINE_AA)

            # Top-left verdict and issue list
            if len(labels) == 0 and len(misplaced) == 0:
                verdict = "OK"
                verdict_color = COLOR_OK
            else:
                verdict = "NOT OK"
                verdict_color = COLOR_BAD

            cv2.putText(out, verdict, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, verdict_color, 2, cv2.LINE_AA)
            y = 70
            for line in labels:
                cv2.putText(out, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_MISSING, 1, cv2.LINE_AA)
                y = y + 22
            for name in misplaced:
                cv2.putText(out, "Misplaced/extra: " + name, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BAD, 1, cv2.LINE_AA)
                y = y + 22

            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), out)

        def inspect(self, image_path, output_dir):
            """Full pipeline for one image: detect -> check -> optional save."""
            parts = self.run_model(image_path)
            missing, misplaced, labels, markers = self.check_parts(parts)

            if len(labels) == 0 and len(misplaced) == 0:
                verdict = "OK"
            else:
                verdict = "NOT OK"

            result = InspectionResult(image_path, verdict)
            result.missing = sorted(set(missing))
            result.misplaced = misplaced
            result.issue_labels = labels
            result.slot_markers = markers

            if output_dir is not None:
                out_path = output_dir / (image_path.stem + "_annotated.png")
                self.draw_result(image_path, parts, labels, markers, misplaced, out_path)
                result.annotated_path = out_path

            return result


    # --- Command-line interface --------------------------------------------

    def print_one_result(result):
        """Pretty-print one inspection to the terminal."""
        print("Image:", result.image_path)
        print("Verdict:", result.verdict)
        for line in result.issue_labels:
            print(" ", line)
        if len(result.misplaced) > 0:
            print("Misplaced or extra:")
            for name in result.misplaced:
                print("  -", name)
        if result.annotated_path is not None:
            print("Annotated:", result.annotated_path)
        print()


    def main():
        root = Path(__file__).resolve().parent

        parser = argparse.ArgumentParser(description="Inspect gear assembly images.")
        parser.add_argument("--image", type=Path, help="Single image to inspect")
        parser.add_argument("--folder", type=Path, help="Folder of images; prints summary table")
        parser.add_argument("--model", type=Path, default=root / DEFAULT_MODEL)
        parser.add_argument("--reference-dir", type=Path, default=root / "reference_good")
        parser.add_argument("--output-dir", type=Path, default=root / "output")
        parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
        parser.add_argument("--device", default="cpu", help="cpu or cuda")
        parser.add_argument("--no-save", action="store_true", help="Skip writing annotated PNGs")
        args = parser.parse_args()

        if args.image is None and args.folder is None:
            print("Provide --image or --folder", file=sys.stderr)
            return 2
        if not args.model.is_file():
            print("Model not found:", args.model, file=sys.stderr)
            return 2

        inspector = AssemblyInspector(args.model, args.reference_dir, args.conf, args.device)
        if args.no_save:
            save_dir = None
        else:
            save_dir = args.output_dir

        valid_ext = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

        # --- Single image mode ---
        if args.image is not None:
            if not args.image.is_file():
                print("Image not found:", args.image, file=sys.stderr)
                return 2
            result = inspector.inspect(args.image, save_dir)
            print_one_result(result)
            if result.verdict == "OK":
                return 0
            return 1

        # --- Folder mode ---
        if not args.folder.is_dir():
            print("Folder not found:", args.folder, file=sys.stderr)
            return 2

        image_paths = []
        for p in sorted(args.folder.iterdir()):
            if p.suffix.lower() in valid_ext:
                image_paths.append(p)

        results = []
        for path in image_paths:
            results.append(inspector.inspect(path, save_dir))

        # Build summary table rows
        rows = []
        for r in results:
            issues = list(r.issue_labels)
            for n in r.misplaced:
                issues.append("misplaced/extra: " + n)
            if len(issues) == 0:
                issue_text = "—"
            else:
                issue_text = "; ".join(issues)
            rows.append((r.image_path.name, r.verdict, issue_text))

        # Column widths for aligned table
        col0 = len("Image")
        col1 = len("Verdict")
        col2 = len("Issues")
        for row in rows:
            if len(row[0]) > col0:
                col0 = len(row[0])
            if len(row[1]) > col1:
                col1 = len(row[1])
            if len(row[2]) > col2:
                col2 = len(row[2])

        fmt = "{:" + str(col0) + "}  {:" + str(col1) + "}  {:" + str(col2) + "}"
        print(fmt.format("Image", "Verdict", "Issues"))
        print(fmt.format("-" * col0, "-" * col1, "-" * col2))
        for row in rows:
            print(fmt.format(row[0], row[1], row[2]))
        print()

        # Detailed print for failed images only
        for r in results:
            if r.verdict != "OK":
                print_one_result(r)

        all_ok = True
        for r in results:
            if r.verdict != "OK":
                all_ok = False
                break
        if all_ok:
            return 0
        return 1


    if __name__ == "__main__":
        raise SystemExit(main())
