#!/usr/bin/env python3
"""
Traffic intersection analytics (Case Study 2).

Reads a fixed-camera clip, tracks vehicles, and reports:
  - how many vehicles entered from each arm (North, South, East, West)
  - a 4x4 table of entry arm x exit arm
  - max queue length, average wait, and max wait on each arm
  - a rough speed (km/h) for each counted vehicle

Top of the frame is North. Only vehicles that pass through the intersection
are counted. Parked vehicles and through-traffic that never crosses are not.

Usage:
    python traffic.py --video "../Screencast from 2026-10-31 15-35-56.mp4"
    python traffic.py --video path/to/clip.mp4 --analyze-only
"""

import argparse
import csv
import json
import pickle
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Geometry for this camera (Coldwater downtown "Corners" webcam).
# Coordinates are pixels in the 2560x1438 screencast. The camera does not move.
# Top of the frame is North, as the brief requires.
# ---------------------------------------------------------------------------

ARMS = ("N", "S", "E", "W")
ARM_NAME = {"N": "North", "S": "South", "E": "East", "W": "West"}

# Intersection box: asphalt inside the four crosswalks.
CENTER = np.array([
    [700, 800],
    [1100, 700],
    [1600, 740],
    [2000, 880],
    [1900, 1020],
    [1500, 1080],
    [1000, 1060],
    [620, 940],
], np.int32)

# Full arm corridors (both directions). A vehicle's entry is the arm it is in
# before it reaches CENTER; the exit is the next arm it reaches after that.
ARMS_POLY = {
    "N": np.array([
        [160, 20], [640, 10], [980, 300], [1280, 560], [1580, 740],
        [1200, 880], [760, 820], [480, 500], [250, 200], [100, 70],
    ], np.int32),
    "S": np.array([
        [880, 1040], [1400, 980], [2000, 1000], [2480, 1120],
        [2520, 1430], [1180, 1430], [820, 1200],
    ], np.int32),
    "E": np.array([
        [1680, 680], [2550, 560], [2550, 980], [2140, 1080],
        [1780, 980], [1560, 840],
    ], np.int32),
    "W": np.array([
        [0, 860], [640, 800], [820, 900], [700, 1080], [0, 1160],
    ], np.int32),
}

# Inbound storage behind each stop line. Stopped vehicles here are the queue.
QUEUE_POLY = {
    # Near the stop line only. Cars parked further up the block are outside this.
    "N": np.array([
        [520, 500], [900, 450], [1250, 650],
        [1480, 770], [1120, 870], [740, 810], [500, 640],
    ], np.int32),
    "S": np.array([
        [1420, 1060], [2050, 1000], [2480, 1140],
        [2500, 1430], [1500, 1430], [1360, 1200],
    ], np.int32),
    "E": np.array([
        [1980, 720], [2550, 640], [2550, 960], [2200, 1040], [1900, 880],
    ], np.int32),
    "W": np.array([
        [0, 900], [520, 860], [700, 980], [560, 1100], [0, 1140],
    ], np.int32),
}

# One southbound through-lane, used only to scale pixels into metres.
# Image corners: far-left, far-right, near-right, near-left (toward the camera).
# World: x across the lane (metres), y along the lane (0 at the crosswalk,
# negative toward the camera). Lane width 3.35 m (11 ft). The visible stretch
# from the crosswalk to near the bottom of the frame is taken as 10 m
# (about two car lengths; the pickup in the foreground sits in that gap).
LANE_IMG = np.float32([
    [1520, 1090],
    [1860, 1050],
    [2050, 1400],
    [1580, 1430],
])
LANE_WORLD = np.float32([
    [0.0, 0.0],
    [3.35, 0.0],
    [3.35, -10.0],
    [0.0, -10.0],
])

# Stopped in the image. Near the stop line, 25 px/s is a slow creep.
# Ground speed is not used for this test: the homography is unstable far from
# the south lane that was used to scale it.
STOP_PX_S = 25.0
STOP_KMH = 8.0
# Must be stopped at least this long before it joins the queue or earns wait.
MIN_STOP_S = 0.5
# How close (pixels) a path must come to the intersection if it never sits
# inside CENTER (tight right turns at the corner).
CENTER_MARGIN_PX = 35

ZONE_COLOR = {
    "N": (80, 200, 80),
    "S": (220, 140, 40),
    "E": (60, 60, 220),
    "W": (40, 210, 210),
    "C": (240, 240, 240),
}

VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck"}


def inside(poly, x, y):
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


def distance_to(poly, x, y):
    """Positive inside the polygon, negative outside."""
    return float(cv2.pointPolygonTest(poly, (float(x), float(y)), True))


def ignored_point(x, y):
    """Logo banner and the park/lot above the east road are not traffic lanes."""
    if x > 2050 and y < 175:
        return True
    if y < 480 and x > 1900:
        return True
    return False


def zone_at(x, y, previous):
    """
    Arm polygons win over the centre where they overlap, so the stop line is
    the moment a vehicle leaves its arm. If several arms contain the point,
    stick with the previous arm to avoid flicker on a shared edge.
    """
    hits = [name for name in ARMS if inside(ARMS_POLY[name], x, y)]
    if previous in hits:
        return previous
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        hits.sort(key=lambda name: distance_to(ARMS_POLY[name], x, y), reverse=True)
        return hits[0]
    if inside(CENTER, x, y):
        return "center"
    return None


def build_homography():
    homography = cv2.getPerspectiveTransform(LANE_IMG, LANE_WORLD)
    return homography


def to_ground(homography, x, y):
    pt = np.array([[[x, y]]], dtype=np.float32)
    gx, gy = cv2.perspectiveTransform(pt, homography)[0, 0]
    return float(gx), float(gy)


def smooth_speeds(samples):
    """
    Pixel speed is used to decide who is stopped.
    km/h is kept only where one pixel is a sensible fraction of a metre
    (about 1.5–22 cm). Outside that band the homography is near its horizon
    and the number is discarded.
    """
    n = len(samples)
    px_raw = [0.0] * n
    kmh_raw = [None] * n
    for i in range(1, n):
        dt = samples[i]["t"] - samples[i - 1]["t"]
        if dt <= 1e-3:
            px_raw[i] = px_raw[i - 1]
            kmh_raw[i] = kmh_raw[i - 1]
            continue
        dx = samples[i]["x"] - samples[i - 1]["x"]
        dy = samples[i]["y"] - samples[i - 1]["y"]
        step_px = (dx * dx + dy * dy) ** 0.5
        px_per_s = step_px / dt
        gx, gy = samples[i]["gx"], samples[i]["gy"]
        pgx, pgy = samples[i - 1]["gx"], samples[i - 1]["gy"]
        metres = ((gx - pgx) ** 2 + (gy - pgy) ** 2) ** 0.5
        px_raw[i] = px_per_s
        if step_px < 0.8:
            kmh_raw[i] = 0.0
            continue
        metres_per_px = metres / step_px
        if 0.015 <= metres_per_px <= 0.22:
            kmh_raw[i] = (metres / dt) * 3.6
        else:
            kmh_raw[i] = None
    px_out, kmh_out, ok_out = [], [], []
    for i in range(n):
        px_window = sorted(px_raw[max(0, i - 2): i + 3])
        px_out.append(px_window[len(px_window) // 2])
        good = [v for v in kmh_raw[max(0, i - 2): i + 3] if v is not None]
        if not good:
            kmh_out.append(None)
            ok_out.append(False)
        else:
            good.sort()
            kmh_out.append(good[len(good) // 2])
            ok_out.append(True)
    return px_out, kmh_out, ok_out


def assign_zones(samples):
    prev = None
    zones = []
    for s in samples:
        z = zone_at(s["x"], s["y"], prev)
        zones.append(z)
        if z is not None:
            prev = z
    # Fill a one-sample hole so a single miss does not split a run.
    for i in range(1, len(zones) - 1):
        if zones[i] is None and zones[i - 1] is not None and zones[i - 1] == zones[i + 1]:
            zones[i] = zones[i - 1]
    return zones


def zone_runs(zones, samples):
    runs = []
    for z, s in zip(zones, samples):
        if not runs or runs[-1]["zone"] != z:
            runs.append({"zone": z, "t0": s["t"], "t1": s["t"], "n": 1, "i0": len(runs) and 0})
            runs[-1]["i0"] = sum(r["n"] for r in runs[:-1])
        else:
            runs[-1]["t1"] = s["t"]
            runs[-1]["n"] += 1
    for r in runs:
        r["dur"] = r["t1"] - r["t0"]
    return runs


def near_center(samples, i0, i1):
    for s in samples[i0:i1]:
        if distance_to(CENTER, s["x"], s["y"]) >= -CENTER_MARGIN_PX:
            return True
    return False


def movement_for(samples, zones):
    """
    Return entry, exit, entry_time, exit_time, wait_s, speed_kmh
    or None if this track never passes through the intersection.
    """
    runs = zone_runs(zones, samples)
    entry = exit_arm = None
    entry_time = exit_time = None

    center_idxs = [i for i, r in enumerate(runs) if r["zone"] == "center" and (r["n"] >= 2 or r["dur"] >= 0.15)]
    if center_idxs:
        first = center_idxs[0]
        last = center_idxs[-1]
        for j in range(first - 1, -1, -1):
            r = runs[j]
            if r["zone"] in ARMS and (r["n"] >= 2 or r["dur"] >= 0.2):
                entry = r["zone"]
                entry_time = runs[first]["t0"]
                break
        for j in range(last + 1, len(runs)):
            r = runs[j]
            if r["zone"] in ARMS and (r["n"] >= 2 or r["dur"] >= 0.2):
                exit_arm = r["zone"]
                exit_time = r["t0"]
                break
    else:
        # Arms in order, ignoring gaps. Catches a right turn that only clips
        # the corner of CENTER, and a crossing where CENTER is only one sample.
        arm_runs = [r for r in runs if r["zone"] in ARMS and r["n"] >= 2]
        for a, b in zip(arm_runs, arm_runs[1:]):
            if a["zone"] == b["zone"]:
                continue
            if near_center(samples, a["i0"], b["i0"] + b["n"]):
                entry, exit_arm = a["zone"], b["zone"]
                entry_time, exit_time = a["t1"], b["t0"]
                break

    if entry is None or exit_arm is None:
        return None
    # A one-frame hop across the stop line and back is not a U-turn.
    if entry == exit_arm and (exit_time - entry_time) < 1.0:
        return None

    wait = wait_time(samples, zones, entry, entry_time)
    speed = movement_speed(samples, zones, entry_time, exit_time)
    return {
        "entry": entry,
        "exit": exit_arm,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "wait_s": wait,
        "speed_kmh": speed,
    }


def wait_time(samples, zones, entry, entry_time):
    """
    Seconds from when the vehicle stopped in this arm's queue until it
    crosses the stop line. Brief creeps (under 2.5 s) stay in the same stop.
    If the vehicle was already rolling for more than 4 s before the line,
    that earlier stop is not this wait.
    """
    episodes = []
    for s, z in zip(samples, zones):
        if s["t"] > entry_time + 0.05:
            break
        slow = (
            z == entry
            and inside(QUEUE_POLY[entry], s["x"], s["y"])
            and s["px_speed"] < STOP_PX_S
        )
        if not slow:
            continue
        if episodes and s["t"] - episodes[-1][1] <= 2.5:
            episodes[-1][1] = s["t"]
        else:
            episodes.append([s["t"], s["t"]])
    if not episodes:
        return 0.0
    start, end = episodes[-1]
    if end - start < MIN_STOP_S:
        return 0.0
    if entry_time - end > 4.0:
        return 0.0
    return round(max(0.0, entry_time - start), 2)


def movement_speed(samples, zones, entry_time, exit_time):
    """Median ground speed while crossing, ignoring the stopped samples."""
    chosen = []
    for s, z in zip(samples, zones):
        if s["t"] < entry_time or s["t"] > exit_time + 0.3:
            continue
        if z == "center" and s["speed_ok"] and s["speed"] is not None and 3.0 <= s["speed"] <= 110:
            chosen.append(s["speed"])
    if len(chosen) < 2:
        for s in samples:
            if entry_time <= s["t"] <= exit_time and s["speed_ok"] and s["speed"] is not None and 3.0 <= s["speed"] <= 110:
                chosen.append(s["speed"])
    if not chosen:
        return None
    chosen.sort()
    return round(chosen[len(chosen) // 2], 1)


def is_parked(samples, zones):
    """
    A box that never moves is a parked car or a static false detection
    (a window, a sign). A vehicle waiting at the light and then driving
    has a much larger travel and is not marked parked.
    """
    del zones
    if len(samples) < 5:
        return False
    dur = samples[-1]["t"] - samples[0]["t"]
    if dur < 5.0:
        return False
    xs = [s["x"] for s in samples]
    ys = [s["y"] for s in samples]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    return span < 22.0


def build_tracks(frames, homography):
    buckets = defaultdict(list)
    for fr in frames:
        for x1, y1, x2, y2, tid, cls_name, conf in fr["boxes"]:
            cx = 0.5 * (x1 + x2)
            cy = y2  # bottom centre: closest we have to the wheels
            gx, gy = to_ground(homography, cx, cy)
            buckets[int(tid)].append({
                "t": fr["t"],
                "frame": fr["index"],
                "x": cx,
                "y": cy,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cls": cls_name,
                "conf": conf,
                "gx": gx,
                "gy": gy,
            })

    tracks = []
    for tid, samples in buckets.items():
        samples.sort(key=lambda s: s["t"])
        # Drop a repeated id on the same timestamp (should not happen).
        dedup = []
        for s in samples:
            if dedup and abs(s["t"] - dedup[-1]["t"]) < 1e-4:
                if s["conf"] > dedup[-1]["conf"]:
                    dedup[-1] = s
                continue
            dedup.append(s)
        samples = dedup
        if len(samples) < 3:
            continue
        px_speeds, kmh_speeds, speed_ok = smooth_speeds(samples)
        for s, px, kmh, good in zip(samples, px_speeds, kmh_speeds, speed_ok):
            s["px_speed"] = px
            s["speed"] = kmh
            s["speed_ok"] = good and kmh is not None
        zones = assign_zones(samples)
        parked = is_parked(samples, zones)
        move = None if parked else movement_for(samples, zones)
        cls = Counter(s["cls"] for s in samples).most_common(1)[0][0]
        tracks.append({
            "id": tid,
            "cls": cls,
            "samples": samples,
            "zones": zones,
            "parked": parked,
            "move": move,
        })
    tracks.sort(key=lambda t: t["id"])
    return tracks


def queue_series(frames, tracks):
    """Per processed frame, how many vehicles are stopped in each queue."""
    by_id = {t["id"]: t for t in tracks}
    # Map (track id, frame index) -> sample index
    index = {}
    for t in tracks:
        for i, s in enumerate(t["samples"]):
            index[(t["id"], s["frame"])] = i

    series = []
    max_q = {a: 0 for a in ARMS}
    max_q_t = {a: None for a in ARMS}
    for fr in frames:
        counts = {a: 0 for a in ARMS}
        seen = set()
        for box in fr["boxes"]:
            tid = int(box[4])
            if tid in seen or tid not in by_id:
                continue
            seen.add(tid)
            tr = by_id[tid]
            if tr["parked"]:
                continue
            key = (tid, fr["index"])
            if key not in index:
                continue
            i = index[key]
            s = tr["samples"][i]
            z = tr["zones"][i]
            if z not in ARMS:
                continue
            if not inside(QUEUE_POLY[z], s["x"], s["y"]):
                continue
            if s["px_speed"] >= STOP_PX_S:
                continue
            # Count only after it has already been slow for a short stretch.
            if not stopped_long_enough(tr, i):
                continue
            counts[z] += 1
        for a in ARMS:
            if counts[a] > max_q[a]:
                max_q[a] = counts[a]
                max_q_t[a] = fr["t"]
        series.append({"t": fr["t"], "q": counts})
    return series, max_q, max_q_t


def stopped_long_enough(track, i):
    s = track["samples"][i]
    z = track["zones"][i]
    t_now = s["t"]
    # Walk backward while still slow and in the same queue.
    start = t_now
    j = i
    while j >= 0:
        sj = track["samples"][j]
        zj = track["zones"][j]
        if zj != z or sj["px_speed"] >= STOP_PX_S or not inside(QUEUE_POLY[z], sj["x"], sj["y"]):
            break
        start = sj["t"]
        j -= 1
    return (t_now - start) >= MIN_STOP_S


def summarise(tracks, max_q, max_q_t, video_meta):
    matrix = {a: {b: 0 for b in ARMS} for a in ARMS}
    entry_counts = {a: 0 for a in ARMS}
    waits = {a: [] for a in ARMS}
    movements = []
    for tr in tracks:
        mv = tr["move"]
        if not mv:
            continue
        matrix[mv["entry"]][mv["exit"]] += 1
        entry_counts[mv["entry"]] += 1
        waits[mv["entry"]].append(mv["wait_s"])
        movements.append({
            "track_id": tr["id"],
            "class": tr["cls"],
            "entry": mv["entry"],
            "exit": mv["exit"],
            "entry_time_s": round(mv["entry_time"], 2),
            "exit_time_s": round(mv["exit_time"], 2),
            "wait_s": mv["wait_s"],
            "speed_kmh": mv["speed_kmh"],
        })

    def avg(xs):
        if not xs:
            return None
        return round(sum(xs) / len(xs), 2)

    queues = {}
    for a in ARMS:
        queues[a] = {
            "max_queue": max_q[a],
            "max_queue_time_s": None if max_q_t[a] is None else round(max_q_t[a], 2),
            "avg_wait_s": avg(waits[a]),
            "max_wait_s": None if not waits[a] else round(max(waits[a]), 2),
            "vehicles_counted": len(waits[a]),
        }

    speeds = [m["speed_kmh"] for m in movements if m["speed_kmh"] is not None]
    report = {
        "video": video_meta,
        "entry_counts": {ARM_NAME[a]: entry_counts[a] for a in ARMS},
        "turning_movements": {
            ARM_NAME[a]: {ARM_NAME[b]: matrix[a][b] for b in ARMS} for a in ARMS
        },
        "queues_and_wait": {ARM_NAME[a]: queues[a] for a in ARMS},
        "speed": {
            "vehicles_with_speed": len(speeds),
            "median_kmh": None if not speeds else round(sorted(speeds)[len(speeds) // 2], 1),
            "note": (
                "Homography from one south through-lane: 3.35 m wide, "
                "10 m from the crosswalk to near the bottom of the frame. "
                "East-west speeds use the lane width and are the more trustworthy ones. "
                "Treat speeds as about ±30%, and ignore them far up the north arm."
            ),
        },
        "movements": movements,
        "assumptions": ASSUMPTIONS,
    }
    return report, matrix, entry_counts, queues


ASSUMPTIONS = [
    "The top of the frame is North. The four arms are the roads that leave the junction toward the top (North), bottom (South), right (East) and left (West).",
    "The stop line on each arm is the boundary between that arm polygon and the intersection polygon, drawn along the crosswalk.",
    "A vehicle is counted only when its track is seen in an entry arm and later in an exit arm, and the path goes through the intersection (or within 35 px of it). Parked vehicles and vehicles that never cross are not counted.",
    "A vehicle is in the queue when its box moves slower than 25 pixels per second for at least 0.5 s inside that arm's queue polygon. The north queue polygon is only the lanes from the crosswalk back part-way up the block, so cars parked further up the street are outside it.",
    "Wait time runs from the start of the stop in the queue until the vehicle crosses the stop line. Creeps shorter than 2.5 s stay inside the same stop. If the vehicle has already been moving for more than 4 s when it reaches the line, that earlier stop is not counted. A vehicle that never stops for 0.5 s has wait 0 and is included in the average. Vehicles still waiting when the clip ends are not in the average, because their wait is unfinished.",
    "The clip is analysed at 10 fps (every 3rd frame of the 29.7 fps recording). Town traffic moves a short distance between samples, which is enough to keep a track while a vehicle is stopped. Full frame rate would not change who crossed.",
    "Detection is YOLO11x and tracking is ByteTrack. Counted classes are car, motorcycle, bus and truck. People and bicycles are not counted.",
    "The ground position of a vehicle is the bottom centre of its box.",
    "One movement is stored per track id. A track that is already inside the intersection on the first sample is not counted. A U-turn (same entry and exit) stays on the diagonal of the table.",
    "A track whose box stays inside a 22-pixel span for 5 seconds or more is treated as parked or as a static false detection, and is left out of the queue.",
    "Lost tracks are kept for about 5 s so a signal pole does not split one vehicle into two.",
    "Speed uses a flat-road homography. The south through lane is assumed 3.35 m (11 ft) wide, and the visible piece from the crosswalk to near the bottom edge is assumed 10 m. Speeds are reported only where one pixel is between 1.5 cm and 22 cm. That scale is rough, about ±30 percent. East-west motion is more trustworthy than motion toward the horizon.",
    "This recording is 2 minutes 40 seconds, shorter than the 10–15 minute window in the brief, so the counts cover only this clip and may include fewer than three full signal cycles.",
    "The YouTube title, the CBPU logo and the park beyond the east arm are masked out. They are not part of the lanes.",
]


def format_summary(report, matrix, entry_counts, queues):
    lines = []
    lines.append("Traffic intersection summary")
    lines.append("=" * 64)
    meta = report["video"]
    lines.append(
        f"Clip: {meta['duration_s']:.1f}s at {meta['source_fps']:.2f} fps, "
        f"analysed at {meta['analysis_fps']:.2f} fps ({meta['frames_analysed']} frames)"
    )
    lines.append(f"Model: {meta['model']}")
    lines.append("")
    lines.append("Vehicles entering from each arm")
    lines.append("-" * 64)
    for a in ARMS:
        lines.append(f"  {ARM_NAME[a]:<8} {entry_counts[a]:>4}")
    lines.append(f"  {'Total':<8} {sum(entry_counts.values()):>4}")
    lines.append("")
    lines.append("Turning movements (rows = entry, columns = exit)")
    lines.append("-" * 64)
    header = "          " + "".join(f"{ARM_NAME[b]:>8}" for b in ARMS)
    lines.append(header)
    for a in ARMS:
        row = f"  {ARM_NAME[a]:<8}" + "".join(f"{matrix[a][b]:>8}" for b in ARMS)
        lines.append(row)
    lines.append("")
    lines.append("Queues and wait")
    lines.append("-" * 64)
    lines.append(f"  {'Arm':<8} {'Max queue':>10} {'Avg wait s':>12} {'Max wait s':>12} {'Counted':>8}")
    for a in ARMS:
        q = queues[a]
        avg = "-" if q["avg_wait_s"] is None else f"{q['avg_wait_s']:.1f}"
        mx = "-" if q["max_wait_s"] is None else f"{q['max_wait_s']:.1f}"
        lines.append(f"  {ARM_NAME[a]:<8} {q['max_queue']:>10} {avg:>12} {mx:>12} {q['vehicles_counted']:>8}")
    lines.append("")
    sp = report["speed"]
    med = "-" if sp["median_kmh"] is None else f"{sp['median_kmh']:.0f} km/h"
    lines.append(f"Speed (bonus): {sp['vehicles_with_speed']} vehicles, median {med}")
    lines.append(sp["note"])
    lines.append("")
    lines.append("Counted vehicles")
    lines.append("-" * 64)
    lines.append(f"  {'ID':>5} {'Class':<12} {'Move':<12} {'Wait s':>8} {'km/h':>8} {'At s':>8}")
    for m in report["movements"]:
        spd = "-" if m["speed_kmh"] is None else f"{m['speed_kmh']:.0f}"
        move = f"{m['entry']}->{m['exit']}"
        lines.append(
            f"  {m['track_id']:>5} {m['class']:<12} {move:<12} {m['wait_s']:>8.1f} {spd:>8} {m['entry_time_s']:>8.1f}"
        )
    return "\n".join(lines) + "\n"


def write_reports(out_dir, report, matrix, entry_counts, queues):
    out_dir.mkdir(parents=True, exist_ok=True)
    text = format_summary(report, matrix, entry_counts, queues)
    (out_dir / "summary.txt").write_text(text)
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["track_id", "class", "entry", "exit", "entry_time_s", "exit_time_s", "wait_s", "speed_kmh"],
        )
        writer.writeheader()
        for row in report["movements"]:
            writer.writerow(row)
    print(text)
    print(f"Wrote {out_dir / 'summary.txt'}")
    print(f"Wrote {out_dir / 'results.json'}")
    print(f"Wrote {csv_path}")


def draw_zones(frame):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [CENTER], ZONE_COLOR["C"])
    for name in ARMS:
        cv2.fillPoly(overlay, [ARMS_POLY[name]], ZONE_COLOR[name])
        cv2.fillPoly(overlay, [QUEUE_POLY[name]], ZONE_COLOR[name])
    painted = cv2.addWeighted(overlay, 0.22, frame, 0.78, 0)
    cv2.polylines(painted, [CENTER], True, (255, 255, 255), 2)
    for name in ARMS:
        cv2.polylines(painted, [ARMS_POLY[name]], True, ZONE_COLOR[name], 2)
    for name, poly in list(ARMS_POLY.items()):
        x, y = poly.mean(axis=0).astype(int)
        cv2.putText(painted, ARM_NAME[name], (int(x) - 40, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(painted, ARM_NAME[name], (int(x) - 40, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, ZONE_COLOR[name], 2, cv2.LINE_AA)
    return painted


def save_trajectory_plot(video_path, tracks, out_path):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, 40_000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return
    canvas = draw_zones(frame)
    rng = np.random.default_rng(0)
    for tr in tracks:
        color = tuple(int(c) for c in rng.integers(40, 255, size=3))
        pts = np.array([(int(s["x"]), int(s["y"])) for s in tr["samples"]], np.int32)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], False, color, 2)
        if tr["move"]:
            mv = tr["move"]
            p = pts[len(pts) // 2]
            label = f"{tr['id']} {mv['entry']}->{mv['exit']}"
            cv2.putText(canvas, label, (int(p[0]), int(p[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, (int(p[0]), int(p[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    small = cv2.resize(canvas, (1400, 786))
    cv2.imwrite(str(out_path), small)


def lookup_sample(track, frame_index):
    # Samples are in time order and frame indexes increase.
    for s, z in zip(track["samples"], track["zones"]):
        if s["frame"] == frame_index:
            return s, z
    return None, None


def render(video_path, frames, tracks, entry_so_far_builder, out_path, queues_by_t):
    by_id = {t["id"]: t for t in tracks}
    cap = cv2.VideoCapture(str(video_path))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = 1280, 720
    fps = frames[1]["t"] - frames[0]["t"] if len(frames) > 1 else 0.1
    out_fps = 1.0 / fps if fps > 1e-3 else 10.0

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}", "-pix_fmt", "bgr24", "-r", f"{out_fps:.4f}",
        "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-preset", "veryfast",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    wanted = {fr["index"]: fr for fr in frames}
    snapshots = {15, 45, 90, 130}
    saved = set()
    snap_dir = out_path.parent
    index = 0
    fi = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index not in wanted:
                index += 1
                continue
            fr = wanted[index]
            painted = draw_zones(frame)
            counts_now = entry_so_far_builder(fr["t"])
            q_now = queues_by_t[fi]
            for box in fr["boxes"]:
                x1, y1, x2, y2, tid, cls_name, conf = box
                tid = int(tid)
                tr = by_id.get(tid)
                color = (230, 210, 40)
                label = f"{tid} {cls_name}"
                if tr is not None:
                    sample, zone = lookup_sample(tr, fr["index"])
                    if sample is not None and sample["speed_ok"] and sample["speed"] is not None and sample["speed"] >= 3:
                        label += f" {sample['speed']:.0f}km/h"
                    mv = tr["move"]
                    if mv and mv["exit_time"] <= fr["t"] + 0.05:
                        color = (40, 200, 40)
                        label = f"{tid} {mv['entry']}->{mv['exit']}"
                    elif zone in ARMS and sample is not None and sample["px_speed"] < STOP_PX_S and inside(QUEUE_POLY[zone], sample["x"], sample["y"]):
                        color = (0, 150, 255)
                        label += " queue"
                    # short trail
                    pts = [(int(s["x"]), int(s["y"])) for s in tr["samples"] if s["t"] <= fr["t"]][-12:]
                    if len(pts) >= 2:
                        cv2.polylines(painted, [np.array(pts, np.int32)], False, color, 2)
                cv2.rectangle(painted, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(painted, label, (int(x1), max(20, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(painted, label, (int(x1), max(20, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            small = cv2.resize(painted, (out_w, out_h))
            draw_hud(small, counts_now, q_now, fr["t"])
            sec = int(fr["t"])
            if sec in snapshots and sec not in saved:
                cv2.imwrite(str(snap_dir / f"frame_{sec:03d}s.jpg"), small)
                saved.add(sec)
            proc.stdin.write(small.tobytes())
            fi += 1
            index += 1
            if fi % 200 == 0:
                print(f"  rendered {fi}/{len(frames)}", flush=True)
    finally:
        cap.release()
        proc.stdin.close()
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg exited with {code}")
    print(f"Wrote {out_path}")


def draw_hud(frame, entries, queue, t_s):
    panel = frame.copy()
    cv2.rectangle(panel, (16, 16), (430, 118), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(panel, 0.55, frame, 0.45, 0)
    e = "  ".join(f"{a}:{entries[a]}" for a in ARMS)
    q = "  ".join(f"{a}:{queue[a]}" for a in ARMS)
    cv2.putText(frame, f"t={t_s:5.1f}s   entries  {e}", (28, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"queue    {q}", (28, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "green = counted    orange = queued", (28, 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


def default_model():
    here = Path(__file__).resolve().parent
    candidates = [
        here / "yolo11x.pt",
        Path("/home/aiunika-5090/Downloads/yolo11x.pt"),
        here / "yolo26n.pt",
        Path("/home/aiunika-5090/Downloads/yolo26n.pt"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return Path("yolo11n.pt")


def default_video():
    here = Path(__file__).resolve().parent
    candidate = here.parent / "Screencast from 2026-10-31 15-35-56.mp4"
    return candidate


def pick_device(asked):
    if asked:
        return asked
    import torch
    return "0" if torch.cuda.is_available() else "cpu"


def detect(video_path, model_path, device, imgsz, target_fps, out_pkl, limit_s):
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = model.names
    class_ids = [i for i, name in names.items() if name in VEHICLE_NAMES]
    if not class_ids:
        raise SystemExit(f"Model {model_path} has no car/truck/bus/motorcycle classes: {names}")
    print(f"Classes: { {i: names[i] for i in class_ids} }")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(round(src_fps / target_fps)))
    analysis_fps = src_fps / stride
    print(f"Video {width}x{height}, {src_fps:.2f} fps, {n_frames} frames")
    print(f"Analysing every {stride} frame(s) -> {analysis_fps:.2f} fps on {device}")

    tracker = str(Path(__file__).resolve().parent / "bytetrack_traffic.yaml")
    frames = []
    index = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = index / src_fps
        if limit_s is not None and t > limit_s:
            break
        if index % stride != 0:
            index += 1
            continue
        if frame.mean() < 25:
            index += 1
            continue
        result = model.track(
            frame,
            persist=True,
            tracker=tracker,
            classes=class_ids,
            conf=0.25,
            iou=0.5,
            imgsz=imgsz,
            device=device,
            verbose=False,
            agnostic_nms=True,
        )[0]
        boxes = []
        if result.boxes is not None and result.boxes.id is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            clss = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), tid, cid, conf in zip(xyxy, ids, clss, confs):
                bw, bh = x2 - x1, y2 - y1
                if bw * bh < 280 or bh < 12:
                    continue
                if y2 > height - 12 and bh < 55:
                    continue
                cx, cy = 0.5 * (x1 + x2), y2
                if ignored_point(cx, cy):
                    continue
                boxes.append([
                    float(x1), float(y1), float(x2), float(y2),
                    int(tid), names[int(cid)], float(conf),
                ])
        frames.append({"index": index, "t": t, "boxes": boxes})
        kept += 1
        if kept % 100 == 0:
            print(f"  detected {kept} frames, t={t:.1f}s, boxes={len(boxes)}", flush=True)
        index += 1
    cap.release()

    payload = {
        "video": str(video_path),
        "model": str(model_path),
        "source_fps": src_fps,
        "analysis_fps": analysis_fps,
        "stride": stride,
        "width": width,
        "height": height,
        "frames": frames,
    }
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as f:
        pickle.dump(payload, f)
    print(f"Saved detections: {out_pkl} ({len(frames)} frames)")
    return payload


def analyse(payload, video_path, out_dir, render_video):
    homography = build_homography()
    # Sanity: the two far corners must come out 3.35 m apart.
    g0 = to_ground(homography, *LANE_IMG[0])
    g1 = to_ground(homography, *LANE_IMG[1])
    width_m = ((g0[0] - g1[0]) ** 2 + (g0[1] - g1[1]) ** 2) ** 0.5
    print(f"Homography check: lane width maps to {width_m:.2f} m (expected 3.35)")

    frames = payload["frames"]
    tracks = build_tracks(frames, homography)
    series, max_q, max_q_t = queue_series(frames, tracks)
    meta = {
        "path": str(video_path),
        "duration_s": frames[-1]["t"] if frames else 0,
        "source_fps": payload["source_fps"],
        "analysis_fps": payload["analysis_fps"],
        "frames_analysed": len(frames),
        "width": payload["width"],
        "height": payload["height"],
        "model": payload["model"],
    }
    report, matrix, entry_counts, queues = summarise(tracks, max_q, max_q_t, meta)
    write_reports(out_dir, report, matrix, entry_counts, queues)
    save_trajectory_plot(video_path, tracks, out_dir / "trajectories.jpg")
    print(f"Wrote {out_dir / 'trajectories.jpg'}")

    n_parked = sum(1 for t in tracks if t["parked"])
    print(f"Tracks: {len(tracks)}  counted: {len(report['movements'])}  parked-like: {n_parked}")

    if render_video and frames:
        # Running entry totals at time t: movements whose exit is at or before t.
        moves = report["movements"]

        def entries_at(t):
            counts = {a: 0 for a in ARMS}
            for m in moves:
                if m["exit_time_s"] <= t + 0.05:
                    counts[m["entry"]] += 1
            return counts

        q_by_frame = [row["q"] for row in series]
        render(
            video_path, frames, tracks, entries_at,
            out_dir / "annotated.mp4", q_by_frame,
        )
    return report


def main():
    parser = argparse.ArgumentParser(description="Traffic intersection analytics (Case Study 2)")
    parser.add_argument("--video", type=Path, default=default_video(), help="Fixed-camera traffic clip")
    parser.add_argument("--model", type=Path, default=default_model(), help="YOLO detection weights")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output_traffic")
    parser.add_argument("--device", default=None, help="cuda device, e.g. 0, or cpu")
    parser.add_argument("--imgsz", type=int, default=1600)
    parser.add_argument("--fps", type=float, default=10.0, help="Analysis frame rate")
    parser.add_argument("--limit-seconds", type=float, default=None, help="Stop early (debug)")
    parser.add_argument("--redetect", action="store_true", help="Run detection again even if a cache exists")
    parser.add_argument("--analyze-only", action="store_true", help="Skip the annotated video")
    args = parser.parse_args()

    if not args.video.is_file():
        raise SystemExit(f"Video not found: {args.video}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "detections.pkl"
    device = pick_device(args.device)

    if args.redetect or not cache.is_file() or args.limit_seconds is not None:
        if not args.model.is_file() and str(args.model) not in {"yolo11n.pt", "yolo11s.pt"}:
            # ultralytics can fetch an official name; a missing local path cannot.
            if not args.model.exists():
                print(f"Model file not found ({args.model}); asking ultralytics for {args.model.name}")
        payload = detect(
            args.video, args.model, device, args.imgsz, args.fps, cache, args.limit_seconds,
        )
    else:
        print(f"Using cached detections: {cache}")
        with cache.open("rb") as f:
            payload = pickle.load(f)

    analyse(payload, args.video, args.output_dir, render_video=not args.analyze_only)


if __name__ == "__main__":
    main()
