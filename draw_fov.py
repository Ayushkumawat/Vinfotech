import argparse
import cv2
import numpy as np

# Click the road for each arm. Top of the frame is North.
# Draw up to the stop line and leave the middle of the crossing empty.
# Copy the printed ZONES block into traffic.py.

ARMS = ["N", "S", "E", "W"]
COLORS = {
    "N": (255, 80, 0),
    "S": (0, 180, 0),
    "E": (0, 140, 255),
    "W": (200, 0, 200),
}

state = {"arm": "N", "zones": {arm: [] for arm in ARMS}, "scale": 1.0}


def print_zones():
    print("\n# paste into traffic.py")
    print("ZONES = {")
    for arm in ARMS:
        print(f'    "{arm}": {state["zones"][arm]},')
    print("}\n")


def on_mouse(event, x, y, flags, param):
    px = int(x / state["scale"])
    py = int(y / state["scale"])
    arm = state["arm"]
    if event == cv2.EVENT_LBUTTONDOWN:
        state["zones"][arm].append((px, py))
    elif event == cv2.EVENT_RBUTTONDOWN and state["zones"][arm]:
        state["zones"][arm].pop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str)
    parser.add_argument("--image", type=str)
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print("could not read video")
            return
    else:
        print("python draw_fov.py --video clip.mp4")
        return

    h, w = frame.shape[:2]
    state["scale"] = min(1.0, 960 / w, 540 / h)

    cv2.namedWindow("fov")
    cv2.setMouseCallback("fov", on_mouse)

    while True:
        disp = cv2.resize(frame, None, fx=state["scale"], fy=state["scale"])
        for arm in ARMS:
            pts = state["zones"][arm]
            if len(pts) == 0:
                continue
            vis = np.array([(int(x * state["scale"]), int(y * state["scale"])) for x, y in pts], np.int32)
            if len(pts) >= 3:
                over = disp.copy()
                cv2.fillPoly(over, [vis], COLORS[arm])
                disp = cv2.addWeighted(over, 0.35, disp, 0.65, 0)
            if len(pts) >= 2:
                cv2.polylines(disp, [vis], True, COLORS[arm], 2)
            for p in vis:
                cv2.circle(disp, (int(p[0]), int(p[1])), 4, (255, 255, 255), -1)

        cv2.putText(disp, "n s e w = arm   click = add   right click = undo   c = clear   p = print   q = quit",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(disp, "n s e w = arm   click = add   right click = undo   c = clear   p = print   q = quit",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(disp, "drawing " + state["arm"], (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[state["arm"]], 2)
        cv2.imshow("fov", disp)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("n"), ord("s"), ord("e"), ord("w")):
            state["arm"] = chr(key).upper()
        elif key == ord("c"):
            state["zones"][state["arm"]] = []
        elif key == ord("p"):
            print_zones()
        elif key == ord("q") or key == 27:
            print_zones()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
