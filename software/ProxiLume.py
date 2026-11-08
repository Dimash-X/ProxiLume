import sys
import threading
import time
import traceback
import math
from collections import deque

import cv2
import numpy as np
import mss
import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO


# =========================================================================
# DPI awareness on Windows (the coordinates of the area selection and the
# overlay match real screen pixels at any scaling - 125%/150% etc.)
# =========================================================================
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


# =========================================================================
# DISTANCE SETTINGS
# =========================================================================
# The formula is simplified to ONE calibration constant:
#     distance_cm = DISTANCE_CONSTANT / box_height_in_pixels
#
# The default value is tuned for a typical "person sitting in front of a
# laptop camera" scenario (close up, head/shoulders visible, not full
# height). Previously the formula calculated distance as for a person 170 cm
# tall, which greatly overestimated the result at close range (when only the
# head and shoulders are in the frame). Be sure to calibrate it for your own
# camera (instructions at the very bottom of this file).
DISTANCE_CONSTANT = 40000  # cm * pixels
# Be sure to calibrate for your camera (see the instructions at the end of the file).

CONFIDENCE_THRESHOLD = 0.5

# Lamp brightness: up to and including FULL_BRIGHTNESS_UP_TO_M the lamp is
# at 100%. Beyond that, up to MIN_BRIGHTNESS_FROM_M, the brightness decreases
# smoothly - but NOT to zero, only down to MIN_BRIGHTNESS_PERCENT (the lamp
# never goes out completely). The decrease follows a "soft" curve (not a
# straight line), so that at medium distances the power is slightly higher
# than with a linear decline.
FULL_BRIGHTNESS_UP_TO_M = 1.5     # up to 1.5 m - always 100%
MIN_BRIGHTNESS_FROM_M = 4.0       # from 4 m - minimum power
MIN_BRIGHTNESS_PERCENT = 20       # the minimum the lamp never goes below -
                                   # even when there are no people at all

# The real lamp power changes SMOOTHLY (exponential smoothing) instead of
# jumping instantly with every jitter of the distance - this way the lamp
# does not flicker during small movements. The number is the approximate
# "settling" time in seconds.
LAMP_SMOOTHING_SECONDS = 2.0

CAMERA_INDEX = 0

# ---- Person tracking when detection confidence drops ----
# While YOLO confidently sees a person, we also run a lightweight tracker
# (it follows the movement of a blob in the image, it doesn't "understand"
# that it is a person). When the person moves farther away and YOLO is no
# longer sure it is a person, the tracker keeps following the same object
# for another TRACK_HOLD_SECONDS seconds (the box becomes dashed -
# "tracking"), until either a new confident detection arrives at the same
# place, or the tracker loses the object, or the time runs out.
TRACK_HOLD_SECONDS = 4.0
IOU_MATCH_THRESHOLD = 0.3  # how much the boxes must overlap to be considered the same person

# Lamp power at 100% (W) - used to calculate the saved energy. A typical
# park LED light is roughly 30-80 W (old sodium lamps - 100-250 W). Set the
# real value of your lamp.
LAMP_POWER_WATTS = 50.0

# Box colors - a single list in HEX (used for both Tkinter and OpenCV)
BOX_COLORS_HEX = [
    "#00FF00", "#FF8000", "#00C8FF", "#FF00FF",
    "#FFA500", "#FFFF00", "#8000FF", "#00FF80",
]


def format_duration(total_seconds: float) -> str:
    total_seconds = int(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def iou(box_a: tuple, box_b: tuple) -> float:
    """Intersection / union of two boxes (x1,y1,x2,y2) - how much they overlap."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter_area / float(area_a + area_b - inter_area)


def create_tracker():
    """Creates a tracker object (follows the movement of a blob in the image).
    Different OpenCV versions keep trackers in different places - we try them
    one by one. Requires the opencv-contrib-python package (not the regular
    opencv-python)."""
    candidates = []
    if hasattr(cv2, "legacy"):
        candidates += [
            getattr(cv2.legacy, "TrackerKCF_create", None),
            getattr(cv2.legacy, "TrackerCSRT_create", None),
            getattr(cv2.legacy, "TrackerMOSSE_create", None),
        ]
    candidates += [
        getattr(cv2, "TrackerKCF_create", None),
        getattr(cv2, "TrackerCSRT_create", None),
    ]
    for factory in candidates:
        if factory is None:
            continue
        try:
            return factory()
        except Exception:
            continue
    return None


def draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash_len=10):
    """Dashed box - for people we are currently TRACKING by movement, rather
    than re-detecting with confidence (unlike the solid box)."""
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, dash_len * 2):
        x_end = min(x + dash_len, x2)
        cv2.line(img, (x, y1), (x_end, y1), color, thickness)
        cv2.line(img, (x, y2), (x_end, y2), color, thickness)
    for y in range(y1, y2, dash_len * 2):
        y_end = min(y + dash_len, y2)
        cv2.line(img, (x1, y), (x1, y_end), color, thickness)
        cv2.line(img, (x2, y), (x2, y_end), color, thickness)


def estimate_distance_m(bbox_height_px: float) -> float:
    if bbox_height_px <= 0:
        return float("inf")
    return (DISTANCE_CONSTANT / bbox_height_px) / 100.0


def distance_to_brightness_percent(distance_m: float) -> int:
    if distance_m <= FULL_BRIGHTNESS_UP_TO_M:
        return 100
    if distance_m >= MIN_BRIGHTNESS_FROM_M:
        return MIN_BRIGHTNESS_PERCENT
    # 1.0 at the near boundary, 0.0 at the far boundary
    ratio = (MIN_BRIGHTNESS_FROM_M - distance_m) / (MIN_BRIGHTNESS_FROM_M - FULL_BRIGHTNESS_UP_TO_M)
    ratio = ratio ** 0.5  # soft curve - holds higher power for longer
    brightness = MIN_BRIGHTNESS_PERCENT + (100 - MIN_BRIGHTNESS_PERCENT) * ratio
    return int(round(brightness))


print("Loading YOLOv8 model...")
MODEL = YOLO("yolov8n.pt")
print("Model loaded.")


# =========================================================================
# SCREEN AREA SELECTION (like the Windows Snipping Tool)
# =========================================================================
class RegionSelector:
    def __init__(self, root):
        self.result = None

        self.win = tk.Toplevel(root)
        self.win.attributes("-topmost", True)

        screen_w = self.win.winfo_screenwidth()
        screen_h = self.win.winfo_screenheight()
        self.win.overrideredirect(True)
        self.win.geometry(f"{screen_w}x{screen_h}+0+0")
        self.win.attributes("-alpha", 0.35)
        self.win.configure(bg="black")
        self.win.config(cursor="cross")

        self.canvas = tk.Canvas(self.win, bg="gray20", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(
            screen_w // 2, 40,
            text="Select a screen area to analyze (with the mouse). Esc - cancel.",
            fill="white", font=("Segoe UI", 16))

        self.start_x = 0
        self.start_y = 0
        self.rect = None

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.win.bind("<Escape>", self._on_cancel)

        self.win.update_idletasks()
        self.win.lift()
        self.win.focus_force()
        self.win.grab_set()
        self.win.wait_window()

    def _on_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="#00FF00", width=3)

    def _on_drag(self, event):
        if self.rect is not None:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def _on_release(self, event):
        x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
        x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)

        canvas_root_x = self.canvas.winfo_rootx()
        canvas_root_y = self.canvas.winfo_rooty()

        left = canvas_root_x + x1
        top = canvas_root_y + y1
        width = x2 - x1
        height = y2 - y1

        if width > 20 and height > 20:
            self.result = {"left": int(left), "top": int(top),
                            "width": int(width), "height": int(height)}
        try:
            self.win.grab_release()
        except tk.TclError:
            pass
        self.win.destroy()

    def _on_cancel(self, event=None):
        self.result = None
        try:
            self.win.grab_release()
        except tk.TclError:
            pass
        self.win.destroy()


# =========================================================================
# TRANSPARENT OVERLAY ON TOP OF THE SELECTED SCREEN AREA
# =========================================================================
class ScreenOverlay:
    TRANSPARENT_KEY = "#123456"  # "magic" color - anything drawn with it is invisible

    def __init__(self, root, region: dict):
        self.region = region
        self.top = tk.Toplevel(root)
        self.top.overrideredirect(True)
        self.top.attributes("-topmost", True)
        self.top.config(bg=self.TRANSPARENT_KEY)
        try:
            self.top.attributes("-transparentcolor", self.TRANSPARENT_KEY)
        except tk.TclError:
            pass  # not Windows - transparency may not work, but the window will still open

        self.top.geometry(f"{region['width']}x{region['height']}+{region['left']}+{region['top']}")

        self.canvas = tk.Canvas(self.top, width=region["width"], height=region["height"],
                                 bg=self.TRANSPARENT_KEY, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

    def update_boxes(self, boxes: list, person_count: int,
                      nearest_distance_m: float | None = None, lamp_power: int | None = None,
                      elapsed_seconds: float | None = None, energy_saved_wh: float | None = None):
        self.canvas.delete("all")

        # People counter + nearest person / lamp / timer data - in the top left corner
        self.canvas.create_rectangle(4, 4, 300, 74, fill="#000000", outline="")
        count_text = f"People: {person_count}" if person_count else "No person detected"
        self.canvas.create_text(10, 17, anchor="w", text=count_text, fill="#FFFFFF",
                                 font=("Segoe UI", 11, "bold"))

        if lamp_power is not None:
            if nearest_distance_m is not None:
                lamp_text = f"Nearest: {nearest_distance_m:.2f} m | Lamp: {lamp_power}%"
            else:
                lamp_text = f"Lamp (min.): {lamp_power}%"
            self.canvas.create_text(10, 39, anchor="w", text=lamp_text, fill="#00E0FF",
                                     font=("Segoe UI", 10, "bold"))

        if elapsed_seconds is not None:
            time_text = (f"Time: {format_duration(elapsed_seconds)} | "
                         f"Saved: {energy_saved_wh:.2f} Wh")
            self.canvas.create_text(10, 61, anchor="w", text=time_text, fill="#A0FFA0",
                                     font=("Segoe UI", 10, "bold"))

        for b in boxes:
            x1, y1, x2, y2, color, label1, label2 = (
                b["x1"], b["y1"], b["x2"], b["y2"], b["color"], b["label1"], b["label2"])
            dash = (6, 4) if b.get("tracked") else None

            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3, dash=dash)

            text_y = max(y1 - 30, 12)
            # dark backing under the text for readability on any background
            self.canvas.create_rectangle(x1, text_y - 12, x1 + 180, text_y + 18,
                                          fill="#000000", outline="")
            self.canvas.create_text(x1 + 5, text_y - 4, anchor="w", text=label1,
                                     fill=color, font=("Segoe UI", 9, "bold"))
            self.canvas.create_text(x1 + 5, text_y + 10, anchor="w", text=label2,
                                     fill=color, font=("Segoe UI", 9, "bold"))

    def destroy(self):
        try:
            self.top.destroy()
        except tk.TclError:
            pass


# =========================================================================
# LAMP POWER GRAPH (X - time, Y - power 0-100%)
# =========================================================================
class LampGraph:
    WINDOW_SECONDS = 30  # how many of the most recent seconds to show on the graph

    def __init__(self, root):
        self.top = tk.Toplevel(root)
        self.top.title("Lamp power")
        self.top.attributes("-topmost", True)
        self.canvas_w, self.canvas_h = 420, 180
        self.top.geometry(f"{self.canvas_w + 20}x{self.canvas_h + 55}")

        self.canvas = tk.Canvas(self.top, width=self.canvas_w, height=self.canvas_h, bg="white")
        self.canvas.pack(padx=10, pady=(10, 5))

        self.status_label = tk.Label(self.top, text="Time: 0:00:00  |  Saved: 0.00 Wh",
                                      font=("Segoe UI", 10, "bold"), fg="#1a7a1a")
        self.status_label.pack(pady=(0, 8))

        self.history = deque()  # list of (timestamp, value_0_100)
        self._redraw()

    def update_status(self, elapsed_seconds: float, energy_saved_wh: float):
        self.status_label.config(
            text=f"Time: {format_duration(elapsed_seconds)}  |  Saved: {energy_saved_wh:.2f} Wh")

    def add_point(self, value: float):
        now = time.time()
        self.history.append((now, value))
        cutoff = now - self.WINDOW_SECONDS
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()
        self._redraw()

    def reset(self):
        self.history.clear()
        self._redraw()

    def _redraw(self):
        self.canvas.delete("all")
        w, h = self.canvas_w, self.canvas_h
        margin_left, margin_bottom, margin_top = 32, 20, 10
        plot_w = w - margin_left - 10
        plot_h = h - margin_bottom - margin_top

        # axes and grid by percent
        self.canvas.create_line(margin_left, margin_top, margin_left, h - margin_bottom, fill="#999999")
        self.canvas.create_line(margin_left, h - margin_bottom, w - 10, h - margin_bottom, fill="#999999")
        for pct in (0, 20, 40, 60, 80, 100):
            y = (h - margin_bottom) - (pct / 100) * plot_h
            self.canvas.create_line(margin_left, y, w - 10, y, fill="#eeeeee")
            self.canvas.create_text(margin_left - 5, y, text=str(pct), anchor="e", font=("Segoe UI", 7))

        if len(self.history) >= 2:
            now = self.history[-1][0]
            start_t = now - self.WINDOW_SECONDS
            points = []
            for t, v in self.history:
                x = margin_left + max(0.0, (t - start_t) / self.WINDOW_SECONDS) * plot_w
                y = (h - margin_bottom) - (v / 100) * plot_h
                points.extend([x, y])
            self.canvas.create_line(*points, fill="#0080FF", width=2, smooth=True)

        if self.history:
            last_val = self.history[-1][1]
            self.canvas.create_text(w - 10, margin_top + 5, anchor="e",
                                     text=f"{last_val:.0f}%", font=("Segoe UI", 13, "bold"),
                                     fill="#0080FF")
        else:
            self.canvas.create_text(w / 2, h / 2, text="No data", fill="#aaaaaa",
                                     font=("Segoe UI", 10))

    def destroy(self):
        try:
            self.top.destroy()
        except tk.TclError:
            pass


# =========================================================================
# FRAME CAPTURE AND PROCESSING LOGIC
# =========================================================================
class DetectionSession:
    def __init__(self, mode: str, region: dict | None = None):
        self.mode = mode
        self.region = region
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self._release = None
        self.window_name = "Camera - people and distance"

        # For "screen" mode: the latest results are written here, and the
        # main thread (Tkinter) reads them and draws them in the overlay.
        self._lock = threading.Lock()
        self._latest_boxes = []
        self._latest_person_count = 0

        # Tracking: active person "tracks" (see TRACK_HOLD_SECONDS) - they keep
        # following a moving object even when YOLO is no longer sure it is a
        # person.
        self._tracks = []
        self._next_track_id = 1

        # Lamp power: smoothed value + distance to the nearest person -
        # shared by both modes (camera and screen).
        self._lamp_power = float(MIN_BRIGHTNESS_PERCENT)
        self._lamp_power_public = MIN_BRIGHTNESS_PERCENT
        self._nearest_distance_m = None
        self._last_lamp_update_time = None

        # Run timer + accumulated saved energy (compared to a lamp that is
        # always at 100%). Accumulates while the session is active.
        self._session_start_time = None
        self._energy_saved_wh = 0.0

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def get_latest_boxes(self):
        with self._lock:
            return list(self._latest_boxes), self._latest_person_count

    def get_lamp_status(self):
        with self._lock:
            return self._nearest_distance_m, self._lamp_power_public

    def get_energy_status(self):
        with self._lock:
            elapsed = time.time() - self._session_start_time if self._session_start_time else 0.0
            return elapsed, self._energy_saved_wh

    def _frame_source(self):
        if self.mode == "camera":
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if not cap.isOpened():
                return None

            def read():
                return cap.read()

            self._release = cap.release
            return read

        else:  # "screen"
            sct = mss.mss()
            capture_area = self.region if self.region else sct.monitors[1]

            if capture_area.get("width", 0) <= 0 or capture_area.get("height", 0) <= 0:
                print("Invalid capture area, canceling start.")
                sct.close()
                return None

            def read():
                img = np.array(sct.grab(capture_area))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return True, frame

            self._release = sct.close
            return read

    def _detect_people(self, frame):
        """Returns a list of people with box coordinates, distance, brightness."""
        results = MODEL(frame, verbose=False)[0]
        people = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id != 0 or conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox_height_px = y2 - y1
            distance_m = estimate_distance_m(bbox_height_px)
            brightness = distance_to_brightness_percent(distance_m)
            people.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": conf, "distance_m": distance_m, "brightness": brightness,
            })
        return people

    def _update_tracks(self, frame, detections):
        """Matches new confident detections with already tracked people (by
        box overlap), updates their trackers, and for those YOLO did not
        confidently see in this frame, tries to continue following them by
        motion (motion tracking) for another TRACK_HOLD_SECONDS seconds."""
        now = time.time()
        matched_ids = set()

        for det in detections:
            det_box = (det["x1"], det["y1"], det["x2"], det["y2"])
            best_iou = 0.0
            best_track = None
            for track in self._tracks:
                if track["id"] in matched_ids:
                    continue
                score = iou(det_box, track["box"])
                if score > best_iou:
                    best_iou = score
                    best_track = track

            box_xywh = (det["x1"], det["y1"], det["x2"] - det["x1"], det["y2"] - det["y1"])
            tracker = create_tracker()
            if tracker is not None:
                try:
                    tracker.init(frame, box_xywh)
                except Exception:
                    tracker = None

            if best_track is not None and best_iou >= IOU_MATCH_THRESHOLD:
                best_track.update({
                    "box": det_box, "conf": det["conf"], "tracked": False,
                    "last_confident_time": now, "tracker": tracker,
                })
                matched_ids.add(best_track["id"])
            else:
                new_track = {
                    "id": self._next_track_id, "box": det_box, "conf": det["conf"],
                    "tracked": False, "last_confident_time": now, "tracker": tracker,
                }
                self._next_track_id += 1
                self._tracks.append(new_track)
                matched_ids.add(new_track["id"])

        still_alive = []
        for track in self._tracks:
            if track["id"] in matched_ids:
                still_alive.append(track)
                continue

            if now - track["last_confident_time"] > TRACK_HOLD_SECONDS:
                continue  # too long without confirmation - forget it

            if track["tracker"] is None:
                continue  # nothing to keep following with (no opencv-contrib-python)

            ok, box_xywh = track["tracker"].update(frame)
            if not ok:
                continue  # the tracker lost the object

            x, y, w, h = box_xywh
            track["box"] = (int(x), int(y), int(x + w), int(y + h))
            track["conf"] = None
            track["tracked"] = True
            still_alive.append(track)

        self._tracks = still_alive

        people = []
        for track in self._tracks:
            x1, y1, x2, y2 = track["box"]
            bbox_height_px = y2 - y1
            distance_m = estimate_distance_m(bbox_height_px)
            brightness = distance_to_brightness_percent(distance_m)
            people.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": track["conf"], "distance_m": distance_m, "brightness": brightness,
                "track_id": track["id"], "tracked": track["tracked"],
            })
        return people

    def _run(self):
        window_created = False
        try:
            read_frame = self._frame_source()
            if read_frame is None:
                print("Failed to start the video source.")
                return

            window_positioned = False
            window_ready = False
            is_fullscreen = False
            self._session_start_time = time.time()

            while not self.stop_event.is_set():
                ok, frame = read_frame()
                if not ok or frame is None or frame.size == 0:
                    print("Empty frame - video source unavailable, stopping the session.")
                    break

                detections = self._detect_people(frame)
                people = self._update_tracks(frame, detections)

                # ---- Distance to the nearest person + smoothed lamp power ----
                # (shared by both modes - camera and screen)
                nearest_distance_m = min((p["distance_m"] for p in people), default=None)
                target_power = (distance_to_brightness_percent(nearest_distance_m)
                                 if nearest_distance_m is not None else MIN_BRIGHTNESS_PERCENT)

                now_t = time.time()
                if self._last_lamp_update_time is None:
                    dt = 0.0
                else:
                    dt = max(now_t - self._last_lamp_update_time, 0.0)
                self._last_lamp_update_time = now_t

                alpha = min(dt / LAMP_SMOOTHING_SECONDS, 1.0) if LAMP_SMOOTHING_SECONDS > 0 else 1.0
                self._lamp_power += (target_power - self._lamp_power) * alpha
                lamp_power_int = int(round(self._lamp_power))

                # Saved energy: the difference between "lamp always at 100%"
                # and what it actually consumes now, accumulated over time.
                actual_watts = lamp_power_int / 100.0 * LAMP_POWER_WATTS
                saved_watts = LAMP_POWER_WATTS - actual_watts
                self._energy_saved_wh += saved_watts * (dt / 3600.0)

                with self._lock:
                    self._nearest_distance_m = nearest_distance_m
                    self._lamp_power_public = lamp_power_int

                if self.mode == "camera":
                    if not window_ready:
                        # WINDOW_NORMAL + FREERATIO - the window can be resized
                        # with the mouse, and the picture stretches to fill the
                        # whole window without gray bars at the sides/top/bottom.
                        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
                        start_w = min(frame.shape[1], 1000)
                        start_h = min(frame.shape[0], 700)
                        cv2.resizeWindow(self.window_name, start_w, start_h)
                        window_ready = True
                        window_created = True

                    for p in people:
                        color_hex = BOX_COLORS_HEX[p["track_id"] % len(BOX_COLORS_HEX)]
                        color_bgr = hex_to_bgr(color_hex)
                        pt1, pt2 = (p["x1"], p["y1"]), (p["x2"], p["y2"])
                        if p["tracked"]:
                            draw_dashed_rect(frame, pt1, pt2, color_bgr, thickness=2)
                            label1 = f"Person #{p['track_id']} (tracking)"
                        else:
                            cv2.rectangle(frame, pt1, pt2, color_bgr, 2)
                            label1 = f"Person #{p['track_id']} {p['conf']:.0%}"
                        label2 = f"~{p['distance_m']:.2f} m | lamp {p['brightness']}%"
                        cv2.putText(frame, label1, (p["x1"], max(p["y1"] - 28, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2)
                        cv2.putText(frame, label2, (p["x1"], max(p["y1"] - 8, 40)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2)

                    status = f"People detected: {len(people)}" if people else "No person detected"
                    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 200, 0) if people else (0, 0, 255), 2)

                    if nearest_distance_m is not None:
                        lamp_status = f"Nearest: {nearest_distance_m:.2f} m | Lamp: {lamp_power_int}%"
                    else:
                        lamp_status = f"Lamp (minimum): {lamp_power_int}%"
                    cv2.putText(frame, lamp_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 224, 255), 2)

                    elapsed_str = format_duration(time.time() - self._session_start_time)
                    energy_status = f"Run time: {elapsed_str} | Saved: {self._energy_saved_wh:.2f} Wh"
                    cv2.putText(frame, energy_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (200, 255, 200), 2)

                    # Stretch the frame to the CURRENT window size (including
                    # full screen via the F key) - without this, gray bars
                    # remained when the window was enlarged, because the
                    # picture kept its original proportions.
                    frame_to_show = frame
                    try:
                        _, _, win_w, win_h = cv2.getWindowImageRect(self.window_name)
                        if win_w > 0 and win_h > 0:
                            frame_to_show = cv2.resize(frame, (win_w, win_h))
                    except cv2.error:
                        pass

                    cv2.imshow(self.window_name, frame_to_show)

                    if not window_positioned:
                        try:
                            cv2.moveWindow(self.window_name, 50, 50)
                        except cv2.error:
                            pass
                        window_positioned = True

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self.stop_event.set()
                        break
                    if key == ord("f"):
                        is_fullscreen = not is_fullscreen
                        prop_value = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                        try:
                            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, prop_value)
                        except cv2.error:
                            pass

                else:  # "screen" - publish the result for the overlay, do NOT show the picture
                    boxes_for_overlay = []
                    for p in people:
                        color_hex = BOX_COLORS_HEX[p["track_id"] % len(BOX_COLORS_HEX)]
                        if p["tracked"]:
                            label1 = f"Person #{p['track_id']} (tracking)"
                        else:
                            label1 = f"Person #{p['track_id']} {p['conf']:.0%}"
                        boxes_for_overlay.append({
                            "x1": p["x1"], "y1": p["y1"], "x2": p["x2"], "y2": p["y2"],
                            "color": color_hex, "tracked": p["tracked"],
                            "label1": label1,
                            "label2": f"~{p['distance_m']:.2f} m | lamp {p['brightness']}%",
                        })
                    with self._lock:
                        self._latest_boxes = boxes_for_overlay
                        self._latest_person_count = len(people)

                    time.sleep(0.05)  # limit the screen capture rate

        except Exception:
            print("Error in the video processing thread:")
            traceback.print_exc()

        finally:
            if self._release:
                try:
                    self._release()
                except Exception:
                    pass
            if window_created:
                try:
                    cv2.destroyWindow(self.window_name)
                    for _ in range(3):
                        cv2.waitKey(1)
                except cv2.error:
                    pass


# =========================================================================
# MAIN WINDOW
# =========================================================================
class App:
    OVERLAY_UPDATE_MS = 100

    def __init__(self, root):
        self.root = root
        self.root.title("People Detection - Camera / Screen")
        self.root.geometry("400x240")

        self.current_session: DetectionSession | None = None
        self.overlay: ScreenOverlay | None = None
        self.lamp_graph: LampGraph | None = None

        title = ttk.Label(root, text="Choose a video source", font=("Segoe UI", 13, "bold"))
        title.pack(pady=15)

        ttk.Button(root, text="📷 Turn on camera",
                   command=lambda: self.start_camera()).pack(pady=8, ipadx=10, ipady=6)

        ttk.Button(root, text="🖥 Screen sharing",
                   command=lambda: self.start_screen()).pack(pady=8, ipadx=10, ipady=6)

        ttk.Button(root, text="⏹ Stop", command=self.stop_session).pack(pady=8, ipadx=10, ipady=6)

        self.status_label = ttk.Label(root, text="Nothing running", foreground="gray")
        self.status_label.pack(pady=10)

        hint = ttk.Label(root, text="Camera window: Q - quit, F - full screen.\n"
                                     "In screen mode the boxes are drawn right\n"
                                     "on top of the selected area.",
                          font=("Segoe UI", 9), foreground="gray", justify="center")
        hint.pack(pady=5)

        self._poll_overlay()

    def _stop_and_wait(self):
        if self.current_session is not None:
            self.current_session.stop()
            self.current_session.thread.join(timeout=3.0)
            self.current_session = None
            self.status_label.config(text="Stopped", foreground="gray")
            self.root.update()
            time.sleep(0.3)

        if self.overlay is not None:
            self.overlay.destroy()
            self.overlay = None

        if self.lamp_graph is not None:
            self.lamp_graph.destroy()
            self.lamp_graph = None

    def start_camera(self):
        self._stop_and_wait()
        self.current_session = DetectionSession("camera")
        self.current_session.start()
        self.lamp_graph = LampGraph(self.root)
        self.status_label.config(text="Camera running", foreground="green")

    def start_screen(self):
        self._stop_and_wait()

        selector = RegionSelector(self.root)
        region = selector.result

        if region is None:
            self.status_label.config(text="Area selection canceled", foreground="gray")
            return

        self.overlay = ScreenOverlay(self.root, region)
        self.current_session = DetectionSession("screen", region=region)
        self.current_session.start()
        self.lamp_graph = LampGraph(self.root)
        self.status_label.config(text="Screen sharing running", foreground="green")

    def stop_session(self):
        self._stop_and_wait()

    def _poll_overlay(self):
        if self.current_session is not None:
            nearest_distance_m, lamp_power = self.current_session.get_lamp_status()
            elapsed_seconds, energy_saved_wh = self.current_session.get_energy_status()

            if self.current_session.mode == "screen" and self.overlay:
                boxes, count = self.current_session.get_latest_boxes()
                self.overlay.update_boxes(boxes, count, nearest_distance_m, lamp_power,
                                           elapsed_seconds, energy_saved_wh)

            if self.lamp_graph is not None:
                self.lamp_graph.add_point(lamp_power)
                self.lamp_graph.update_status(elapsed_seconds, energy_saved_wh)

        self.root.after(self.OVERLAY_UPDATE_MS, self._poll_overlay)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
