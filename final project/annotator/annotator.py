"""
Dendrite Skeleton Annotation Tool
----------------------------------
Draw skeleton centerlines on SEM images. The tool auto-expands each stroke
into a polygon by detecting the dendrite boundary perpendicular to the stroke.

Controls:
  Left-click + drag  : draw a stroke
  Right-click        : undo last stroke
  Middle-click + drag: pan
  Scroll wheel       : zoom in/out
  W / S              : increase / decrease probe distance
  R                  : reset all strokes on current image
  Enter / Space      : save and go to next image
  Q                  : skip image without saving
"""

import cv2
import numpy as np
import os
import json
import glob
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

# ─── Config ───────────────────────────────────────────────────────────────────

YOLO_CLASS_ID = 0
DEFAULT_HALF_WIDTH = 1    # half-width of the fixed buffer in pixels (W/S to adjust)
MIN_HALF_WIDTH = 1
MAX_HALF_WIDTH = 150
STROKE_SMOOTH_SIGMA = 2.0
MIN_STROKE_POINTS = 4
POLYGON_SIMPLIFY_EPS = 1.5

IMAGE_EXTS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp")


def images_from_dir(d):
    imgs = []
    for ext in IMAGE_EXTS:
        imgs.extend(glob.glob(os.path.join(d, ext)))
    return sorted(imgs)


# ─── Launch GUI ───────────────────────────────────────────────────────────────

def show_launcher():
    """
    Small tkinter window to pick:
      - Input folder (or individual files)
      - Output folder
    Returns (image_paths, output_dir) or (None, None) if cancelled.
    """
    result = {"images": None, "output": None}

    root = tk.Tk()
    root.title("Skeleton Annotator — Setup")
    root.resizable(False, False)

    pad = dict(padx=10, pady=4)

    # ── Input folder row ──────────────────────────────────────────────────────
    tk.Label(root, text="Image folder:", anchor="w").grid(row=0, column=0, sticky="w", **pad)
    input_var = tk.StringVar()
    input_entry = tk.Entry(root, textvariable=input_var, width=50)
    input_entry.grid(row=0, column=1, **pad)

    def browse_folder():
        d = filedialog.askdirectory(title="Select image folder")
        if d:
            input_var.set(d)
            # Auto-fill output next to input folder if empty
            if not output_var.get():
                output_var.set(os.path.join(d, "annotations"))

    def browse_files():
        paths = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[("Images", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"), ("All", "*.*")]
        )
        if paths:
            input_var.set(";".join(paths))
            if not output_var.get():
                output_var.set(os.path.join(os.path.dirname(paths[0]), "annotations"))

    btn_frame = tk.Frame(root)
    btn_frame.grid(row=0, column=2, padx=4)
    tk.Button(btn_frame, text="Folder", command=browse_folder, width=7).pack(side="left", padx=2)
    tk.Button(btn_frame, text="Files",  command=browse_files,  width=7).pack(side="left", padx=2)

    # ── Output folder row ─────────────────────────────────────────────────────
    tk.Label(root, text="Output folder:", anchor="w").grid(row=1, column=0, sticky="w", **pad)
    output_var = tk.StringVar()
    tk.Entry(root, textvariable=output_var, width=50).grid(row=1, column=1, **pad)

    def browse_output():
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            output_var.set(d)

    tk.Button(root, text="Browse", command=browse_output, width=7).grid(row=1, column=2, padx=4)

    # ── Include subfolders checkbox ───────────────────────────────────────────
    recurse_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Include subfolders", variable=recurse_var).grid(
        row=2, column=1, sticky="w", padx=10)

    # ── Status label ─────────────────────────────────────────────────────────
    status_var = tk.StringVar()
    tk.Label(root, textvariable=status_var, fg="red").grid(row=3, column=0, columnspan=3, **pad)

    # ── Buttons ───────────────────────────────────────────────────────────────
    def on_start():
        inp = input_var.get().strip()
        out = output_var.get().strip()

        if not inp:
            status_var.set("Please select an image folder or files.")
            return
        if not out:
            status_var.set("Please select an output folder.")
            return

        # Collect images
        if ";" in inp:
            # Individual files chosen
            images = [p for p in inp.split(";") if os.path.isfile(p)]
        else:
            if not os.path.isdir(inp):
                status_var.set("Input folder not found.")
                return
            if recurse_var.get():
                images = []
                for root_dir, _, _ in os.walk(inp):
                    images.extend(images_from_dir(root_dir))
                images = sorted(set(images))
            else:
                images = images_from_dir(inp)

        if not images:
            status_var.set("No images found in the selected location.")
            return

        result["images"] = images
        result["output"] = out
        root.destroy()

    def on_cancel():
        root.destroy()

    btn_row = tk.Frame(root)
    btn_row.grid(row=4, column=0, columnspan=3, pady=10)
    tk.Button(btn_row, text="Start Annotating", command=on_start,
              bg="#4CAF50", fg="white", width=18, font=("", 10, "bold")).pack(side="left", padx=8)
    tk.Button(btn_row, text="Cancel", command=on_cancel, width=10).pack(side="left", padx=8)

    root.mainloop()
    return result["images"], result["output"]


# ─── Image loading ────────────────────────────────────────────────────────────

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read {path}")
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


# ─── Stroke → fixed-width polygon ────────────────────────────────────────────

def stroke_to_polygon(stroke_pts, half_width):
    """Buffer a polyline by half_width pixels on each side → closed polygon."""
    pts = np.array(stroke_pts, dtype=float)

    if len(pts) >= 4:
        pts[:, 0] = gaussian_filter1d(pts[:, 0], STROKE_SMOOTH_SIGMA)
        pts[:, 1] = gaussian_filter1d(pts[:, 1], STROKE_SMOOTH_SIGMA)

    step = max(1, len(pts) // 300)
    pts = pts[::step]
    if len(pts) < 2:
        return None

    left_pts, right_pts = [], []

    for i, (cx, cy) in enumerate(pts):
        if i == 0:
            tx, ty = pts[1] - pts[0]
        elif i == len(pts) - 1:
            tx, ty = pts[-1] - pts[-2]
        else:
            tx, ty = pts[i + 1] - pts[i - 1]

        norm = np.sqrt(tx**2 + ty**2)
        if norm < 1e-6:
            continue
        tx, ty = tx / norm, ty / norm
        # perpendicular
        px, py = -ty, tx

        left_pts.append((cx + px * half_width, cy + py * half_width))
        right_pts.append((cx - px * half_width, cy - py * half_width))

    if len(left_pts) < 2:
        return None

    polygon = np.array(left_pts + list(reversed(right_pts)), dtype=np.float32)
    polygon = cv2.approxPolyDP(polygon.reshape(-1, 1, 2), POLYGON_SIMPLIFY_EPS, closed=True)
    return polygon.reshape(-1, 2).astype(np.int32)


# ─── YOLO / visual output ─────────────────────────────────────────────────────

def save_yolo(image_path, polygons, img_shape, output_dir):
    h, w = img_shape[:2]
    stem = Path(image_path).stem
    out_txt = os.path.join(output_dir, stem + ".txt")
    lines = []
    for poly in polygons:
        if poly is None or len(poly) < 3:
            continue
        coords = []
        for x, y in poly:
            coords.append(f"{np.clip(x/w, 0, 1):.6f}")
            coords.append(f"{np.clip(y/h, 0, 1):.6f}")
        lines.append(f"{YOLO_CLASS_ID} " + " ".join(coords))
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))
    return out_txt


def save_visual(image_path, img_bgr, polygons, strokes, output_dir):
    stem = Path(image_path).stem
    out_png = os.path.join(output_dir, stem + "_annotated.png")
    vis = img_bgr.copy()
    overlay = vis.copy()
    for poly in polygons:
        if poly is not None and len(poly) >= 3:
            cv2.fillPoly(overlay, [poly], (0, 255, 100))
    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
    for poly in polygons:
        if poly is not None and len(poly) >= 3:
            cv2.polylines(vis, [poly], True, (0, 255, 0), 2)
    for stroke in strokes:
        for i in range(1, len(stroke)):
            cv2.line(vis,
                     (int(stroke[i-1][0]), int(stroke[i-1][1])),
                     (int(stroke[i][0]),   int(stroke[i][1])),
                     (0, 0, 255), 1)
    cv2.imwrite(out_png, vis)
    return out_png


# ─── Annotator (with zoom + pan) ──────────────────────────────────────────────

class Annotator:
    def __init__(self):
        self.strokes = []
        self.polygons = []
        self.current_stroke = []
        self.drawing = False
        self.half_width = DEFAULT_HALF_WIDTH

        # View state
        self.zoom = 1.0
        self.pan_x = 0      # offset in image pixels
        self.pan_y = 0
        self.panning = False
        self.pan_start_mouse = (0, 0)
        self.pan_start_offset = (0, 0)

        self.gray = None
        self.img_orig = None
        self.img_display = None
        self.win_w = 1200
        self.win_h = 900

    def reset(self):
        self.strokes = []
        self.polygons = []
        self.current_stroke = []
        self.drawing = False
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0

    # ── coordinate transforms ──────────────────────────────────────────────────

    def screen_to_image(self, sx, sy):
        """Convert screen (window) pixel → image pixel."""
        ix = (sx - self.win_w / 2) / self.zoom + self.pan_x + self.img_orig.shape[1] / 2
        iy = (sy - self.win_h / 2) / self.zoom + self.pan_y + self.img_orig.shape[0] / 2
        return ix, iy

    def image_to_screen(self, ix, iy):
        """Convert image pixel → screen pixel."""
        sx = (ix - self.img_orig.shape[1] / 2 - self.pan_x) * self.zoom + self.win_w / 2
        sy = (iy - self.img_orig.shape[0] / 2 - self.pan_y) * self.zoom + self.win_h / 2
        return sx, sy

    def clamp_pan(self):
        ih, iw = self.img_orig.shape[:2]
        # Limit pan so image doesn't fly off screen completely
        max_px = iw / 2 + self.win_w / (2 * self.zoom) * 0.9
        max_py = ih / 2 + self.win_h / (2 * self.zoom) * 0.9
        self.pan_x = np.clip(self.pan_x, -max_px, max_px)
        self.pan_y = np.clip(self.pan_y, -max_py, max_py)

    # ── rendering ──────────────────────────────────────────────────────────────

    def rebuild_display(self):
        ih, iw = self.img_orig.shape[:2]

        # Compute the region of the image visible in the window
        # (in image coordinates)
        tl_ix, tl_iy = self.screen_to_image(0, 0)
        br_ix, br_iy = self.screen_to_image(self.win_w, self.win_h)

        # Crop with padding
        x0 = max(0, int(tl_ix) - 2)
        y0 = max(0, int(tl_iy) - 2)
        x1 = min(iw, int(br_ix) + 2)
        y1 = min(ih, int(br_iy) + 2)

        if x1 <= x0 or y1 <= y0:
            self.img_display = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)
            return

        crop = self.img_orig[y0:y1, x0:x1].copy()

        # Draw overlays on the crop (converting image coords → crop coords)
        def to_crop(ix, iy):
            return int(round(ix - x0)), int(round(iy - y0))

        # Filled polygon overlay
        overlay = crop.copy()
        for poly in self.polygons:
            if poly is not None and len(poly) >= 3:
                shifted = np.array([[int(px - x0), int(py - y0)] for px, py in poly], dtype=np.int32)
                cv2.fillPoly(overlay, [shifted], (0, 255, 100))
        cv2.addWeighted(overlay, 0.35, crop, 0.65, 0, crop)

        for poly in self.polygons:
            if poly is not None and len(poly) >= 3:
                shifted = np.array([[int(px - x0), int(py - y0)] for px, py in poly], dtype=np.int32)
                cv2.polylines(crop, [shifted], True, (0, 255, 0), max(1, int(1.5 / self.zoom)))

        for stroke in self.strokes:
            for i in range(1, len(stroke)):
                p1 = to_crop(*stroke[i-1])
                p2 = to_crop(*stroke[i])
                cv2.line(crop, p1, p2, (0, 0, 255), max(1, int(1 / self.zoom)))

        for i in range(1, len(self.current_stroke)):
            p1 = to_crop(*self.current_stroke[i-1])
            p2 = to_crop(*self.current_stroke[i])
            cv2.line(crop, p1, p2, (255, 120, 0), max(1, int(1 / self.zoom)))

        # Scale crop to window size
        crop_h, crop_w = crop.shape[:2]
        out_w = int(round(crop_w * self.zoom))
        out_h = int(round(crop_h * self.zoom))

        if out_w <= 0 or out_h <= 0:
            self.img_display = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)
            return

        scaled = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        # Place into canvas
        canvas = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)

        # Where does the top-left of the crop land on screen?
        sl_x, sl_y = self.image_to_screen(x0, y0)
        sl_x, sl_y = int(round(sl_x)), int(round(sl_y))

        # Clip to canvas
        src_x0 = max(0, -sl_x)
        src_y0 = max(0, -sl_y)
        dst_x0 = max(0, sl_x)
        dst_y0 = max(0, sl_y)
        copy_w = min(out_w - src_x0, self.win_w - dst_x0)
        copy_h = min(out_h - src_y0, self.win_h - dst_y0)

        if copy_w > 0 and copy_h > 0:
            canvas[dst_y0:dst_y0+copy_h, dst_x0:dst_x0+copy_w] = \
                scaled[src_y0:src_y0+copy_h, src_x0:src_x0+copy_w]

        # HUD
        hud1 = f"Zoom: {self.zoom:.1f}x | Width: {self.half_width*2}px | Strokes: {len(self.strokes)}"
        hud2 = "W/S: width  R: reset  RClick: undo  Scroll: zoom  Mid-drag: pan  0: fit  Enter: save  Q: skip"
        cv2.putText(canvas, hud1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(canvas, hud2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        self.img_display = canvas

    # ── mouse callback ─────────────────────────────────────────────────────────

    def mouse_cb(self, event, sx, sy, flags, param):
        ix, iy = self.screen_to_image(sx, sy)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_stroke = [(ix, iy)]

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_stroke.append((ix, iy))
                self.rebuild_display()
            elif self.panning:
                dx = (sx - self.pan_start_mouse[0]) / self.zoom
                dy = (sy - self.pan_start_mouse[1]) / self.zoom
                self.pan_x = self.pan_start_offset[0] - dx
                self.pan_y = self.pan_start_offset[1] - dy
                self.clamp_pan()
                self.rebuild_display()

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            if len(self.current_stroke) >= MIN_STROKE_POINTS:
                poly = stroke_to_polygon(self.current_stroke, self.half_width)
                self.strokes.append(self.current_stroke)
                self.polygons.append(poly)
            self.current_stroke = []
            self.rebuild_display()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.strokes:
                self.strokes.pop()
                self.polygons.pop()
                self.rebuild_display()

        elif event == cv2.EVENT_MBUTTONDOWN:
            self.panning = True
            self.pan_start_mouse = (sx, sy)
            self.pan_start_offset = (self.pan_x, self.pan_y)

        elif event == cv2.EVENT_MBUTTONUP:
            self.panning = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            # Zoom centred on cursor
            factor = 1.15 if flags > 0 else 1 / 1.15
            # Keep the image point under cursor fixed
            old_zoom = self.zoom
            self.zoom = np.clip(self.zoom * factor, 0.1, 20.0)
            # Adjust pan so the pixel under cursor stays fixed
            self.pan_x += (ix - self.img_orig.shape[1] / 2) * (1/old_zoom - 1/self.zoom) * old_zoom * 0
            # Simpler: recompute where cursor maps after zoom change
            # The image point ix,iy should still map to screen sx,sy
            # sx = (ix - iw/2 - pan_x)*zoom + win_w/2
            # => pan_x = ix - iw/2 - (sx - win_w/2)/zoom
            self.pan_x = ix - self.img_orig.shape[1] / 2 - (sx - self.win_w / 2) / self.zoom
            self.pan_y = iy - self.img_orig.shape[0] / 2 - (sy - self.win_h / 2) / self.zoom
            self.clamp_pan()
            self.rebuild_display()


# ─── Main ────────────────────────────────────────────────────────────────────

def collect_window_size(win):
    """Try to read the actual window dimensions."""
    try:
        r = cv2.getWindowImageRect(win)
        if r[2] > 0 and r[3] > 0:
            return r[3], r[2]   # h, w
    except Exception:
        pass
    return 900, 1200


def main():
    images, OUTPUT_DIR = show_launcher()
    if not images:
        print("Cancelled.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    done_file = os.path.join(OUTPUT_DIR, "done.json")
    done = set()
    if os.path.exists(done_file):
        with open(done_file) as f:
            done = set(json.load(f))

    ann = Annotator()
    win = "Dendrite Annotator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, ann.win_w, ann.win_h)
    cv2.setMouseCallback(win, ann.mouse_cb)

    for img_path in images:
        stem = Path(img_path).stem
        if stem in done:
            print(f"[skip] {stem} already annotated")
            continue

        print(f"\n[annotating] {img_path}")
        img_bgr = load_image(img_path)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        ann.reset()
        ann.gray = gray
        ann.img_orig = img_bgr.copy()

        # Fit image to window initially
        ih, iw = img_bgr.shape[:2]
        ann.zoom = min(ann.win_w / iw, ann.win_h / ih) * 0.95
        ann.rebuild_display()

        while True:
            # Update window size in case user resized it
            h, w = collect_window_size(win)
            if h != ann.win_h or w != ann.win_w:
                ann.win_h, ann.win_w = h, w
                ann.rebuild_display()

            cv2.imshow(win, ann.img_display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('w'):
                ann.half_width = min(ann.half_width + 1, MAX_HALF_WIDTH)
                ann.polygons = [stroke_to_polygon(s, ann.half_width) for s in ann.strokes]
                ann.rebuild_display()

            elif key == ord('s'):
                ann.half_width = max(ann.half_width - 1, MIN_HALF_WIDTH)
                ann.polygons = [stroke_to_polygon(s, ann.half_width) for s in ann.strokes]
                ann.rebuild_display()

            elif key == ord('r'):
                ann.reset()
                ann.gray = gray
                ann.img_orig = img_bgr.copy()
                ih, iw = img_bgr.shape[:2]
                ann.zoom = min(ann.win_w / iw, ann.win_h / ih) * 0.95
                ann.rebuild_display()

            elif key == ord('0'):
                # Reset zoom/pan only
                ih, iw = img_bgr.shape[:2]
                ann.zoom = min(ann.win_w / iw, ann.win_h / ih) * 0.95
                ann.pan_x = 0
                ann.pan_y = 0
                ann.rebuild_display()

            elif key in (13, 32):  # Enter or Space
                if ann.polygons:
                    txt = save_yolo(img_path, ann.polygons, img_bgr.shape, OUTPUT_DIR)
                    png = save_visual(img_path, img_bgr, ann.polygons, ann.strokes, OUTPUT_DIR)
                    print(f"  Saved YOLO: {txt}")
                    print(f"  Visual:     {png}")
                    done.add(stem)
                    with open(done_file, "w") as f:
                        json.dump(list(done), f)
                else:
                    print("  No strokes — nothing saved. Press Q to skip.")
                break

            elif key == ord('q'):
                print("  Skipped.")
                break

            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("\nAll done. Annotations in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
