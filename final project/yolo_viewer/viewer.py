"""
YOLO Dendrite Segmentation Viewer  (PySide6)

The raw image is always displayed.
Two overlay layers toggled via checkboxes:
  - Ground Truth  : polygons from the .txt label file (YOLO seg format)
  - YOLO Prediction: predicted masks from best.pt inference
"""

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QObject, QTimer
from PySide6.QtGui import QImage, QPixmap, QColor, QPalette
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QCheckBox, QListWidget, QListWidgetItem,
    QFileDialog, QFrame, QSizePolicy, QProgressBar, QStatusBar,
    QSplitter, QGroupBox, QLineEdit, QScrollArea,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR    = Path(__file__).parent.parent
MODEL_PATH    = SCRIPT_DIR / "yolo_training/runs/dendrite_seg/weights/best.pt"
TEST_IMG_DIR  = SCRIPT_DIR / "yolo_training/yolo_dataset/test/images"
TEST_LBL_DIR  = SCRIPT_DIR / "yolo_training/yolo_dataset/test/labels"
IMAGE_EXTS    = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# Colors  (BGR)
# ---------------------------------------------------------------------------
GT_COLOR    = (80,  200, 255)   # sky-blue  — ground truth
GT_FILL_A   = 0.25
PRED_COLORS = [                  # per-instance cycling colors — predictions
    (124,  58, 237),
    ( 34, 197,  94),
    (249, 115,  22),
    (236,  72, 153),
    (234, 179,   8),
    ( 14, 165, 233),
]
PRED_FILL_A = 0.40

# ---------------------------------------------------------------------------
# Dark palette
# ---------------------------------------------------------------------------
DARK_BG      = "#1e1e2e"
PANEL_BG     = "#2a2a3e"
ACCENT       = "#7c3aed"
ACCENT_LIGHT = "#a78bfa"
HOVER        = "#6d28d9"
TEXT         = "#e2e8f0"
TEXT_DIM     = "#94a3b8"
SUCCESS      = "#22c55e"
WARNING      = "#f59e0b"
DANGER       = "#ef4444"
BORDER       = "#3f3f5c"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {DARK_BG};
    color: {TEXT};
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}}
QSplitter::handle {{ background: {BORDER}; width: 2px; }}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 10px;
    padding: 8px;
    font-weight: bold;
    color: {ACCENT_LIGHT};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; }}
QPushButton {{
    background-color: {ACCENT}; color: white; border: none;
    border-radius: 6px; padding: 7px 14px; font-weight: 600;
}}
QPushButton:hover  {{ background-color: {HOVER}; }}
QPushButton:pressed{{ background-color: #5b21b6; }}
QPushButton#secondary {{
    background-color: {PANEL_BG}; color: {TEXT}; border: 1px solid {BORDER};
}}
QPushButton#secondary:hover {{ background-color: {BORDER}; }}
QLineEdit {{
    background-color: {PANEL_BG}; border: 1px solid {BORDER};
    border-radius: 5px; padding: 5px 8px; color: {TEXT};
}}
QListWidget {{
    background-color: {PANEL_BG}; border: 1px solid {BORDER};
    border-radius: 6px; outline: none;
}}
QListWidget::item {{ padding: 6px 8px; border-radius: 4px; }}
QListWidget::item:selected {{ background-color: {ACCENT}; color: white; }}
QListWidget::item:hover:!selected {{ background-color: {BORDER}; }}
QCheckBox {{ spacing: 8px; font-weight: 600; font-size: 13px; }}
QCheckBox::indicator {{
    width: 18px; height: 18px; border-radius: 4px;
    border: 2px solid {BORDER}; background: {PANEL_BG};
}}
QProgressBar {{
    border: none; border-radius: 4px; background: {PANEL_BG}; height: 4px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {ACCENT}, stop:1 {ACCENT_LIGHT});
    border-radius: 4px;
}}
QStatusBar {{
    background: {PANEL_BG}; color: {TEXT_DIM};
    border-top: 1px solid {BORDER}; padding: 2px 8px;
}}
"""

# ---------------------------------------------------------------------------
# Signal bridge (worker → main thread)
# ---------------------------------------------------------------------------
class WorkerSignals(QObject):
    infer_done      = Signal(object, str)   # (pred_masks_list or None, msg)
    model_loaded    = Signal(bool, str)
    composite_ready = Signal(object)        # QImage
    batch_progress  = Signal(int, int, str) # (done, total, current_filename)
    batch_done      = Signal(str)           # output folder path


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def load_gt_polygons(label_path: Path, img_w: int, img_h: int):
    """Parse YOLO-seg .txt and return list of (N,1,2) int32 contour arrays."""
    polys = []
    if not label_path.exists():
        return polys
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            coords = list(map(float, parts[1:]))
            pts = [(int(coords[i] * img_w), int(coords[i+1] * img_h))
                   for i in range(0, len(coords) - 1, 2)]
            polys.append(np.array(pts, dtype=np.int32).reshape(-1, 1, 2))
    return polys


def ensure_bgr(img):
    if img is None:
        return img
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def normalize_to_uint8(img):
    """Convert scientific images (16-bit/float) into display/model-safe uint8."""
    if img is None or img.dtype == np.uint8:
        return img

    if np.issubdtype(img.dtype, np.floating):
        finite_mask = np.isfinite(img)
        if not finite_mask.any():
            return np.zeros_like(img, dtype=np.uint8)
        finite_vals = img[finite_mask]
        min_val = float(finite_vals.min())
        max_val = float(finite_vals.max())
    else:
        min_val = float(img.min())
        max_val = float(img.max())

    if max_val <= min_val:
        return np.zeros_like(img, dtype=np.uint8)

    scaled = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def load_image_bgr(path: Path):
    """
    Read an image with original depth, then convert it into 8-bit BGR.
    This avoids QImage stride/dtype corruption with 16-bit TIFF inputs.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if len(img.shape) == 2:
        img = normalize_to_uint8(img)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] > 4:
        img = img[:, :, :3]

    return normalize_to_uint8(img)

def draw_gt_layer(base_bgr, polys):
    """Draw ground-truth polygons (filled + outline) onto a copy of base_bgr."""
    if not polys:
        return base_bgr.copy()
    base_bgr = ensure_bgr(base_bgr)
    out = base_bgr.copy()
    overlay = base_bgr.copy()
    color = GT_COLOR
    for poly in polys:
        cv2.fillPoly(overlay, [poly], color)
    out = cv2.addWeighted(overlay, GT_FILL_A, out, 1 - GT_FILL_A, 0)
    for poly in polys:
        cv2.polylines(out, [poly], isClosed=True, color=color, thickness=2)
    return out


def draw_pred_layer(base_bgr, masks_data):
    """
    Draw YOLO prediction masks (binary mask arrays) onto a copy of base_bgr.
    masks_data: list of (H,W) float32 arrays in [0,1].
    """
    if not masks_data:
        return base_bgr.copy()
    base_bgr = ensure_bgr(base_bgr)
    h, w = base_bgr.shape[:2]
    out = base_bgr.copy()
    overlay = base_bgr.copy()
    for i, mask in enumerate(masks_data):
        mask_r = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        binary = (mask_r > 0.5).astype(np.uint8)
        colored = np.full_like(overlay, GT_COLOR)
        mask3 = binary[:, :, None]
        overlay = np.where(mask3, colored, overlay)
    out = cv2.addWeighted(overlay, PRED_FILL_A, out, 1 - PRED_FILL_A, 0)
    # Outlines
    for i, mask in enumerate(masks_data):
        color = GT_COLOR
        mask_r = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        binary = (mask_r > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
    return out


def bgr_to_qimage(bgr):
    bgr = normalize_to_uint8(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()


def get_test_images():
    if TEST_IMG_DIR.exists():
        return sorted([p for p in TEST_IMG_DIR.iterdir()
                       if p.suffix.lower() in IMAGE_EXTS])
    return []


def label_path_for(img_path: Path) -> Path:
    """
    Find a matching .txt label for an image, trying multiple locations:
    1. Sibling labels/ folder  (e.g. .../split/images/foo.png → .../split/labels/foo.txt)
    2. Same folder as the image
    3. Any labels/ folder anywhere under the dataset root
    Returns the first match that exists, or a non-existent path if none found.
    """
    stem = img_path.stem

    # 1. Standard YOLO dataset layout
    candidate = img_path.parent.parent / "labels" / (stem + ".txt")
    if candidate.exists():
        return candidate

    # 2. Same directory
    candidate = img_path.parent / (stem + ".txt")
    if candidate.exists():
        return candidate

    # 3. Search all labels/ folders under the dataset root
    dataset_root = SCRIPT_DIR / "yolo_training" / "yolo_dataset"
    if dataset_root.exists():
        for lbl in dataset_root.rglob(stem + ".txt"):
            if lbl.parent.name == "labels":
                return lbl

    return img_path.parent / (stem + ".txt")  # non-existent fallback


# ---------------------------------------------------------------------------
# Image canvas  –  plain QWidget, draws in paintEvent
#   No Qt layout involvement at all — immune to resize side-effects
#   • scroll-wheel  → zoom in/out anchored to cursor
#   • right-button drag → pan
# ---------------------------------------------------------------------------
class ImageCanvas(QWidget):
    ZOOM_STEP = 0.15
    ZOOM_MIN  = 0.05
    ZOOM_MAX  = 16.0

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 350)
        self.setStyleSheet(f"background: #111120; border-radius: 8px; border: 1px solid {BORDER};")
        self.setFocusPolicy(Qt.WheelFocus)

        self._pixmap   = None   # full-resolution composite QPixmap
        self._zoom     = 1.0    # multiplier on top of fit-to-widget scale
        self._offset   = [0, 0] # pan offset in screen pixels
        self._drag_pos = None

    # ── public API ──────────────────────────────────────────────────────
    def set_pixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self._reset_view()
        self.update()

    def clear_image(self):
        self._pixmap = None
        self._zoom   = 1.0
        self._offset = [0, 0]
        self.update()

    # ── internals ───────────────────────────────────────────────────────
    def _fit_scale(self) -> float:
        """Scale so the image fills the widget at zoom=1, any aspect ratio."""
        if not self._pixmap:
            return 1.0
        w, h = self.width(), self.height()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if pw == 0 or ph == 0 or w == 0 or h == 0:
            return 1.0
        return min(w / pw, h / ph)

    def _reset_view(self):
        """Centre image, zoom=1 (fit)."""
        self._zoom   = 1.0
        self._offset = [0, 0]

    def _img_size(self):
        scale = self._fit_scale() * self._zoom
        return int(self._pixmap.width() * scale), int(self._pixmap.height() * scale)

    def _draw_origin(self):
        """Top-left corner of the image in widget coordinates."""
        iw, ih = self._img_size()
        x = (self.width()  - iw) // 2 + self._offset[0]
        y = (self.height() - ih) // 2 + self._offset[1]
        return x, y

    # ── events ──────────────────────────────────────────────────────────
    def paintEvent(self, event):
        from PySide6.QtGui import QPainter
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#111120"))

        if self._pixmap is None:
            painter.setPen(QColor(TEXT_DIM))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image selected")
            return

        iw, ih = self._img_size()
        x, y = self._draw_origin()
        painter.drawPixmap(x, y, iw, ih, self._pixmap)

    def resizeEvent(self, event):
        # Reset pan offset when widget resizes so image stays centred
        self._offset = [0, 0]
        super().resizeEvent(event)
        self.update()

    def wheelEvent(self, event):
        if self._pixmap is None:
            return
        delta  = event.angleDelta().y()
        factor = (1 + self.ZOOM_STEP) if delta > 0 else (1 - self.ZOOM_STEP)
        new_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self._zoom * factor))

        # Anchor zoom to cursor position
        cx = event.position().x()
        cy = event.position().y()
        x0, y0 = self._draw_origin()
        # Pixel in image space under cursor
        ratio = new_zoom / self._zoom
        self._offset[0] = int(cx - (cx - x0) * ratio - (self.width()  - int(self._pixmap.width()  * self._fit_scale() * new_zoom)) // 2)
        self._offset[1] = int(cy - (cy - y0) * ratio - (self.height() - int(self._pixmap.height() * self._fit_scale() * new_zoom)) // 2)
        self._zoom = new_zoom
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton and self._pixmap is not None:
            self._drag_pos = event.globalPosition().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None:
            delta = event.globalPosition().toPoint() - self._drag_pos
            self._drag_pos = event.globalPosition().toPoint()
            self._offset[0] += delta.x()
            self._offset[1] += delta.y()
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._drag_pos = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dendrite YOLO Viewer")
        self.resize(1280, 800)

        self.model = None
        self._image_paths: list[Path] = []
        self._pred_cache: dict[Path, list] = {}  # path → list of (H,W) mask arrays
        self._current_bgr      = None
        self._current_gt_polys = []
        self._current_pred_masks = []
        self._current_path: Path | None = None

        self._signals = WorkerSignals()
        self._signals.infer_done.connect(self._on_infer_done)
        self._signals.model_loaded.connect(self._on_model_loaded)
        self._signals.composite_ready.connect(self._on_composite_ready)
        self._signals.batch_progress.connect(self._on_batch_progress)
        self._signals.batch_done.connect(self._on_batch_done)

        self._build_ui()
        self._populate_test_images()

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)
        root.addWidget(splitter)

        # ── Left sidebar ──────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(260)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(10)

        # Model
        model_grp = QGroupBox("Model")
        mg = QVBoxLayout(model_grp)
        mg.setSpacing(6)
        row = QHBoxLayout()
        self.model_edit = QLineEdit(str(MODEL_PATH))
        self.model_edit.setPlaceholderText("Path to .pt…")
        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(30)
        browse_btn.setObjectName("secondary")
        browse_btn.clicked.connect(self._browse_model)
        row.addWidget(self.model_edit)
        row.addWidget(browse_btn)
        mg.addLayout(row)
        load_btn = QPushButton("⚡  Load Model")
        load_btn.clicked.connect(self._load_model_async)
        mg.addWidget(load_btn)
        self.model_status = QLabel("Not loaded")
        self.model_status.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px;")
        self.model_status.setAlignment(Qt.AlignCenter)
        mg.addWidget(self.model_status)
        sl.addWidget(model_grp)

        # Images
        img_grp = QGroupBox("Images")
        ig = QVBoxLayout(img_grp)
        ig.setSpacing(6)
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QListWidget.SingleSelection)
        self.image_list.currentRowChanged.connect(self._on_row_changed)
        ig.addWidget(self.image_list)
        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add Images")
        add_btn.setObjectName("secondary")
        add_btn.clicked.connect(self._browse_images)
        clr_btn = QPushButton("Clear")
        clr_btn.setObjectName("secondary")
        clr_btn.clicked.connect(self._clear_list)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(clr_btn)
        ig.addLayout(btn_row)
        sl.addWidget(img_grp, stretch=1)

        self.run_btn = QPushButton("▶  Run YOLO")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run_yolo)
        sl.addWidget(self.run_btn)

        self.export_all_btn = QPushButton("📦  Export All")
        self.export_all_btn.setObjectName("secondary")
        self.export_all_btn.setToolTip("Run YOLO on every image in the list and save to output folder")
        self.export_all_btn.setEnabled(False)
        self.export_all_btn.clicked.connect(self._export_all)
        sl.addWidget(self.export_all_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setFixedHeight(6)
        sl.addWidget(self.progress)

        splitter.addWidget(sidebar)

        # ── Right: toggle bar + canvas ────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(10)

        toggle_bar = QFrame()
        toggle_bar.setStyleSheet(
            f"background: {PANEL_BG}; border-radius: 8px; border: 1px solid {BORDER};")
        toggle_bar.setFixedHeight(56)
        tb = QHBoxLayout(toggle_bar)
        tb.setContentsMargins(20, 0, 20, 0)
        tb.setSpacing(32)

        lbl = QLabel("Show overlays:")
        lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px; font-weight:600;")
        tb.addWidget(lbl)

        # Ground-truth checkbox  (sky-blue)
        self.cb_gt = QCheckBox("Ground Truth")
        self.cb_gt.setChecked(True)
        self.cb_gt.setStyleSheet(
            "QCheckBox { color: #7dd3fc; font-weight: 700; }"
            f"QCheckBox::indicator {{ width:18px; height:18px; border-radius:4px;"
            f"  border:2px solid {BORDER}; background:{PANEL_BG}; }}"
            "QCheckBox::indicator:checked { background:#0ea5e9; border-color:#0ea5e9; }"
        )
        self.cb_gt.stateChanged.connect(self._refresh_composite)
        tb.addWidget(self.cb_gt)

        # YOLO prediction checkbox  (purple)
        self.cb_pred = QCheckBox("YOLO Prediction")
        self.cb_pred.setChecked(False)
        self.cb_pred.setStyleSheet(
            "QCheckBox { color: #c4b5fd; font-weight: 700; }"
            f"QCheckBox::indicator {{ width:18px; height:18px; border-radius:4px;"
            f"  border:2px solid {BORDER}; background:{PANEL_BG}; }}"
            "QCheckBox::indicator:checked { background:#7c3aed; border-color:#7c3aed; }"
        )
        self.cb_pred.stateChanged.connect(self._refresh_composite)
        tb.addWidget(self.cb_pred)

        tb.addStretch()

        self.export_btn = QPushButton("💾  Export")
        self.export_btn.setObjectName("secondary")
        self.export_btn.setToolTip("Save current image with YOLO prediction overlay (full resolution)")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_image)
        tb.addWidget(self.export_btn)

        self.info_label = QLabel("")
        self.info_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px;")
        tb.addWidget(self.info_label)

        rl.addWidget(toggle_bar)
        self.canvas = ImageCanvas()
        self.canvas.setMinimumSize(400, 350)
        rl.addWidget(self.canvas, stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready.")

    # ------------------------------------------------------------------
    def _populate_test_images(self):
        for p in get_test_images():
            self._add_to_list(p)

    def _add_to_list(self, path: Path):
        if path in self._image_paths:
            return
        self._image_paths.append(path)
        item = QListWidgetItem(path.name)
        item.setToolTip(str(path))
        self.image_list.addItem(item)

    def _clear_list(self):
        self.image_list.clear()
        self._image_paths.clear()
        self._pred_cache.clear()
        self._current_bgr = None
        self._current_gt_polys = []
        self._current_pred_masks = []
        self._current_path = None
        self.canvas.clear_image()
        self.info_label.setText("")

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO model", str(SCRIPT_DIR / "yolo_training"),
            "PyTorch weights (*.pt);;All (*.*)")
        if path:
            self.model_edit.setText(path)

    def _load_model_async(self):
        self.model_status.setText("Loading…")
        self.model_status.setStyleSheet(f"color: {WARNING};")
        self.progress.setVisible(True)
        threading.Thread(target=self._load_model_worker, daemon=True).start()

    def _load_model_worker(self):
        try:
            from ultralytics import YOLO
            m = YOLO(self.model_edit.text())
            self.model = m
            self._signals.model_loaded.emit(True, Path(self.model_edit.text()).name)
        except Exception as e:
            self._signals.model_loaded.emit(False, str(e))

    def _on_model_loaded(self, ok, msg):
        self.progress.setVisible(False)
        if ok:
            self.model_status.setText(f"✓  {msg}")
            self.model_status.setStyleSheet(f"color: {SUCCESS};")
            self.status_bar.showMessage(f"Model loaded: {msg}")
        else:
            self.model_status.setText("✗  Failed")
            self.model_status.setStyleSheet(f"color: {DANGER};")
            self.status_bar.showMessage(f"Error: {msg}")
        self._update_run_btn()

    def _update_run_btn(self):
        has_model = self.model is not None
        self.run_btn.setEnabled(has_model and self.image_list.currentRow() >= 0)
        self.export_all_btn.setEnabled(has_model and len(self._image_paths) > 0)

    def _browse_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select images", str(SCRIPT_DIR),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All (*.*)")
        for p in paths:
            self._add_to_list(Path(p))

    def _on_row_changed(self, row):
        if row < 0:
            return
        path = self._image_paths[row]
        bgr = load_image_bgr(path)
        if bgr is None:
            self.status_bar.showMessage(f"Cannot read: {path}")
            return
        self._current_bgr  = ensure_bgr(bgr)
        self._current_path = path
        h, w = bgr.shape[:2]

        # Load ground truth if available
        lbl = label_path_for(path)
        self._current_gt_polys = load_gt_polygons(lbl, w, h)
        gt_info = f"{len(self._current_gt_polys)} GT poly(s)" if self._current_gt_polys else "no GT labels"

        # Restore cached predictions for this image (if any)
        cached = self._pred_cache.get(path, [])
        self._current_pred_masks = cached
        self.cb_pred.setChecked(bool(cached))
        self.export_btn.setEnabled(bool(cached))

        self.canvas.clear_image()
        self.info_label.setText(f"{w}×{h}  |  {gt_info}")
        self._refresh_composite()
        self._update_run_btn()
        pred_info = f"{len(cached)} prediction(s) cached" if cached else "no predictions yet"
        self.status_bar.showMessage(f"{path.name}  —  {gt_info}  |  {pred_info}")

    def _run_yolo(self):
        row = self.image_list.currentRow()
        if row < 0 or not self.model:
            return
        path = self._image_paths[row]
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.status_bar.showMessage(f"Running inference on {path.name}…")
        threading.Thread(target=self._infer_worker, args=(path,), daemon=True).start()

    def _infer_worker(self, path):
        try:
            bgr = load_image_bgr(path)
            if bgr is None:
                self._signals.infer_done.emit(None, f"Cannot read: {path.name}")
                return
            # Ensure 3-channel 8-bit BGR for the model.
            if len(bgr.shape) == 2 or bgr.shape[2] == 1:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
            result = self.model(bgr, verbose=False)[0]
            if result.masks is not None:
                masks = [m for m in result.masks.data.cpu().numpy()]
            else:
                masks = []
            n = len(masks)
            self._signals.infer_done.emit(masks, f"{n} prediction(s) — {path.name}")
        except Exception as e:
            self._signals.infer_done.emit(None, f"Inference error: {e}")

    def _on_infer_done(self, masks, msg):
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        if masks is None:
            self.status_bar.showMessage(msg)
            return
        if self._current_path is not None:
            self._pred_cache[self._current_path] = masks
        self._current_pred_masks = masks
        self.cb_pred.setChecked(True)
        self.export_btn.setEnabled(True)
        self._refresh_composite()
        self.status_bar.showMessage(msg)

    def _export_image(self):
        if self._current_bgr is None or not self._current_pred_masks:
            return
        stem = self._current_path.stem if self._current_path else "export"
        default_name = str(SCRIPT_DIR / f"{stem}_yolo.png")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save image", default_name,
            "PNG (*.png);;JPEG (*.jpg);;All files (*.*)")
        if not path:
            return
        # Build full-resolution composite with prediction overlay
        display = self._current_bgr.copy()
        if self.cb_gt.isChecked() and self._current_gt_polys:
            display = draw_gt_layer(display, self._current_gt_polys)
        display = draw_pred_layer(display, self._current_pred_masks)
        ok = cv2.imwrite(path, display)
        if ok:
            self.status_bar.showMessage(f"Saved: {path}")
        else:
            self.status_bar.showMessage(f"Failed to save: {path}")

    def _export_all(self):
        if not self.model or not self._image_paths:
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select output folder", str(SCRIPT_DIR))
        if not out_dir:
            return
        out_path = Path(out_dir)

        # Lock UI during batch
        self.export_all_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.progress.setRange(0, len(self._image_paths))
        self.progress.setValue(0)
        self.progress.setVisible(True)

        paths = list(self._image_paths)
        model = self.model

        def worker():
            for i, img_path in enumerate(paths):
                self._signals.batch_progress.emit(i, len(paths), img_path.name)
                bgr = load_image_bgr(img_path)
                if bgr is None:
                    continue
                # Use cached masks if already computed, otherwise run inference
                if img_path in self._pred_cache:
                    masks = self._pred_cache[img_path]
                else:
                    try:
                        if len(bgr.shape) == 2 or bgr.shape[2] == 1:
                            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
                        result = model(bgr, verbose=False)[0]
                        masks = list(result.masks.data.cpu().numpy()) if result.masks else []
                        self._pred_cache[img_path] = masks
                    except Exception:
                        masks = []
                if not masks:
                    continue
                display = draw_pred_layer(bgr.copy(), masks)
                out_file = out_path / f"{img_path.stem}_yolo.png"
                cv2.imwrite(str(out_file), display)

            self._signals.batch_done.emit(str(out_path))

        threading.Thread(target=worker, daemon=True).start()

    def _on_batch_progress(self, done, total, filename):
        self.progress.setValue(done)
        self.status_bar.showMessage(f"Exporting {done+1}/{total}: {filename}")

    def _on_batch_done(self, out_dir):
        self.progress.setVisible(False)
        self.export_all_btn.setEnabled(True)
        self._update_run_btn()
        self.status_bar.showMessage(f"Export complete → {out_dir}")

    # ------------------------------------------------------------------
    # Composite: offload drawing to a background thread
    # ------------------------------------------------------------------
    def _refresh_composite(self):
        if self._current_bgr is None:
            self.canvas.clear_image()
            return

        # Snapshot everything — worker is fully self-contained, no shared state
        bgr        = self._current_bgr
        gt_polys   = self._current_gt_polys[:]   if self.cb_gt.isChecked()   else []
        pred_masks = self._current_pred_masks[:] if self.cb_pred.isChecked() else []

        def worker():
            try:
                display = bgr.copy()
                if gt_polys:
                    display = draw_gt_layer(display, gt_polys)
                if pred_masks:
                    display = draw_pred_layer(display, pred_masks)
                self._signals.composite_ready.emit(bgr_to_qimage(display))
            except Exception as e:
                print(f"[composite worker error] {e}")

        threading.Thread(target=worker, daemon=True).start()

    def _on_composite_ready(self, image):
        self.canvas.set_pixmap(QPixmap.fromImage(image))


# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(DARK_BG))
    palette.setColor(QPalette.WindowText,      QColor(TEXT))
    palette.setColor(QPalette.Base,            QColor(PANEL_BG))
    palette.setColor(QPalette.AlternateBase,   QColor(DARK_BG))
    palette.setColor(QPalette.Text,            QColor(TEXT))
    palette.setColor(QPalette.Button,          QColor(PANEL_BG))
    palette.setColor(QPalette.ButtonText,      QColor(TEXT))
    app.setPalette(palette)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
