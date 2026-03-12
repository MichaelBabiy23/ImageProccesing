
import sys
import os
import traceback
import cv2
import numpy as np
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsPolygonItem, QGraphicsEllipseItem,
    QListWidget, QMessageBox, QShortcut, QFrame
)
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPolygonF, QPen, QBrush, QColor, QKeySequence, QPainter

# Add project root to path for utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import load_image

class PointItem(QGraphicsEllipseItem):
    """A draggable point representing a polygon vertex."""
    def __init__(self, x, y, parent_polygon, index, radius=1):
        super().__init__(-radius, -radius, radius*2, radius*2)
        self.setPos(x, y)
        self.parent_polygon = parent_polygon
        self.index = index
        self.is_selected_point = False
        self.default_brush = QBrush(QColor(255, 255, 0))
        self.selected_brush = QBrush(QColor(255, 80, 80))
        self.default_pen = QPen(Qt.black, 1)
        self.selected_pen = QPen(QColor(255, 255, 255), 2)
        self.setFlags(
            QGraphicsEllipseItem.ItemIsMovable |
            QGraphicsEllipseItem.ItemSendsScenePositionChanges
        )
        self.setZValue(10)
        self.update_appearance()

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.ItemPositionChange and self.scene():
            self.parent_polygon.update_point(self.index, value)
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.parent_polygon.setSelected(True)
            self.select_point(additive=bool(event.modifiers() & Qt.ControlModifier))
        super().mousePressEvent(event)

    def select_point(self, additive=False):
        if not additive:
            scene = self.scene()
            if scene is not None:
                for item in scene.items():
                    if isinstance(item, PointItem) and item is not self:
                        item.set_point_selected(False)
        self.set_point_selected(True)

    def set_point_selected(self, selected):
        self.is_selected_point = selected
        self.update_appearance()

    def update_appearance(self):
        if self.is_selected_point:
            self.setBrush(self.selected_brush)
            self.setPen(self.selected_pen)
        else:
            self.setBrush(self.default_brush)
            self.setPen(self.default_pen)

class EditablePolygon(QGraphicsPolygonItem):
    """A polygon that can be edited by dragging its vertices."""
    def __init__(self, points, scene, class_id=0):
        super().__init__(QPolygonF(points))
        self.scene = scene
        self.class_id = class_id
        self.points = points
        self.point_items = []
        self.setPen(QPen(QColor(0, 255, 0), 2))
        self.setBrush(QBrush(QColor(0, 255, 0, 80)))
        self.setZValue(5)
        self.setFlags(QGraphicsPolygonItem.ItemIsSelectable | QGraphicsPolygonItem.ItemIsFocusable)
        self.setAcceptHoverEvents(True)
        
    def itemChange(self, change, value):
        if change == QGraphicsPolygonItem.ItemSelectedHasChanged:
            if value: # Selected
                self.show_points()
            else: # Deselected
                self.hide_points()
        return super().itemChange(change, value)

    def show_points(self):
        selected_indices = {item.index for item in self.point_items if item.is_selected_point}
        self.hide_points()
        for i, pt in enumerate(self.points):
            item = PointItem(pt.x(), pt.y(), self, i)
            self.scene.addItem(item)
            if i in selected_indices:
                item.set_point_selected(True)
            self.point_items.append(item)

    def hide_points(self):
        for item in self.point_items:
            self.scene.removeItem(item)
        self.point_items = []

    def update_point(self, index, pos):
        self.points[index] = pos
        self.setPolygon(QPolygonF(self.points))

    def remove_point(self, index):
        return self.remove_points([index])

    def remove_points(self, indices):
        valid_indices = sorted({index for index in indices if 0 <= index < len(self.points)}, reverse=True)
        if not valid_indices:
            return True

        remaining_points = [point for i, point in enumerate(self.points) if i not in set(valid_indices)]
        self.hide_points()

        if len(remaining_points) < 3:
            self.points = remaining_points
            self.scene.removeItem(self)
            return False

        self.points = remaining_points
        self.setPolygon(QPolygonF(self.points))
        self.setSelected(True)
        self.show_points()
        return True

    def remove_from_scene(self):
        self.hide_points()
        self.scene.removeItem(self)

    def get_points(self):
        return [(p.x(), p.y()) for p in self.points]

    def set_annotation_visible(self, visible):
        self.setVisible(visible)
        if visible and self.isSelected():
            self.show_points()
        else:
            self.hide_points()

class AnnotationScene(QGraphicsScene):
    """Scene for drawing and editing annotations."""
    polygon_finished = pyqtSignal()
    drawing_state_changed = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.image_item = None
        self.polygons = []
        self.current_points = []
        self.temp_line = None
        self.is_drawing = False
        self.resume_snapshot = None
        self.undo_stack = []
        self.redo_stack = []
        self.annotations_visible = True

    def set_image(self, pixmap):
        self.clear()
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.addItem(self.image_item)
        self.polygons = []
        self.current_points = []
        self.temp_line = None
        self.is_drawing = False
        self.resume_snapshot = None
        self.undo_stack = []
        self.redo_stack = []
        self.annotations_visible = True
        self.setSceneRect(QRectF(pixmap.rect()))
        self.drawing_state_changed.emit(False)

    def mousePressEvent(self, event):
        if self.is_drawing:
            clicked_item = self.itemAt(event.scenePos(), self.views()[0].transform()) if self.views() else None
            if isinstance(clicked_item, PointItem):
                clicked_item.parent_polygon.setSelected(True)
                clicked_item.select_point(additive=bool(event.modifiers() & Qt.ControlModifier))
                super().mousePressEvent(event)
                return
            if isinstance(clicked_item, EditablePolygon):
                clicked_item.setSelected(True)
                super().mousePressEvent(event)
                return
            if event.button() == Qt.LeftButton:
                pos = event.scenePos()
                self.current_points.append(pos)
                self.update_drawing()
                event.accept()
                return
            if event.button() == Qt.RightButton:
                self.cancel_drawing()
                event.accept()
                return
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_drawing and self.current_points:
            self.update_drawing(event.scenePos())
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)

    def update_drawing(self, mouse_pos=None):
        # Remove old temp lines
        for item in self.items():
            if isinstance(item, QGraphicsPolygonItem) and item.zValue() == 100:
                self.removeItem(item)
        
        if not self.current_points:
            return

        pts = list(self.current_points)
        if mouse_pos:
            pts.append(mouse_pos)
        
        poly = QPolygonF(pts)
        item = QGraphicsPolygonItem(poly)
        item.setPen(QPen(Qt.red, 2, Qt.DashLine))
        item.setZValue(100)
        self.addItem(item)

    def finish_polygon(self):
        if len(self.current_points) > 2:
            poly = EditablePolygon(self.current_points, self)
            self.addItem(poly)
            self.polygons.append(poly)
            poly.set_annotation_visible(self.annotations_visible)
            new_snapshot = self._polygon_snapshot(poly)
            if self.resume_snapshot is not None:
                self._record_action({
                    "type": "replace_polygon",
                    "before": self.resume_snapshot,
                    "after": new_snapshot,
                })
            else:
                self._record_action({
                    "type": "create_polygon",
                    "polygon": new_snapshot,
                })
            self.current_points = []
            self.update_drawing()
            self.is_drawing = False
            self.resume_snapshot = None
            self.drawing_state_changed.emit(False)
            self.polygon_finished.emit()

    def cancel_drawing(self):
        if self.resume_snapshot is not None:
            self._restore_polygon_snapshot(self.resume_snapshot)
        self.current_points = []
        self.is_drawing = False
        self.resume_snapshot = None
        self.update_drawing()
        self.drawing_state_changed.emit(False)

    def _selected_polygon_for_resume(self):
        selected_polygons = [item for item in self.selectedItems() if isinstance(item, EditablePolygon)]
        if len(selected_polygons) != 1:
            return None, None

        polygon = selected_polygons[0]
        selected_nodes = [item for item in polygon.point_items if item.is_selected_point]
        if len(selected_nodes) != 1:
            return polygon, None
        return polygon, selected_nodes[0].index

    def start_drawing(self):
        target_polygon, selected_index = self._selected_polygon_for_resume()
        if target_polygon is not None and selected_index is None:
            return False, "Select exactly one node on the annotation before resuming drawing."

        self.clearSelection()
        for existing_polygon in self.polygons:
            for item in existing_polygon.point_items:
                item.set_point_selected(False)

        if target_polygon is not None and selected_index is not None:
            self.resume_snapshot = self._polygon_snapshot(target_polygon)
            points = list(target_polygon.points)
            self.current_points = points[selected_index + 1:] + points[:selected_index + 1]
            target_polygon.remove_from_scene()
            if target_polygon in self.polygons:
                self.polygons.remove(target_polygon)
        else:
            self.resume_snapshot = None
            self.current_points = []

        self.is_drawing = True
        self.update_drawing()
        self.drawing_state_changed.emit(True)
        return True, ""

    def add_polygon_from_points(self, points):
        qpoints = [QPointF(x, y) for x, y in points]
        if len(qpoints) > 2:
            poly = EditablePolygon(qpoints, self)
            self.addItem(poly)
            self.polygons.append(poly)
            poly.set_annotation_visible(self.annotations_visible)
            return poly
        return None

    def _polygon_snapshot(self, polygon):
        return {
            "points": polygon.get_points(),
            "class_id": polygon.class_id,
        }

    def _restore_polygon_snapshot(self, snapshot):
        qpoints = [QPointF(x, y) for x, y in snapshot["points"]]
        if len(qpoints) < 3:
            return None
        poly = EditablePolygon(qpoints, self, class_id=snapshot.get("class_id", 0))
        self.addItem(poly)
        self.polygons.append(poly)
        poly.set_annotation_visible(self.annotations_visible)
        return poly

    def _remove_polygon_by_snapshot(self, snapshot):
        target_points = snapshot["points"]
        target_class_id = snapshot.get("class_id", 0)
        for polygon in list(self.polygons):
            if polygon.class_id == target_class_id and polygon.get_points() == target_points:
                polygon.remove_from_scene()
                if polygon in self.polygons:
                    self.polygons.remove(polygon)
                return True
        return False

    def _record_action(self, action):
        self.undo_stack.append(action)
        self.redo_stack.clear()

    def set_annotations_visible(self, visible):
        self.annotations_visible = visible
        for polygon in self.polygons:
            polygon.set_annotation_visible(visible)

    def delete_selected(self):
        selected_items = list(self.selectedItems())
        selected_points = []
        for polygon in self.polygons:
            for item in polygon.point_items:
                if item.is_selected_point:
                    selected_points.append((item.parent_polygon, item.index))
        if selected_points:
            grouped_points = {}
            for polygon, index in selected_points:
                grouped_points.setdefault(polygon, []).append(index)

            for polygon in self.polygons:
                for item in polygon.point_items:
                    item.set_point_selected(False)

            for polygon, indices in grouped_points.items():
                if polygon not in self.polygons:
                    continue
                polygon_survived = polygon.remove_points(indices)
                if not polygon_survived and polygon in self.polygons:
                    self.polygons.remove(polygon)
            return

        deleted_polygons = []
        for item in selected_items:
            if isinstance(item, EditablePolygon):
                deleted_polygons.append(self._polygon_snapshot(item))
                item.remove_from_scene()
                if item in self.polygons:
                    self.polygons.remove(item)

        if deleted_polygons:
            self._record_action({
                "type": "delete_polygons",
                "polygons": deleted_polygons,
            })

    def undo(self):
        if not self.undo_stack:
            return

        action = self.undo_stack.pop()
        if action["type"] == "delete_polygons":
            restored = []
            for snapshot in action["polygons"]:
                poly = self._restore_polygon_snapshot(snapshot)
                if poly is not None:
                    restored.append(snapshot)
            self.redo_stack.append({
                "type": "delete_polygons",
                "polygons": restored,
            })
        elif action["type"] == "create_polygon":
            if self._remove_polygon_by_snapshot(action["polygon"]):
                self.redo_stack.append(action)
        elif action["type"] == "replace_polygon":
            removed_new = self._remove_polygon_by_snapshot(action["after"])
            restored_old = self._restore_polygon_snapshot(action["before"])
            if removed_new or restored_old is not None:
                self.redo_stack.append(action)

    def redo(self):
        if not self.redo_stack:
            return

        action = self.redo_stack.pop()
        if action["type"] == "delete_polygons":
            deleted_again = []
            for snapshot in action["polygons"]:
                if self._remove_polygon_by_snapshot(snapshot):
                    deleted_again.append(snapshot)
            if deleted_again:
                self.undo_stack.append({
                    "type": "delete_polygons",
                    "polygons": deleted_again,
                })
        elif action["type"] == "create_polygon":
            poly = self._restore_polygon_snapshot(action["polygon"])
            if poly is not None:
                self.undo_stack.append(action)
        elif action["type"] == "replace_polygon":
            self._remove_polygon_by_snapshot(action["before"])
            poly = self._restore_polygon_snapshot(action["after"])
            if poly is not None:
                self.undo_stack.append(action)

class AnnotationView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self._is_panning = False
        self._last_pan_pos = None

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        
        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._is_panning = True
            self._last_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_panning and self._last_pan_pos is not None:
            delta = event.pos() - self._last_pan_pos
            self._last_pan_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self._is_panning:
            self._is_panning = False
            self._last_pan_pos = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

class AnnotatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Dendrite Annotator")
        self.resize(1200, 800)

        self.project_root = os.path.dirname(os.path.dirname(__file__))
        self.image_dir = self._resolve_directory(os.path.join(self.project_root, "data", "raw"), "image")
        self.mask_dir = self._resolve_directory(os.path.join(self.project_root, "output", "classic"), "mask")
        self.label_dir = self._resolve_directory(os.path.join(self.project_root, "data", "raw"), "label")
        self.images = []
        self.current_idx = -1
        self._hide_annotations_held = False

        self.init_ui()
        self.setup_shortcuts()
        self.load_images()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left side: Controls
        controls = QVBoxLayout()
        layout.addLayout(controls, 1)

        controls.addWidget(QLabel("Directories:"))
        
        self.btn_img_dir = QPushButton("Set Image Dir")
        self.btn_img_dir.clicked.connect(self.select_img_dir)
        controls.addWidget(self.btn_img_dir)

        self.btn_mask_dir = QPushButton("Set Mask Dir (Classic Output)")
        self.btn_mask_dir.clicked.connect(self.select_mask_dir)
        controls.addWidget(self.btn_mask_dir)

        self.btn_label_dir = QPushButton("Set Label Dir (YOLO)")
        self.btn_label_dir.clicked.connect(self.select_label_dir)
        controls.addWidget(self.btn_label_dir)

        controls.addWidget(QFrame()) # Spacer
        
        self.info_label = QLabel("No images loaded")
        controls.addWidget(self.info_label)

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.change_image)
        controls.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        self.btn_prev = QPushButton("Prev (Left)")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next = QPushButton("Next (Right)")
        self.btn_next.clicked.connect(self.next_image)
        btn_layout.addWidget(self.btn_prev)
        btn_layout.addWidget(self.btn_next)
        controls.addLayout(btn_layout)

        controls.addWidget(QLabel("Tools:"))
        self.btn_draw = QPushButton("Draw Shape (D)")
        self.btn_draw.clicked.connect(self.handle_draw_action)
        controls.addWidget(self.btn_draw)

        self.btn_load_mask = QPushButton("Load from Mask (M)")
        self.btn_load_mask.clicked.connect(self.auto_load_mask)
        controls.addWidget(self.btn_load_mask)

        self.btn_save = QPushButton("SAVE (Ctrl+S)")
        self.btn_save.clicked.connect(self.save_annotation)
        controls.addWidget(self.btn_save)

        self.btn_delete = QPushButton("Delete Selected (Del)")
        self.btn_delete.clicked.connect(self.delete_selected)
        controls.addWidget(self.btn_delete)

        controls.addWidget(QLabel("Shortcuts:\n- Scroll: Zoom\n- Right Click + Drag: Pan image\n- Left/Right: Prev/Next image\n- D: Start or finish drawing\n- H (hold): Hide annotations\n- M: Load Mask\n- Left Click: Add Point while drawing\n- Right Click: Cancel drawing\n- Drag Points: Edit nodes\n- Del: Delete selected node or polygon\n- Ctrl+Z / Ctrl+Y: Undo/Redo shape delete"))

        # Right side: View
        self.scene = AnnotationScene()
        self.scene.drawing_state_changed.connect(self.sync_drawing_state)
        self.view = AnnotationView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        layout.addWidget(self.view, 4)

    def setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_annotation)
        QShortcut(QKeySequence("Delete"), self, self.delete_selected)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.redo)
        QShortcut(QKeySequence("D"), self, self.handle_draw_action)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.next_image)
        QShortcut(QKeySequence(Qt.Key_Left), self, self.prev_image)
        QShortcut(QKeySequence("M"), self, self.auto_load_mask)

    def _resolve_directory(self, selected_path, directory_type):
        if not selected_path:
            return ""

        selected_path = os.path.abspath(selected_path)
        candidates = [selected_path]

        if directory_type == "image":
            candidates = [
                os.path.join(selected_path, "data", "raw"),
                os.path.join(selected_path, "raw"),
                selected_path,
            ]
        elif directory_type == "mask":
            candidates = [
                os.path.join(selected_path, "output", "classic"),
                os.path.join(selected_path, "classic"),
                selected_path,
            ]
        elif directory_type == "label":
            candidates = [
                os.path.join(selected_path, "data", "raw"),
                os.path.join(selected_path, "raw"),
                selected_path,
            ]

        for candidate in candidates:
            if os.path.isdir(candidate):
                return os.path.abspath(candidate)
        return selected_path

    def _choose_directory(self, title, current_path, directory_type):
        start_dir = current_path or self.project_root
        path = QFileDialog.getExistingDirectory(self, title, start_dir)
        if not path:
            return ""
        return self._resolve_directory(path, directory_type)

    def _current_image_relative_dir(self):
        if self.current_idx < 0 or self.current_idx >= len(self.images):
            return ""
        return os.path.dirname(self.images[self.current_idx])

    def _label_path_for_current_image(self):
        if not self.label_dir or self.current_idx < 0:
            return ""

        basename = os.path.splitext(os.path.basename(self.images[self.current_idx]))[0]
        rel_dir = self._current_image_relative_dir()
        candidates = [
            os.path.join(self.label_dir, rel_dir, "annotations", basename + ".txt"),
            os.path.join(self.label_dir, "annotations", rel_dir, "annotations", basename + ".txt"),
            os.path.join(self.label_dir, "annotations", rel_dir, basename + ".txt"),
            os.path.join(self.label_dir, rel_dir, basename + ".txt"),
            os.path.join(self.label_dir, basename + ".txt"),
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        return candidates[0]

    def select_img_dir(self):
        path = self._choose_directory("Select Image Directory", self.image_dir, "image")
        if path:
            self.image_dir = path
            self.load_images()

    def select_mask_dir(self):
        path = self._choose_directory("Select Mask Directory", self.mask_dir, "mask")
        if path:
            self.mask_dir = path

    def select_label_dir(self):
        path = self._choose_directory("Select Label Directory", self.label_dir, "label")
        if path:
            self.label_dir = path

    def load_images(self):
        if not self.image_dir or not os.path.isdir(self.image_dir):
            self.images = []
            self.list_widget.clear()
            self.info_label.setText("No images loaded")
            return

        image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        self.images = []
        for root, _, files in os.walk(self.image_dir):
            if os.path.basename(root).lower() == "annotations":
                continue
            for filename in files:
                if filename.lower().endswith(image_extensions):
                    full_path = os.path.join(root, filename)
                    self.images.append(os.path.relpath(full_path, self.image_dir))

        self.images.sort()
        self.list_widget.clear()
        self.list_widget.addItems(self.images)
        if self.images:
            self.list_widget.setCurrentRow(0)
        else:
            self.info_label.setText("No images loaded")

    def change_image(self, row):
        if row < 0 or row >= len(self.images):
            return
        self.current_idx = row
        img_path = os.path.join(self.image_dir, self.images[row])
        
        try:
            image = load_image(img_path, grayscale=False)
            height, width, channel = image.shape
            bytesPerLine = 3 * width
            qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qImg)
            self.scene.set_image(pixmap)
            self.info_label.setText(f"Image: {self.images[row]} ({width}x{height})")
            
            self.load_existing_label()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def load_existing_label(self):
        label_path = self._label_path_for_current_image()
        if not label_path:
            return

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            h = self.scene.sceneRect().height()
            w = self.scene.sceneRect().width()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5: continue
                coords = [float(x) for x in parts[1:]]
                points = []
                for i in range(0, len(coords), 2):
                    points.append((coords[i] * w, coords[i+1] * h))
                self.scene.add_polygon_from_points(points)

    def auto_load_mask(self):
        if not self.mask_dir:
            QMessageBox.warning(self, "Warning", "Please set Mask Directory first.")
            return

        basename = os.path.splitext(os.path.basename(self.images[self.current_idx]))[0]
        rel_dir = self._current_image_relative_dir()
        
        mask_candidates = [
            os.path.join(self.mask_dir, rel_dir, basename, "09_separated.png"),
            os.path.join(self.mask_dir, rel_dir, basename, "08_small_removed.png"),
            os.path.join(self.mask_dir, basename, "09_separated.png"),
            os.path.join(self.mask_dir, f"{basename}_mask.png"),
            os.path.join(self.mask_dir, basename, "08_small_removed.png"),
        ]
        
        mask_path = None
        for cand in mask_candidates:
            if os.path.exists(cand):
                mask_path = cand
                break
        
        if not mask_path:
            for root, dirs, files in os.walk(self.mask_dir):
                if "09_separated.png" in files or f"{basename}_mask.png" in files:
                    if basename in root or basename in files:
                        if "09_separated.png" in files:
                            mask_path = os.path.join(root, "09_separated.png")
                        else:
                            mask_path = os.path.join(root, f"{basename}_mask.png")
                        break

        if not mask_path:
            QMessageBox.warning(self, "Not Found", f"Could not find mask for {basename}")
            return

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        
        for cnt in contours:
            if len(cnt) < 3: continue
            points = [(p[0][0], p[0][1]) for p in cnt]
            self.scene.add_polygon_from_points(points)

    def save_annotation(self):
        if not self.label_dir:
            QMessageBox.warning(self, "Warning", "Please set Label Directory first.")
            return
        
        if self.current_idx < 0: return

        label_path = self._label_path_for_current_image()
        if not label_path:
            return
        
        h = self.scene.sceneRect().height()
        w = self.scene.sceneRect().width()
        
        lines = []
        for poly in self.scene.polygons:
            pts = poly.get_points()
            norm_pts = []
            for x, y in pts:
                norm_pts.append(f"{x/w:.6f} {y/h:.6f}")
            lines.append(f"0 {' '.join(norm_pts)}")
        
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path, 'w') as f:
            f.write("\n".join(lines))
        
        self.statusBar().showMessage(f"Saved to {label_path}", 3000)

    def handle_draw_action(self):
        if self.scene.is_drawing:
            if len(self.scene.current_points) > 2:
                self.scene.finish_polygon()
                self.statusBar().showMessage("Shape finished.", 1500)
            else:
                self.statusBar().showMessage("Need at least 3 points to finish the shape.", 3000)
            return

        self.start_drawing()

    def start_drawing(self):
        started, message = self.scene.start_drawing()
        if message:
            self.statusBar().showMessage(message, 3000)
        elif started:
            self.statusBar().showMessage("Drawing started.", 1500)

    def sync_drawing_state(self, is_drawing):
        self.view.setDragMode(QGraphicsView.NoDrag if is_drawing else QGraphicsView.RubberBandDrag)
        self.btn_draw.setText("Drawing..." if is_drawing else "Draw Shape (D)")

    def delete_selected(self):
        self.scene.delete_selected()

    def undo(self):
        self.scene.undo()

    def redo(self):
        self.scene.redo()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_H and not event.isAutoRepeat():
            self._hide_annotations_held = True
            self.scene.set_annotations_visible(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_H and not event.isAutoRepeat():
            self._hide_annotations_held = False
            self.scene.set_annotations_visible(True)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def next_image(self):
        if self.current_idx < len(self.images) - 1:
            self.list_widget.setCurrentRow(self.current_idx + 1)

    def prev_image(self):
        if self.current_idx > 0:
            self.list_widget.setCurrentRow(self.current_idx - 1)

if __name__ == "__main__":
    def excepthook(exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)
        sys.stderr.flush()
        QApplication.quit()

    sys.excepthook = excepthook
    app = QApplication(sys.argv)
    window = AnnotatorApp()
    window.show()
    sys.exit(app.exec_())
