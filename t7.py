import os
os.environ["OPENCV_UI_BACKEND"] = "NONE"
# os.environ["QT_QPA_PLATFORM"] = "wayland"
os.environ["OPENCV_VIDEOIO_BACKEND"] = "V4L2"

import sys
import json
import datetime
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSizePolicy, QComboBox, QCheckBox
)
from PySide6.QtCore import Qt, QTimer, QRect, QPoint, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QGuiApplication

# ------------------------------------------------------------------
# Konstanten TC001
# ------------------------------------------------------------------

TC001_W = 256
TC001_H = 192

# ------------------------------------------------------------------
# Colormaps
# ------------------------------------------------------------------

COLORMAPS = {
    "Turbo":      cv2.COLORMAP_TURBO,
    "Inferno":    cv2.COLORMAP_INFERNO,
    "Hot":        cv2.COLORMAP_HOT,
    "Jet":        cv2.COLORMAP_JET,
    "Magma":      cv2.COLORMAP_MAGMA,
    "Plasma":     cv2.COLORMAP_PLASMA,
    "Rainbow":    cv2.COLORMAP_RAINBOW,
    "Viridis":    cv2.COLORMAP_VIRIDIS,
    "Parula":     cv2.COLORMAP_PARULA,
    "Graustufen": None,
}

# ------------------------------------------------------------------
# Einstellungen
# ------------------------------------------------------------------

SETTINGS_FILE = Path("settings.json")

def load_settings():
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_settings(data: dict):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ------------------------------------------------------------------
# Kameraerkennung
# ------------------------------------------------------------------

def is_tc001(cap) -> bool:
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    ret, frame = cap.read()
    if not ret or frame is None:
        return False
    h, w = frame.shape[:2]
    return w == TC001_W and h == TC001_H * 2

def find_cameras(max_index: int = 16) -> tuple[list[int], int | None]:
    all_cams  = []
    tc001_idx = None
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        ret, _ = cap.read()
        if ret:
            all_cams.append(i)
            if tc001_idx is None and is_tc001(cap):
                tc001_idx = i
        cap.release()
    return all_cams, tc001_idx

# ------------------------------------------------------------------
# Thermaldaten aus Rohdaten
# ------------------------------------------------------------------

def raw_to_celsius(raw_frame) -> np.ndarray:
    raw_data   = raw_frame[TC001_H:]
    raw_uint8  = raw_data.flatten()
    raw_uint16 = (raw_uint8[0::2].astype(np.uint16) |
                  (raw_uint8[1::2].astype(np.uint16) << 8))
    temp_C = raw_uint16 / 64.0 - 273.15
    return temp_C.reshape(TC001_H, TC001_W)

# ------------------------------------------------------------------
# Font-Hilfsfunktionen
# ------------------------------------------------------------------

def _get_font(size: int = 18):
    """Lädt einen System-Font mit Unicode-Unterstützung."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}

def get_font(size: int = 18):
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = _get_font(size)
    return _FONT_CACHE[size]

# ------------------------------------------------------------------
# HUD-Text mit Pillow (Unicode-fähig)
# ------------------------------------------------------------------

def draw_hud_text(
    bgr: np.ndarray,
    texts: list[tuple[str, tuple[int, int], tuple[int, int, int]]],
    font_size: int = 18,
) -> np.ndarray:
    """
    Zeichnet Unicode-Texte via Pillow auf ein BGR-numpy-Array.
    texts: Liste von (text, (x, y), (r, g, b))
    """
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil  = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = get_font(font_size)
    for text, (x, y), color in texts:
        draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
        draw.text((x,     y    ), text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ------------------------------------------------------------------
# Legende
# ------------------------------------------------------------------

def draw_legend(
    bgr: np.ndarray,
    temp_min: float,
    temp_max: float,
    colormap_id,
) -> np.ndarray:
    """
    Zeichnet eine vertikale Colormap-Legende mit Temperaturskala
    am rechten Bildrand. Alle Größen skalieren mit der Frame-Höhe.
    """
    img_h, img_w = bgr.shape[:2]

    # Dynamische Größen
    bar_h     = int(img_h * 0.70)
    bar_w     = max(14, int(img_h * 0.025))
    margin    = max(8,  int(img_h * 0.015))
    font_size = max(11, int(img_h * 0.022))

    # Farbbalken generieren
    gradient = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(bar_h, 1)
    if colormap_id is not None:
        bar_bgr = cv2.applyColorMap(gradient, colormap_id)
    else:
        bar_bgr = np.stack([gradient] * 3, axis=-1)
    bar_bgr = np.repeat(bar_bgr, bar_w, axis=1)

    # Position: rechts, vertikal zentriert
    x0 = img_w - margin - bar_w
    y0 = (img_h - bar_h) // 2
    y1 = y0 + bar_h

    if y0 < 0 or y1 > img_h or x0 < margin:
        return bgr

    # Balken halbtransparent einfügen
    roi = bgr[y0:y1, x0:x0 + bar_w]
    bgr[y0:y1, x0:x0 + bar_w] = cv2.addWeighted(roi, 0.25, bar_bgr, 0.75, 0)

    # Dünner Rahmen
    cv2.rectangle(bgr,
                  (x0 - 1, y0 - 1),
                  (x0 + bar_w, y1),
                  (200, 200, 200), 1)

    # Beschriftung via Pillow
    font_lbl = get_font(font_size)
    rgb      = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil      = Image.fromarray(rgb)
    draw     = ImageDraw.Draw(pil)
    text_x   = x0 - margin - 2

    steps = 5
    for i in range(steps):
        frac  = i / (steps - 1)
        temp  = temp_max - frac * (temp_max - temp_min)
        py    = y0 + int(frac * (bar_h - 1))
        label = f"{temp:.1f} °C"

        bbox = draw.textbbox((0, 0), label, font=font_lbl)
        tw   = bbox[2] - bbox[0]
        tx   = text_x - tw
        ty   = py - font_size // 2

        draw.text((tx + 1, ty + 1), label, font=font_lbl, fill=(0, 0, 0))
        draw.text((tx,     ty    ), label, font=font_lbl, fill=(255, 255, 255))

        cv2.line(bgr, (x0 - 1, py), (x0 - margin + 2, py), (200, 200, 200), 1)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ------------------------------------------------------------------
# Kamera-Thread
# ------------------------------------------------------------------

class CameraThread(QThread):
    frame_ready = Signal(object)

    def __init__(self, cap):
        super().__init__()
        self.cap     = cap
        self.running = True

    def run(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    self.frame_ready.emit(frame)
                else:
                    self.msleep(10)
            except cv2.error:
                self.msleep(50)

    def stop(self):
        self.running = False
        self.wait()

# ------------------------------------------------------------------
# Kamera-Widget
# ------------------------------------------------------------------

class CameraWidget(QWidget):
    clipboard_copied = Signal()

    def __init__(self, camera_index: int, tc001_index: int | None, main_window=None):
        super().__init__()
        self.camera_index = camera_index
        self.tc001_index  = tc001_index
        self.main_window  = main_window

        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        self.frozen        = False
        self.frozen_frame  = None
        self.current_frame = None

        self.colormap_id = cv2.COLORMAP_TURBO
        self.rotation    = 0  # 0 / 90 / 180 / 270

        self.show_center = False
        self.show_max    = False
        self.show_min    = False
        self.show_legend = False          # <-- neu

        self.recorder  = None
        self.recording = False

        self.selecting             = False
        self.selection_start       = QPoint()
        self.selection_end         = QPoint()
        self.selection_rect        = QRect()
        self.selection_clear_timer = QTimer()
        self.selection_clear_timer.setSingleShot(True)
        self.selection_clear_timer.timeout.connect(self.clear_selection)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(640, 480)

        self.thread = CameraThread(self.cap)
        self.thread.frame_ready.connect(self.on_frame_ready)
        self.thread.start()

    # ----------------------------------------------------------------
    # Frame-Verarbeitung
    # ----------------------------------------------------------------

    def on_frame_ready(self, frame):
        if not self.frozen:
            self.current_frame = frame
            self.update()

    def _is_tc001_frame(self, frame) -> bool:
        h, w = frame.shape[:2]
        return w == TC001_W and h == TC001_H * 2

    def _rotate(self, bgr: np.ndarray) -> np.ndarray:
        if self.rotation == 90:
            return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            return cv2.rotate(bgr, cv2.ROTATE_180)
        elif self.rotation == 270:
            return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return bgr

    def _process_tc001(self, raw_frame):
        """Gibt (bgr_display, temp_array_celsius) zurück."""
        img_yuv = raw_frame[:TC001_H]
        bgr     = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR_YUYV)
        gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 1. Zielgröße ermitteln
        target_w     = self.width()
        target_h     = self.height()
        src_h, src_w = gray.shape[:2]
        scale        = min(target_w / src_w, target_h / src_h)
        disp_w       = int(src_w * scale)
        disp_h       = int(src_h * scale)

        # 2. ZUERST das Graustufenbild BIKUBISCH vergrößern:
        gray_scaled = cv2.resize(gray, (disp_w, disp_h),
                                 interpolation=cv2.INTER_CUBIC)

        # 3. ERST JETZT die Farbpalette auf das hochauflösende Bild anwenden:
        if self.colormap_id is not None:
            colored = cv2.applyColorMap(gray_scaled, self.colormap_id)
        else:
            colored = cv2.cvtColor(gray_scaled, cv2.COLOR_GRAY2BGR)

        colored = self._rotate(colored)
        temp_C  = raw_to_celsius(raw_frame)
        return colored, temp_C

    def _draw_hud(self, bgr: np.ndarray, temp_C: np.ndarray) -> np.ndarray:
        h, w      = bgr.shape[:2]
        texts     = []
        font_size = max(11, int(h * 0.022))  # identisch mit draw_legend

        if self.show_center:
            cx, cy = w // 2, h // 2
            arm = 14
            cv2.line(bgr, (cx - arm, cy), (cx + arm, cy), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(bgr, (cx, cy - arm), (cx, cy + arm), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(bgr, (cx, cy), 4, (255, 255, 255), 1, cv2.LINE_AA)
            t = temp_C[TC001_H // 2, TC001_W // 2]
            texts.append((f"{t:.1f} °C", (cx + 8, cy - 24), (255, 255, 255)))

        if self.show_max:
            idx    = np.unravel_index(np.argmax(temp_C), temp_C.shape)
            my, mx = idx
            px = int(mx / TC001_W * w)
            py = int(my / TC001_H * h)
            cv2.drawMarker(bgr, (px, py), (0, 80, 255), cv2.MARKER_TRIANGLE_UP, 12, 1)
            texts.append((
                f"max {temp_C[my, mx]:.1f} °C",
                (min(px + 6, w - 130), max(py - 6, 2)),
                (255, 80, 0),
            ))

        if self.show_min:
            idx    = np.unravel_index(np.argmin(temp_C), temp_C.shape)
            my, mx = idx
            px = int(mx / TC001_W * w)
            py = int(my / TC001_H * h)
            cv2.drawMarker(bgr, (px, py), (255, 80, 0), cv2.MARKER_TRIANGLE_DOWN, 12, 1)
            texts.append((
                f"min {temp_C[my, mx]:.1f} °C",
                (min(px + 6, w - 130), min(py + 18, h - 22)),
                (0, 80, 255),
            ))

        if texts:
            bgr = draw_hud_text(bgr, texts, font_size)
        return bgr

    def get_display_frame(self):
        frame = self.frozen_frame if self.frozen else self.current_frame
        if frame is None:
            return None

        if self._is_tc001_frame(frame):
            bgr, temp_C = self._process_tc001(frame)

            if self.show_center or self.show_max or self.show_min:
                bgr = self._draw_hud(bgr, temp_C)

            if self.show_legend:                                      # <-- neu
                bgr = draw_legend(bgr,                                # <-- neu
                                  float(temp_C.min()),                # <-- neu
                                  float(temp_C.max()),                # <-- neu
                                  self.colormap_id)                   # <-- neu

            return bgr
        else:
            bgr   = self._rotate(frame)
            h, w  = bgr.shape[:2]
            scale = min(self.width() / w, self.height() / h)
            return cv2.resize(bgr, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_LINEAR)

    # ---------------------------------------------------------------- #
    # Aufzeichnung                                                       #
    # ---------------------------------------------------------------- #

    REC_W = TC001_W * 3   # 768 – feste Aufnahmeauflösung
    REC_H = TC001_H * 3   # 576

    def _render_for_recording(self, raw_frame) -> np.ndarray | None:
        """Rendert einen Frame in fester Aufnahmeauflösung (unabhängig von Widget-Größe)."""
        if not self._is_tc001_frame(raw_frame):
            return None

        # 1. Helligkeits-/Graustufenbild extrahieren
        img_yuv = raw_frame[:TC001_H]
        bgr     = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR_YUYV)
        gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 2. ZUERST das Graustufenbild BIKUBISCH auf die Aufnahmeauflösung vergrößern
        # Hinweis: Vor der Drehung liegt das Sensorbild im Querformat (4:3) vor.
        # Skaliert man es auf (REC_W, REC_H) [768x576] und dreht es danach um 90°/270°,
        # hat das finale Bild genau die Maße (REC_H, REC_W) [576x768], wie vom VideoWriter erwartet.
        gray_scaled = cv2.resize(gray, (self.REC_W, self.REC_H),
                                 interpolation=cv2.INTER_CUBIC)

        # 3. ERST JETZT die Farbpalette auf das hochauflösende Bild anwenden
        if self.colormap_id is not None:
            colored = cv2.applyColorMap(gray_scaled, self.colormap_id)
        else:
            colored = cv2.cvtColor(gray_scaled, cv2.COLOR_GRAY2BGR)

        # 4. Bild drehen
        colored = self._rotate(colored)

        # 5. Overlays (Temperaturwerte & Skala) auf das fertige Farbbild zeichnen
        temp_C = raw_to_celsius(raw_frame)
        if self.show_center or self.show_max or self.show_min:
            colored = self._draw_hud(colored, temp_C)
        if self.show_legend:
            colored = draw_legend(colored, float(temp_C.min()),
                                  float(temp_C.max()), self.colormap_id)
        return colored

    def start_recording(self) -> str:
        if self.recording:
            return ""
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"thermo_{ts}.avi"
        fourcc   = cv2.VideoWriter_fourcc(*"XVID")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        # Ausgabegröße passend zur Rotation
        if self.rotation in (90, 270):
            out_w, out_h = self.REC_H, self.REC_W
        else:
            out_w, out_h = self.REC_W, self.REC_H

        self.recorder  = cv2.VideoWriter(filename, fourcc, fps, (out_w, out_h))
        self.recording = True
        return filename

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        if self.recorder:
            self.recorder.release()
            self.recorder = None

    def _record_frame(self, bgr: np.ndarray):
        if self.recording and self.recorder:
            self.recorder.write(bgr)

    # ---------------------------------------------------------------- #
    # Zeichnen                                                           #
    # ---------------------------------------------------------------- #

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        frame = self.get_display_frame()

        if frame is not None:
            if self.recording:
                raw = self.frozen_frame if self.frozen else self.current_frame
                if raw is not None:
                    rec_bgr = self._render_for_recording(raw)  # feste Größe
                    if rec_bgr is not None:
                        self._record_frame(rec_bgr)

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qi     = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qi)
            x_off  = (self.width()  - pixmap.width())  // 2
            y_off  = (self.height() - pixmap.height()) // 2
            painter.drawPixmap(x_off, y_off, pixmap)

        if not self.selection_rect.isNull():
            pen = QPen(QColor(0, 180, 255), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.selection_rect)
            
    # ----------------------------------------------------------------
    # Maus-Events
    # ----------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.selection_clear_timer.stop()
            self.selecting       = True
            self.selection_start = event.position().toPoint()
            self.selection_end   = self.selection_start
            self.selection_rect  = QRect()
            self.update()
        elif event.button() == Qt.RightButton:
            if self.main_window:
                self.main_window.toggle_freeze()
#         elif event.button() == Qt.MiddleButton:
#             if self.main_window:
#                 self.main_window.toggle_recording()

    def mouseMoveEvent(self, event):
        if self.selecting:
            self.selection_end  = event.position().toPoint()
            self.selection_rect = QRect(
                self.selection_start, self.selection_end).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.selecting:
            self.selecting      = False
            self.selection_end  = event.position().toPoint()
            self.selection_rect = QRect(
                self.selection_start, self.selection_end).normalized()
            self.update()
            self.copy_selection_to_clipboard()
            self.selection_clear_timer.start(500)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self.main_window:
            self.main_window.toggle_fullscreen()

    # ----------------------------------------------------------------
    # Tastatur
    # ----------------------------------------------------------------

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if self.main_window:
                self.main_window.toggle_freeze()
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.main_window:
                self.main_window.toggle_recording()

    # ----------------------------------------------------------------
    # Aktionen
    # ----------------------------------------------------------------

    def clear_selection(self):
        self.selection_rect = QRect()
        self.update()

    def toggle_freeze(self):
        if not self.frozen:
            if self.current_frame is not None:
                self.frozen_frame = self.current_frame.copy()
                self.frozen = True
        else:
            self.frozen         = False
            self.frozen_frame   = None
            self.selection_rect = QRect()
            self.update()

    def switch_camera(self, index):
        self.thread.stop()
        self.cap.release()
        self.camera_index   = index
        self.frozen         = False
        self.frozen_frame   = None
        self.current_frame  = None
        self.selection_rect = QRect()
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        QThread.msleep(300)
        self.thread = CameraThread(self.cap)
        self.thread.frame_ready.connect(self.on_frame_ready)
        self.thread.start()
        self.update()

    def copy_selection_to_clipboard(self):
        if self.selection_rect.isNull():
            return
        frame = self.get_display_frame()
        if frame is None:
            return
        h, w  = frame.shape[:2]
        x_off = (self.width()  - w) // 2
        y_off = (self.height() - h) // 2
        x1 = max(0, self.selection_rect.left()   - x_off)
        y1 = max(0, self.selection_rect.top()    - y_off)
        x2 = min(w, self.selection_rect.right()  - x_off)
        y2 = min(h, self.selection_rect.bottom() - y_off)
        if x2 <= x1 or y2 <= y1:
            return
        crop = frame[y1:y2, x1:x2]
        rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ch_  = rgb.shape[2]
        qi   = QImage(rgb.data.tobytes(), x2 - x1, y2 - y1,
                      ch_ * (x2 - x1), QImage.Format_RGB888)
        QGuiApplication.clipboard().setPixmap(QPixmap.fromImage(qi))
        self.clipboard_copied.emit()

    def closeEvent(self, event):
        self.stop_recording()
        self.thread.stop()
        self.cap.release()

# ------------------------------------------------------------------
# Hauptfenster
# ------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wärmebildkamera – Tooltop T7")
        self.resize(1024, 907)

        self.available_cameras, self.tc001_index = find_cameras()
        start_index = (
            self.tc001_index if self.tc001_index is not None
            else (self.available_cameras[0] if self.available_cameras else 0)
        )

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self.camera = CameraWidget(start_index, self.tc001_index, main_window=self)
        layout.addWidget(self.camera)

        # ---- Button-Leiste ---------------------------------------
        BTN  = "height: 36px; padding: 0px 8px;"
        CBOX = "height: 36px; padding: 0px 4px;"

        row1 = QHBoxLayout()
        row1.setSpacing(6)
        row1.setContentsMargins(4, 6, 4, 6)
        layout.addLayout(row1)

        # Drehung
        self.rotation_box = QComboBox()
        self.rotation_box.setStyleSheet(BTN)
        self.rotation_box.setFixedHeight(36)
        for label in ["0°", "90°", "180°", "270°"]:
            self.rotation_box.addItem(label)
        self.rotation_box.currentIndexChanged.connect(self.change_rotation)
        row1.addWidget(self.rotation_box)

        # Colormap
        self.colormap_box = QComboBox()
        self.colormap_box.setStyleSheet(BTN)
        self.colormap_box.setFixedHeight(36)
        self._colormap_names = list(COLORMAPS.keys())
        for name in self._colormap_names:
            self.colormap_box.addItem(name)
        self.colormap_box.currentIndexChanged.connect(self.change_colormap)
        row1.addWidget(self.colormap_box)

        # Anhalten
        self.btn_freeze = QPushButton("Anhalten [Space / Rechtsklick]")
        self.btn_freeze.setStyleSheet(BTN)
        self.btn_freeze.setFixedWidth(210)
        self.btn_freeze.clicked.connect(self.toggle_freeze)
        row1.addWidget(self.btn_freeze)

        # Aufnahme
        self.btn_record = QPushButton("Aufnahme [Enter]")
        self.btn_record.setStyleSheet(BTN)
        self.btn_record.setCheckable(True)
        self.btn_record.clicked.connect(self.toggle_recording)
        row1.addWidget(self.btn_record)

        # Vollbild
        self.btn_fullscreen = QPushButton("Vollbild [F11 / Doppelklick]")
        self.btn_fullscreen.setStyleSheet(BTN)
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)
        row1.addWidget(self.btn_fullscreen)

        row1.addStretch()


        # Temperatur-Checkboxen
        self.chk_center = QCheckBox("Mitte-Temp")
        self.chk_center.setStyleSheet(CBOX)
        self.chk_center.setChecked(False)
        self.chk_center.toggled.connect(
            lambda v: setattr(self.camera, "show_center", v) or self.camera.update())
        row1.addWidget(self.chk_center)

        self.chk_max = QCheckBox("Max-Temp")
        self.chk_max.setStyleSheet(CBOX)
        self.chk_max.setChecked(False)
        self.chk_max.toggled.connect(
            lambda v: setattr(self.camera, "show_max", v) or self.camera.update())
        row1.addWidget(self.chk_max)

        self.chk_min = QCheckBox("Min-Temp")
        self.chk_min.setStyleSheet(CBOX)
        self.chk_min.setChecked(False)
        self.chk_min.toggled.connect(
            lambda v: setattr(self.camera, "show_min", v) or self.camera.update())
        row1.addWidget(self.chk_min)
        
        # Legende                                                   # <-- neu
        self.chk_legend = QCheckBox("Legende")                      # <-- neu
        self.chk_legend.setStyleSheet(CBOX)                         # <-- neu
        self.chk_legend.setChecked(False)                           # <-- neu
        self.chk_legend.toggled.connect(                            # <-- neu
            lambda v: setattr(self.camera, "show_legend", v)        # <-- neu
            or self.camera.update())                                 # <-- neu
        row1.addWidget(self.chk_legend)                             # <-- neu


    # ----------------------------------------------------------------
    # Vollbild / Drehung / Colormap
    # ----------------------------------------------------------------

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self.btn_fullscreen.setText("Vollbild [F11 / Doppelklick]")
        else:
            self.showFullScreen()
            self.btn_fullscreen.setText("Vollbild beenden [F11 / Doppelklick]")

    def change_rotation(self, index):
        self.camera.rotation = index * 90
        self.camera.update()

    def change_colormap(self, index):
        self.camera.colormap_id = COLORMAPS[self._colormap_names[index]]
        self.camera.update()

    def toggle_freeze(self):
        self.camera.toggle_freeze()
        if self.camera.frozen:
            self.btn_freeze.setText("Fortsetzen [Space / Rechtsklick]")
        else:
            self.btn_freeze.setText("Anhalten [Space / Rechtsklick]")

    # ----------------------------------------------------------------
    # Aufzeichnung
    # ----------------------------------------------------------------

    def toggle_recording(self):
        if not self.camera.recording:
            filename = self.camera.start_recording()
            self.btn_record.setChecked(True)
            self.btn_record.setText(f"Stop [{filename}] [Enter]")
            self.btn_record.setStyleSheet(
                "height: 36px; padding: 0px 8px; color: red; font-weight: bold;")
        else:
            self.camera.stop_recording()
            self.btn_record.setChecked(False)
            self.btn_record.setText("Aufnahme [Enter]")
            self.btn_record.setStyleSheet("height: 36px; padding: 0px 8px;")


    # ----------------------------------------------------------------
    # Tastatur / Schließen
    # ----------------------------------------------------------------

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            self.toggle_fullscreen()
        else:
            self.camera.keyPressEvent(event)

    def closeEvent(self, event):
        self.camera.stop_recording()
        self.camera.thread.stop()
        self.camera.cap.release()
        event.accept()

# ------------------------------------------------------------------
# Start
# ------------------------------------------------------------------

if __name__ == "__main__":
    app    = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())