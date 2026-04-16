import re
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import config.settings as app_settings


class SettingsPage(QWidget):
    def __init__(self):
        super().__init__()
        self.controls = {}

        self.setStyleSheet(
            """
            QWidget {
                color: #e2e8f0;
                background-color: transparent;
            }
            QGroupBox {
                font-weight: 700;
                color: #cbd5e1;
                background-color: #0b1224;
                border: 1px solid #334155;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px 0 4px;
                color: #94a3b8;
                background-color: #0b1224;
            }
            QLabel {
                color: #cbd5e1;
                background-color: transparent;
            }
            QCheckBox {
                color: #e2e8f0;
                spacing: 8px;
                background-color: transparent;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #64748b;
                border-radius: 4px;
                background-color: #0f172a;
            }
            QCheckBox::indicator:checked {
                background-color: #0ea5e9;
                border: 1px solid #0284c7;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #0f172a;
                color: #e2e8f0;
                border: 1px solid #475569;
                border-radius: 6px;
                padding: 4px 6px;
                min-width: 90px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #1e293b;
                height: 6px;
                border-radius: 3px;
                background: #1e293b;
            }
            QSlider::handle:horizontal {
                background: #38bdf8;
                border: 1px solid #0284c7;
                width: 12px;
                margin: -5px 0;
                border-radius: 6px;
            }
            QPushButton {
                border-radius: 8px;
                padding: 8px 14px;
                font-weight: 700;
                outline: none;
            }
            QPushButton:focus {
                outline: none;
            }
            """
        )

        root = QVBoxLayout()
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)
        self.setLayout(root)

        title = QLabel("System Settings")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        root.addWidget(title)

        note = QLabel(
            "Adjust key camera settings, then save to persist into config/settings.py.\n"
            "Restart the app after saving to ensure all modules use updated values."
        )
        note.setStyleSheet("color: #94a3b8;")
        root.addWidget(note)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #94a3b8;")
        root.addWidget(self.status_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea { background-color: #0f172a; border: none; }"
            "QScrollArea > QWidget > QWidget { background-color: #0f172a; }"
        )
        root.addWidget(scroll, 1)

        content = QWidget()
        scroll.setWidget(content)
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)
        content.setLayout(content_layout)

        content_layout.addWidget(self._build_default_toggles_group())
        content_layout.addWidget(self._build_performance_group())
        content_layout.addWidget(self._build_detection_group())
        content_layout.addWidget(self._build_face_group())
        content_layout.addWidget(self._build_emergency_group())
        content_layout.addStretch()

        actions = QHBoxLayout()
        actions.addStretch()
        root.addLayout(actions)

        self.reset_btn = QPushButton("Reset to Default")
        self.reset_btn.setFocusPolicy(Qt.NoFocus)
        self.reset_btn.setStyleSheet(
            "QPushButton { background-color: #7f1d1d; color: #fee2e2; border: 1px solid #991b1b; }"
            "QPushButton:hover { background-color: #991b1b; }"
        )
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        actions.addWidget(self.reset_btn)

        self.save_btn = QPushButton("Save")
        self.save_btn.setFocusPolicy(Qt.NoFocus)
        self.save_btn.setStyleSheet(
            "QPushButton { background-color: #15803d; color: #ffffff; border: 1px solid #166534; }"
            "QPushButton:hover { background-color: #166534; }"
        )
        self.save_btn.clicked.connect(self.save_settings)
        actions.addWidget(self.save_btn)

    def _build_default_toggles_group(self):
        group = QGroupBox("Default Detection Toggles")
        layout = QFormLayout()
        layout.setLabelAlignment(Qt.AlignLeft)
        layout.setFormAlignment(Qt.AlignTop)
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(8)
        group.setLayout(layout)

        self._add_bool(layout, "Intrusion", "DEFAULT_INTRUSION")
        self._add_bool(layout, "Crowd", "DEFAULT_CROWD")
        self._add_bool(layout, "Vehicle", "DEFAULT_VEHICLE")
        self._add_bool(layout, "Threat", "DEFAULT_THREAT")
        self._add_bool(layout, "Motion", "DEFAULT_MOTION")
        self._add_bool(layout, "Loitering", "DEFAULT_LOITER")
        self._add_bool(layout, "Emergency", "DEFAULT_EMERGENCY")
        self._add_bool(layout, "Face Recognition", "DEFAULT_FACE_RECOGNITION")
        return group

    def _build_performance_group(self):
        group = QGroupBox("Camera & Performance")
        layout = QFormLayout()
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(8)
        group.setLayout(layout)

        self._add_int(layout, "Camera Index", "CAMERA_INDEX", 0, 8)
        self._add_int(layout, "Frame Interval (ms)", "FRAME_INTERVAL_MS", 15, 250, slider=True)
        self._add_int(layout, "YOLO Frame Skip", "YOLO_INFERENCE_FRAME_SKIP", 1, 12, slider=True)
        return group

    def _build_detection_group(self):
        group = QGroupBox("Detection Tuning")
        layout = QFormLayout()
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(8)
        group.setLayout(layout)

        self._add_float(layout, "YOLO Min Confidence", "YOLO_MIN_CONFIDENCE", 0.1, 0.95, 0.01, slider=True)
        self._add_int(layout, "Crowd Threshold", "CROWD_THRESHOLD", 1, 20, slider=True)
        self._add_float(layout, "Loiter Dwell (sec)", "LOITER_DWELL_SECONDS", 2.0, 60.0, 0.5, slider=True)
        self._add_float(layout, "Threat Min Confidence", "THREAT_MIN_CONFIDENCE", 0.1, 0.95, 0.01, slider=True)
        self._add_int(layout, "Motion Threshold", "MOTION_THRESHOLD", 5, 100, slider=True)
        self._add_int(layout, "Motion Min Area", "MOTION_MIN_AREA", 100, 10000)
        return group

    def _build_face_group(self):
        group = QGroupBox("Face Recognition")
        layout = QFormLayout()
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(8)
        group.setLayout(layout)

        self._add_int(layout, "Recognition Frame Skip", "FACE_RECOGNITION_FRAME_SKIP", 1, 12, slider=True)
        self._add_float(
            layout,
            "Recognition Threshold",
            "FACE_RECOGNITION_THRESHOLD",
            0.5,
            0.99,
            0.01,
            slider=True,
        )
        self._add_int(layout, "Register Samples Required", "FACE_REGISTER_SAMPLES_REQUIRED", 1, 20, slider=True)
        return group

    def _build_emergency_group(self):
        group = QGroupBox("Emergency")
        layout = QFormLayout()
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(8)
        group.setLayout(layout)

        self._add_int(
            layout,
            "Emergency Reset Timeout (sec)",
            "EMERGENCY_RESET_TIMEOUT_SECONDS",
            2,
            30,
            slider=True,
        )
        return group

    def _current_value(self, key):
        return getattr(app_settings, key)

    def _add_bool(self, layout, label, key):
        box = QCheckBox("Enabled")
        box.setChecked(bool(self._current_value(key)))
        box.setFocusPolicy(Qt.NoFocus)
        self.controls[key] = ("bool", box)
        layout.addRow(QLabel(label), box)

    def _add_int(self, layout, label, key, min_v, max_v, slider=False):
        spin = QSpinBox()
        spin.setRange(min_v, max_v)
        spin.setSingleStep(1)
        spin.setValue(int(self._current_value(key)))

        row = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(10)
        row.setLayout(row_layout)
        row_layout.addWidget(spin)

        slider_widget = None
        if slider:
            slider_widget = QSlider(Qt.Horizontal)
            slider_widget.setRange(min_v, max_v)
            slider_widget.setValue(spin.value())
            slider_widget.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider_widget.setValue)
            row_layout.addWidget(slider_widget, 1)

        self.controls[key] = ("int", spin)
        layout.addRow(QLabel(label), row)

    def _add_float(self, layout, label, key, min_v, max_v, step, slider=False):
        decimals = 2
        scale = 100

        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(float(self._current_value(key)))

        row = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(10)
        row.setLayout(row_layout)
        row_layout.addWidget(spin)

        if slider:
            slider_widget = QSlider(Qt.Horizontal)
            slider_widget.setRange(int(min_v * scale), int(max_v * scale))
            slider_widget.setValue(int(round(spin.value() * scale)))
            slider_widget.valueChanged.connect(lambda value: spin.setValue(value / scale))
            spin.valueChanged.connect(lambda value: slider_widget.setValue(int(round(value * scale))))
            row_layout.addWidget(slider_widget, 1)

        self.controls[key] = ("float", spin)
        layout.addRow(QLabel(label), row)

    def _collect_values(self):
        values = {}
        for key, (kind, widget) in self.controls.items():
            if kind == "bool":
                values[key] = bool(widget.isChecked())
            elif kind == "int":
                values[key] = int(widget.value())
            elif kind == "float":
                values[key] = float(widget.value())
        return values

    def _set_status(self, text, ok=True):
        color = "#86efac" if ok else "#fca5a5"
        self.status_label.setStyleSheet(f"color: {color};")
        self.status_label.setText(text)

    @staticmethod
    def _format_value(value):
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, float):
            return f"{value:.2f}".rstrip("0").rstrip(".")
        return str(value)

    def _write_values_to_settings_file(self, values):
        settings_path = Path(app_settings.__file__)
        text = settings_path.read_text(encoding="utf-8")

        editable = set(getattr(app_settings, "SETTINGS_EDITABLE_KEYS", ()))
        for key, value in values.items():
            if key not in editable:
                continue
            pattern = rf"(?m)^{re.escape(key)}\s*=\s*.*$"
            replacement = f"{key} = {self._format_value(value)}"
            text, count = re.subn(pattern, replacement, text, count=1)
            if count == 0:
                raise ValueError(f"Setting key not found in file: {key}")

        settings_path.write_text(text, encoding="utf-8")

    def save_settings(self):
        values = self._collect_values()
        try:
            self._write_values_to_settings_file(values)
        except Exception as exc:
            self._set_status(f"Save failed: {exc}", ok=False)
            return
        self._set_status("Saved. Restart app to apply everywhere.", ok=True)

    def reset_to_defaults(self):
        defaults = dict(getattr(app_settings, "SETTINGS_FACTORY_DEFAULTS", {}))
        if not defaults:
            self._set_status("No defaults configured.", ok=False)
            return

        for key, value in defaults.items():
            control = self.controls.get(key)
            if not control:
                continue
            kind, widget = control
            if kind == "bool":
                widget.setChecked(bool(value))
            elif kind == "int":
                widget.setValue(int(value))
            elif kind == "float":
                widget.setValue(float(value))

        try:
            self._write_values_to_settings_file(defaults)
        except Exception as exc:
            self._set_status(f"Reset failed: {exc}", ok=False)
            return
        self._set_status("Reset to defaults. Restart app to apply everywhere.", ok=True)
