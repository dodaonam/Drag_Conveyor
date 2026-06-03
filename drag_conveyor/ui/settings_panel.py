from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..config import Profile


class SettingsPanel(QtWidgets.QWidget):
    selectVideoRequested = QtCore.Signal()
    scanCameraRequested = QtCore.Signal()
    connectCameraRequested = QtCore.Signal()
    saveProfileRequested = QtCore.Signal()
    loadProfileRequested = QtCore.Signal()
    calibrateRequested = QtCore.Signal()
    startDetectionRequested = QtCore.Signal()
    pauseDetectionRequested = QtCore.Signal()
    resumeDetectionRequested = QtCore.Signal()
    stopDetectionRequested = QtCore.Signal()
    freezeToggled = QtCore.Signal(bool)
    controlsChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(QtWidgets.QLabel("Nguon demo/camera"))
        self.source_edit = QtWidgets.QLineEdit()
        self.source_edit.setPlaceholderText("Duong dan video hoac camera index")
        layout.addWidget(self.source_edit)
        self.model_status_label = QtWidgets.QLabel("Model: Chua tai")
        layout.addWidget(self.model_status_label)

        self.btn_select_video = QtWidgets.QPushButton("Chon video demo")
        self.btn_scan = QtWidgets.QPushButton("Quet camera")
        self.btn_connect = QtWidgets.QPushButton("Ket noi source")
        self.btn_save = QtWidgets.QPushButton("Luu profile")
        self.btn_load = QtWidgets.QPushButton("Nap profile")
        for button in [self.btn_select_video, self.btn_scan, self.btn_connect, self.btn_save, self.btn_load]:
            layout.addWidget(button)

        self.freeze_check = QtWidgets.QCheckBox("Freeze frame setup")
        layout.addWidget(self.freeze_check)
        layout.addWidget(self._separator())

        self.btn_calibrate = QtWidgets.QPushButton("Lay chuan tu dong")
        self.btn_start = QtWidgets.QPushButton("Start Detection")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_resume = QtWidgets.QPushButton("Resume")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        for button in [self.btn_calibrate, self.btn_start, self.btn_pause, self.btn_resume, self.btn_stop]:
            layout.addWidget(button)
        layout.addStretch(1)

        self.btn_select_video.clicked.connect(self.selectVideoRequested)
        self.btn_scan.clicked.connect(self.scanCameraRequested)
        self.btn_connect.clicked.connect(self.connectCameraRequested)
        self.btn_save.clicked.connect(self.saveProfileRequested)
        self.btn_load.clicked.connect(self.loadProfileRequested)
        self.btn_calibrate.clicked.connect(self.calibrateRequested)
        self.btn_start.clicked.connect(self.startDetectionRequested)
        self.btn_pause.clicked.connect(self.pauseDetectionRequested)
        self.btn_resume.clicked.connect(self.resumeDetectionRequested)
        self.btn_stop.clicked.connect(self.stopDetectionRequested)

        self.freeze_check.toggled.connect(self.freezeToggled)
        self.source_edit.editingFinished.connect(self.controlsChanged)

    def _separator(self) -> QtWidgets.QFrame:
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        return line

    def set_profile(self, profile: Profile) -> None:
        self.source_edit.setText(str(profile.camera.index))

    def source(self) -> str:
        return self.source_edit.text().strip()

    def set_source(self, source: str) -> None:
        self.source_edit.setText(source)

    def set_model_status(self, text: str) -> None:
        self.model_status_label.setText(f"Model: {text}")

    def freeze_enabled(self) -> bool:
        return self.freeze_check.isChecked()
