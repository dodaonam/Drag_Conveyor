from __future__ import annotations

from PySide6 import QtWidgets


class ResultPanel(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self.fps_label = QtWidgets.QLabel("FPS: 0.00")

        self.processed_label = QtWidgets.QLabel("Processed: 0")
        self.normal_label = QtWidgets.QLabel("Normal: 0")
        self.defect_label = QtWidgets.QLabel("Defect: 0")
        self.message_label = QtWidgets.QLabel("")
        self.message_label.setWordWrap(True)

        for widget in [
            self.fps_label,
            self.processed_label,
            self.normal_label,
            self.defect_label,
            self.message_label,
        ]:
            layout.addWidget(widget)
        layout.addStretch(1)

    def set_message(self, text: str) -> None:
        self.message_label.setText(text)

    def set_state(self, state: str) -> None:
        pass

    def set_camera(self, message: str) -> None:
        pass

    def set_model(self, message: str) -> None:
        pass

    def set_worker_states(self, infer: str, pipeline: str, logger: str) -> None:
        pass

    def set_perf(
        self,
        throughput_fps: float,
        inference_fps_estimate: float,
        avg_latency_ms: float,
        p95_latency_ms: float,
    ) -> None:
        self.fps_label.setText(f"FPS: {throughput_fps:.2f}")

    def set_counters(self, processed: int, normal: int, defect: int) -> None:
        self.processed_label.setText(f"Processed: {processed}")
        self.normal_label.setText(f"Normal: {normal}")
        self.defect_label.setText(f"Defect: {defect}")
