from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import cv2  # noqa: E402

import preprocess  # noqa: E402


def _write_video(path: Path, n: int, w: int = 120, h: int = 90, fps: int = 30) -> bool:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        return False
    for i in range(n):
        frame = np.zeros((h, w, 3), np.uint8)
        y = 10 + i * 6
        cv2.rectangle(frame, (40, y), (80, y + 12), (0, 0, 255), -1)  # a bar moving down
        writer.write(frame)
    writer.release()
    return path.exists() and path.stat().st_size > 0


def _frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


class SlowDownVideoTests(unittest.TestCase):
    def test_disabled_when_factor_ge_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.mp4"
            if not _write_video(src, 6):
                self.skipTest("VideoWriter (mp4v) not available")
            out = preprocess.slow_down_video(src, Path(tmp) / "out.mp4", factor=1.0)
            self.assertEqual(out, str(src))  # unchanged, no new file

    def test_slowdown_adds_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.mp4"
            dst = Path(tmp) / "out.mp4"
            if not _write_video(src, 12):
                self.skipTest("VideoWriter (mp4v) not available")
            n_in = _frame_count(str(src))
            out = preprocess.slow_down_video(src, dst, factor=0.75)
            self.assertEqual(out, str(dst))
            n_out = _frame_count(out)
            self.assertGreater(n_out, n_in)  # genuinely slowed down

    def test_first_frame_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.mp4"
            dst = Path(tmp) / "out.mp4"
            if not _write_video(src, 8):
                self.skipTest("VideoWriter (mp4v) not available")
            out = preprocess.slow_down_video(src, dst, factor=0.75)
            cap_in = cv2.VideoCapture(str(src))
            cap_out = cv2.VideoCapture(out)
            ok_in, f_in = cap_in.read()
            ok_out, f_out = cap_out.read()
            cap_in.release()
            cap_out.release()
            self.assertTrue(ok_in and ok_out)
            # Output frame 0 maps to source position 0 → copied verbatim (allow codec noise).
            self.assertEqual(f_in.shape, f_out.shape)
            self.assertLess(float(np.abs(f_in.astype(int) - f_out.astype(int)).mean()), 5.0)


if __name__ == "__main__":
    unittest.main()
