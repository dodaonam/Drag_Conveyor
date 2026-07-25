from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


class FrameSourceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SourceFrame:
    source_frame_id: int
    source_timestamp_sec: float
    image_bgr: np.ndarray
    timestamp_source: str = "decoder_pts"
    is_original: bool = True


def iter_original_frames(path: str | Path, *, timestamp_epsilon_sec: float = 0.000000001) -> Iterator[SourceFrame]:
    """Yield decoded original frames in strict decoder-PTS order using PyAV."""
    if timestamp_epsilon_sec <= 0.0:
        raise ValueError("timestamp_epsilon_sec must be positive")
    try:
        import av
    except ModuleNotFoundError as exc:
        raise FrameSourceError("PyAV is required for geometry_v2 frame provenance") from exc
    source_path = Path(path)
    if not source_path.is_file():
        raise FrameSourceError(f"Video source does not exist: {source_path}")
    first_decoder_timestamp: float | None = None
    previous_timestamp: float | None = None
    frame_id = 0
    try:
        with av.open(str(source_path)) as container:
            stream = next((stream for stream in container.streams if stream.type == "video"), None)
            if stream is None:
                raise FrameSourceError("Video source has no video stream")
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    raise FrameSourceError("Video frame has no decoder PTS")
                decoder_timestamp = float(frame.pts * frame.time_base)
                if first_decoder_timestamp is None:
                    first_decoder_timestamp = decoder_timestamp
                timestamp = decoder_timestamp - first_decoder_timestamp
                if previous_timestamp is not None and timestamp - previous_timestamp <= timestamp_epsilon_sec:
                    raise FrameSourceError("Decoder PTS must strictly increase for original frames")
                previous_timestamp = timestamp
                frame_id += 1
                yield SourceFrame(frame_id, timestamp, frame.to_ndarray(format="bgr24"))
    except FrameSourceError:
        raise
    except Exception as exc:
        raise FrameSourceError(f"Unable to decode video source: {source_path}") from exc
