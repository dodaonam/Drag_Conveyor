from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..inference import Detection


@dataclass(slots=True)
class TrackedObject:
    track_id: int
    prev_centroid_xy: tuple[float, float] | None
    centroid_xy: tuple[float, float]
    hits: int
    missed_frames: int
    state: str
    confirmed: bool
    area: float
    detection: Detection


class CentroidTracker:
    def __init__(
        self,
        max_jump_px: float,
        ttl_frames: int,
        min_hits: int,
        max_reverse_px: float = 5.0,
        max_area_ratio_change: float = 3.0,
    ) -> None:
        self.max_jump_px = max_jump_px
        self.ttl_frames = ttl_frames
        self.min_hits = min_hits
        self.max_reverse_px = max_reverse_px
        self.max_area_ratio_change = max_area_ratio_change
        self._next_id = 1
        self._tracks: dict[int, TrackedObject] = {}

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        if not self._tracks:
            for det in detections:
                self._create_track(det)
            return list(self._tracks.values())

        track_ids = list(self._tracks.keys())
        unmatched_tracks = set(track_ids)
        unmatched_dets = set(range(len(detections)))

        # Greedy nearest-neighbor matching with jump constraint.
        if track_ids and detections:
            track_centroids = np.array([self._tracks[t].centroid_xy for t in track_ids], dtype=np.float32)
            det_centroids = np.array([d.centroid_frame_xy for d in detections], dtype=np.float32)

            dist = np.linalg.norm(track_centroids[:, None, :] - det_centroids[None, :, :], axis=2)
            pairs = np.dstack(np.unravel_index(np.argsort(dist.ravel()), dist.shape))[0]

            for ti, di in pairs:
                track_id = track_ids[int(ti)]
                det_idx = int(di)
                if track_id not in unmatched_tracks or det_idx not in unmatched_dets:
                    continue
                if dist[ti, di] > self.max_jump_px:
                    continue
                if not self._movement_gate(self._tracks[track_id], detections[det_idx]):
                    continue
                if not self._area_gate(self._tracks[track_id], detections[det_idx]):
                    continue

                self._update_track(track_id, detections[det_idx])
                unmatched_tracks.remove(track_id)
                unmatched_dets.remove(det_idx)

        for track_id in unmatched_tracks:
            track = self._tracks[track_id]
            track.missed_frames += 1
            track.state = "lost"
            track.confirmed = False

        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx])

        stale = [tid for tid, track in self._tracks.items() if track.missed_frames > self.ttl_frames]
        for tid in stale:
            self._tracks.pop(tid, None)

        return list(self._tracks.values())

    def _create_track(self, det: Detection) -> None:
        track_id = self._next_id
        self._next_id += 1
        initial_state = "active" if self.min_hits <= 1 else "tentative"
        self._tracks[track_id] = TrackedObject(
            track_id=track_id,
            prev_centroid_xy=None,
            centroid_xy=det.centroid_frame_xy,
            hits=1,
            missed_frames=0,
            state=initial_state,
            confirmed=self.min_hits <= 1,
            area=self._detection_area(det),
            detection=det,
        )

    def _update_track(self, track_id: int, det: Detection) -> None:
        track = self._tracks[track_id]
        track.prev_centroid_xy = track.centroid_xy
        track.centroid_xy = det.centroid_frame_xy
        track.hits += 1
        track.missed_frames = 0
        track.state = "active" if track.hits >= self.min_hits else "tentative"
        track.confirmed = track.hits >= self.min_hits
        track.area = self._detection_area(det)
        track.detection = det

    def _movement_gate(self, track: TrackedObject, det: Detection) -> bool:
        prev_y = track.centroid_xy[1]
        curr_y = det.centroid_frame_xy[1]
        return curr_y + self.max_reverse_px >= prev_y

    def _area_gate(self, track: TrackedObject, det: Detection) -> bool:
        prev_area = max(track.area, 1.0)
        curr_area = max(self._detection_area(det), 1.0)
        ratio = max(prev_area, curr_area) / min(prev_area, curr_area)
        return ratio <= self.max_area_ratio_change

    def _detection_area(self, det: Detection) -> float:
        x1, y1, x2, y2 = det.bbox_frame_xyxy
        w = max(0.0, float(x2 - x1))
        h = max(0.0, float(y2 - y1))
        return w * h


__all__ = ["CentroidTracker", "TrackedObject"]
