from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Profile
from ..inference import Detection
from ..logging.core import LogPayload, LoggerWorker
from .measure import measure_contour
from .rules import RuleEngine
from .tracking import CentroidTracker
from .trigger import BandRect, TriggerEngine, build_trigger_band, mask_overlap_ratio_with_band


@dataclass(frozen=True, slots=True)
class CounterSnapshot:
    total_processed_bars: int
    normal_bars: int
    suspected_defect_bars: int
    current_tracked_bars: int


@dataclass(frozen=True, slots=True)
class TrackOverlay:
    track_id: int
    confirmed: bool
    missed_frames: int
    bbox_frame_xyxy: tuple[float, float, float, float]
    centroid_frame_xy: tuple[float, float]
    mask_roi: np.ndarray
    contour_frame: np.ndarray


@dataclass(frozen=True, slots=True)
class TriggerEventOverlay:
    frame_id: int
    track_id: int
    result: str
    score: float
    reasons: list[str]
    bbox_frame_xyxy: tuple[float, float, float, float]


class PipelineCore:
    """Postprocess + tracking + trigger + measure + rules + logging."""

    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self.rule_engine = RuleEngine()
        self.tracker = CentroidTracker(
            max_jump_px=profile.tracker.max_jump_px,
            ttl_frames=profile.tracker.ttl_frames,
            min_hits=profile.tracker.min_hits,
            direction=profile.inspection_region.direction,
        )
        self.band: BandRect = build_trigger_band(profile.inspection_region)
        self.trigger_engine = TriggerEngine(
            pending_ttl_frames=profile.inspection_region.trigger_band.pending_ttl_frames,
            allow_inside_band_trigger=profile.inspection_region.trigger_band.allow_inside_band_trigger,
        )

        self.total_processed_bars = 0
        self.normal_bars = 0
        self.suspected_defect_bars = 0
        self._latest_tracks: list[TrackOverlay] = []
        self._latest_events: list[TriggerEventOverlay] = []

    def reset(self) -> None:
        self.tracker.reset()
        self.trigger_engine.reset()
        self.total_processed_bars = 0
        self.normal_bars = 0
        self.suspected_defect_bars = 0
        self._latest_tracks = []
        self._latest_events = []

    def get_latest_tracks(self) -> list[TrackOverlay]:
        return [
            TrackOverlay(
                track_id=item.track_id,
                confirmed=item.confirmed,
                missed_frames=item.missed_frames,
                bbox_frame_xyxy=item.bbox_frame_xyxy,
                centroid_frame_xy=item.centroid_frame_xy,
                mask_roi=item.mask_roi.copy(),
                contour_frame=item.contour_frame.copy(),
            )
            for item in self._latest_tracks
        ]

    def get_latest_events(self) -> list[TriggerEventOverlay]:
        return [
            TriggerEventOverlay(
                frame_id=item.frame_id,
                track_id=item.track_id,
                result=item.result,
                score=item.score,
                reasons=list(item.reasons),
                bbox_frame_xyxy=item.bbox_frame_xyxy,
            )
            for item in self._latest_events
        ]

    def process(
        self,
        *,
        run_id: str,
        frame_id: int,
        frame: np.ndarray,
        detections: list[Detection],
        latency_ms: float,
        logger: LoggerWorker | None,
    ) -> CounterSnapshot:
        tracks = self.tracker.update(detections)
        self.trigger_engine.begin_frame()
        current_tracked_bars = sum(1 for t in tracks if t.confirmed and t.missed_frames == 0)

        region = self.profile.inspection_region
        roi_rect = (region.x, region.y, region.x + region.w, region.y + region.h)
        frame_events: list[TriggerEventOverlay] = []

        for track in tracks:
            if not track.confirmed or track.missed_frames != 0:
                continue
            if self.trigger_engine.is_processed(track.track_id):
                continue

            overlap_ratio = mask_overlap_ratio_with_band(
                mask_roi=track.detection.mask_roi,
                roi_origin_xy=(region.x, region.y),
                band=self.band,
            )
            should_trigger = self.trigger_engine.should_trigger(
                track_id=track.track_id,
                prev_xy=track.prev_centroid_xy,
                curr_xy=track.centroid_xy,
                direction=region.direction,
                centerline=self.band.centerline,
                band=self.band,
                overlap_ratio=overlap_ratio,
                min_overlap_ratio=region.trigger_band.min_overlap_ratio,
            )
            if not should_trigger:
                continue

            if self.profile.calibration_result is None:
                # No calibration => do not trigger final decision.
                continue

            measurements = measure_contour(track.detection.contour_frame)
            evaluation = self.rule_engine.evaluate(
                measurements=measurements.to_dict(),
                rules=self.profile.rules,
                calibration_result=self.profile.calibration_result,
            )

            self.trigger_engine.mark_processed(track.track_id)
            self.total_processed_bars += 1
            if evaluation.result == "suspected_defect":
                self.suspected_defect_bars += 1
            else:
                self.normal_bars += 1

            frame_events.append(
                TriggerEventOverlay(
                    frame_id=frame_id,
                    track_id=track.track_id,
                    result=evaluation.result,
                    score=evaluation.score,
                    reasons=list(evaluation.reasons),
                    bbox_frame_xyxy=track.detection.bbox_frame_xyxy,
                )
            )

            if logger is not None:
                inference_fps_estimate = 1000.0 / latency_ms if latency_ms > 0 else 0.0
                payload = LogPayload(
                    run_id=run_id,
                    frame_id=frame_id,
                    track_id=track.track_id,
                    result=evaluation.result,
                    score=evaluation.score,
                    reasons=evaluation.reasons,
                    measurements=evaluation.measurements,
                    inference_fps_estimate=inference_fps_estimate,
                    latency_ms=latency_ms,
                    bbox_frame_xyxy=track.detection.bbox_frame_xyxy,
                    roi_rect_xyxy=roi_rect,
                    trigger_band_xyxy=(self.band.x1, self.band.y1, self.band.x2, self.band.y2),
                    thresholds=evaluation.thresholds,
                    margins=evaluation.margins,
                    hard_fail=evaluation.hard_fail,
                    violated_soft_rules=evaluation.violated_soft_rules,
                    contour_frame=track.detection.contour_frame,
                    snapshot_frame=frame,
                )
                logger.enqueue(payload)

        self._latest_tracks = [
            TrackOverlay(
                track_id=track.track_id,
                confirmed=track.confirmed,
                missed_frames=track.missed_frames,
                bbox_frame_xyxy=track.detection.bbox_frame_xyxy,
                centroid_frame_xy=track.centroid_xy,
                mask_roi=track.detection.mask_roi.copy(),
                contour_frame=track.detection.contour_frame.copy(),
            )
            for track in tracks
            if track.confirmed and track.missed_frames == 0
        ]
        self._latest_events = frame_events

        return CounterSnapshot(
            total_processed_bars=self.total_processed_bars,
            normal_bars=self.normal_bars,
            suspected_defect_bars=self.suspected_defect_bars,
            current_tracked_bars=current_tracked_bars,
        )
