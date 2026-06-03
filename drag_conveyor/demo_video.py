from __future__ import annotations

from dataclasses import dataclass

from .camera_io import VideoSourceInfo, is_camera_index_source, probe_video_source
from .config import Profile


@dataclass(frozen=True, slots=True)
class ProfileVideoSyncResult:
    profile: Profile
    calibration_cleared: bool
    roi_reset: bool


def is_video_file_source(source: str) -> bool:
    return bool(source.strip()) and not is_camera_index_source(source)


def sync_profile_to_video(profile: Profile, info: VideoSourceInfo) -> ProfileVideoSyncResult:
    updated = profile.clone()
    region = updated.inspection_region

    new_frame_size = (int(info.width), int(info.height))
    calibration_cleared = updated.calibration_result is not None

    updated.camera.width = int(info.width)
    updated.camera.height = int(info.height)
    updated.camera.backend = "FILE"
    updated.inspection_region.frame_width = int(info.width)
    updated.inspection_region.frame_height = int(info.height)
    updated.inspection_region.direction = "top_to_bottom"

    roi_reset = _roi_outside_frame(updated)
    if roi_reset:
        updated.inspection_region.x = 0
        updated.inspection_region.y = 0
        updated.inspection_region.w = int(info.width)
        updated.inspection_region.h = int(info.height)
        updated.inspection_region.trigger_band.position_ratio = 0.73
        updated.inspection_region.trigger_band.thickness_ratio = min(
            1.0,
            max(0.10, float(updated.inspection_region.trigger_band.thickness_ratio)),
        )

    if updated.calibration_result is not None:
        updated.calibration_result = None

    return ProfileVideoSyncResult(
        profile=updated,
        calibration_cleared=calibration_cleared,
        roi_reset=roi_reset,
    )


def probe_and_sync_profile_to_video(profile: Profile, source: str) -> tuple[VideoSourceInfo, ProfileVideoSyncResult]:
    info = probe_video_source(source)
    return info, sync_profile_to_video(profile, info)


def _roi_outside_frame(profile: Profile) -> bool:
    region = profile.inspection_region
    return (
        region.x < 0
        or region.y < 0
        or region.w <= 0
        or region.h <= 0
        or region.x + region.w > region.frame_width
        or region.y + region.h > region.frame_height
    )


__all__ = [
    "ProfileVideoSyncResult",
    "is_video_file_source",
    "probe_and_sync_profile_to_video",
    "sync_profile_to_video",
]
