from __future__ import annotations

import base64
import functools
import json
import logging
import os
from pathlib import Path
from typing import get_args

import cv2
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from .prompt import NORMAL_OVERRIDE, SYSTEM_PROMPT
from .schema import DefectReason, make_openai_llm

LOGGER = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"
_EXAMPLE_LABELS = set(get_args(DefectReason.model_fields["defect_type"].annotation))
_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "example"


def _encode(bar_id: str, image: np.ndarray) -> dict:
    """Stamp bar_id on a copy and encode as base64 JPEG for OpenAI vision."""
    img = image.copy()
    cv2.putText(img, bar_id, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    _, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
    }


def _encode_low(image: np.ndarray) -> dict:
    """Encode example image with low detail (85 tokens, sufficient for few-shot patterns)."""
    _, buf = cv2.imencode(".jpg", image)
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
    }


@functools.lru_cache(maxsize=1)
def _load_example_content() -> tuple:
    """Load few-shot example images from data/example/. Cached after first call."""
    if not _EXAMPLES_DIR.exists():
        LOGGER.warning("Few-shot example dir not found: %s", _EXAMPLES_DIR)
        return ()

    by_label: dict[str, list[Path]] = {}
    for f in sorted(_EXAMPLES_DIR.glob("*.jpg")):
        label = "_".join(f.stem.split("_")[3:])
        if label in _EXAMPLE_LABELS:
            by_label.setdefault(label, []).append(f)

    items: list = [{"type": "text", "text": "--- EXAMPLES (reference only, do not classify) ---"}]
    for label in sorted(by_label):
        for img_path in by_label[label]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            items.append({"type": "text", "text": f"{label}:"})
            items.append(_encode_low(img))
    items.append({"type": "text", "text": "--- BARS TO CLASSIFY ---"})

    n = sum(1 for it in items if it["type"] == "image_url")
    LOGGER.info("Loaded %d few-shot example image(s) from %s", n, _EXAMPLES_DIR)
    return tuple(items)


_REASON_LABELS = {
    "length_too_short": "too short",
    "length_too_long":  "too long",
    "width_too_small":  "too narrow",
    "width_too_large":  "too wide",
}


def _build_bar_hints(bars: list) -> str:
    """Build a per-bar measurement context block to prepend to the classify request."""
    lines = [
        "Measurement anomalies detected by the calibration system:",
        "These are statistical hints based on geometry only — not ground truth. "
        "Your visual judgment from the image always takes priority. "
        "Use the hints to guide your attention, but classify based on what you actually see.",
    ]
    for bar in bars:
        labels = [_REASON_LABELS.get(r) or r for r in bar.reasons]
        reasons_str = ", ".join(labels) if labels else "unknown"
        has_short = any("length_too_short" == r for r in bar.reasons)
        has_wide  = any("width_too_large"  == r for r in bar.reasons)
        has_long  = any("length_too_long"  == r for r in bar.reasons)

        if has_short and has_wide:
            hint = (
                "SHORT + WIDE → strongly suggests BENT. "
                "To determine direction: look at which half of the bar has lost its horizontal line. "
                "If only the RIGHT half droops/curves while the left half is still straight → bent_right. "
                "If only the LEFT half droops/curves while the right half is still straight → bent_left. "
                "If BOTH halves are deformed (V-shape, arc, or sagging at the center) → bent_both. "
                "Do NOT default to bent_both — actively check each half independently."
            )
        elif has_wide and not has_short:
            hint = (
                "TOO WIDE but normal length → suggests a BENT bar where the deformation is mild "
                "and has not yet shortened the visible length significantly. "
                "To determine direction: check which half of the bar appears curved or drooping. "
                "If only the RIGHT half is deformed → bent_right. "
                "If only the LEFT half is deformed → bent_left. "
                "If both halves show deformation → bent_both. "
                "Do NOT default to bent_both — actively check each half independently."
            )
        elif has_short and not has_wide:
            hint = (
                "SHORT but normal width — this can indicate either a BROKEN bar or a BENT bar. "
                "Do NOT assume broken without carefully examining the image. "
                "Check: if one wing is physically missing, snapped off, or there is a visible fracture → broken. "
                "If the bar appears structurally complete but curved or deformed → bent_left, bent_right, or bent_both. "
                "A severely bent bar can appear shorter than normal even without a fracture. "
                "Base your decision entirely on what you see, not on this hint."
            )
        elif has_long:
            hint = "TOO LONG → unusual deformation; inspect carefully."
        else:
            hint = "Inspect carefully for subtle deformation."

        lines.append(f"  • {bar.bar_id}  [{reasons_str}]  →  {hint}")
    return "\n".join(lines)


def _parse(raw: str) -> list[DefectReason]:
    """Accept [...], {"results": [...]}, or NDJSON (one object per line)."""
    raw = raw.strip()
    try:
        data = json.loads(raw)
        items = data if isinstance(data, list) else data.get("results", [data])
    except json.JSONDecodeError:
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    out = []
    for item in items:
        try:
            out.append(DefectReason(**item))
        except Exception:
            pass
    return out


class VlmInspector:
    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL, *, can_mark_normal: bool = False) -> None:
        self._api_key = api_key
        self._model = model
        self._can_mark_normal = can_mark_normal
        self.request_count = 0

    @classmethod
    def from_env(cls, model: str = _DEFAULT_MODEL, *, can_mark_normal: bool = False) -> "VlmInspector":
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY env var is not set")
        return cls(api_key=key, model=model, can_mark_normal=can_mark_normal)

    @classmethod
    def from_profile(cls, profile) -> "VlmInspector":
        vlm = profile.inspection.vlm
        return cls.from_env(model=vlm.model, can_mark_normal=vlm.can_mark_normal)

    def inspect(self, bars: list) -> dict[str, DefectReason]:
        """Classify all defective bars in a single request. Returns {bar_id: DefectReason}."""
        self.request_count = 0
        if not bars:
            return {}

        llm = make_openai_llm(self._api_key, self._model, num_bars=len(bars))
        ids = [b.bar_id for b in bars]

        content: list = list(_load_example_content())
        content.append({"type": "text", "text": _build_bar_hints(bars)})
        content.append({"type": "text", "text": f"Classify bars: {', '.join(ids)}. One result per bar_id."})
        for bar in bars:
            content.append(_encode(bar.bar_id, bar.crop_image))

        system_prompt = SYSTEM_PROMPT + (NORMAL_OVERRIDE if self._can_mark_normal else "")

        self.request_count = 1
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=content),
            ])
            raw = response.content  # type: ignore[union-attr]
            usage = response.response_metadata.get("token_usage", {})
            reasoning = response.additional_kwargs.get("reasoning_content", "")
            LOGGER.info(
                "VLM usage | prompt=%s completion=%s reasoning_tokens=%s | finish=%s",
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
                response.response_metadata.get("finish_reason"),
            )
            if reasoning:
                LOGGER.info("VLM reasoning:\n%.3000s", reasoning)
            LOGGER.info("VLM output:\n%.3000s", raw)
            decisions = {d.bar_id: d for d in _parse(raw)}
            if not decisions:
                LOGGER.warning("VLM returned no decisions for %s", ids)
            return decisions
        except Exception as exc:
            LOGGER.warning("VLM call failed: %s", exc)
            return {}


__all__ = ["VlmInspector"]
