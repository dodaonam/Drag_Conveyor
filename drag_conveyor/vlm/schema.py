from __future__ import annotations

from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class DefectReason(BaseModel):
    bar_id: str = Field(description="The id of the conveyor bar, from the label in the top-left corner")
    defect_type: Literal["normal", "bent_left", "bent_right", "bent_both", "broken", "other"] = Field(
        description=(
            "The type of defect on the conveyor bar, or 'normal' if the bar actually looks "
            "healthy/straight despite being flagged by the geometry system (false positive)"
        )
    )


def make_openai_llm(api_key: str, model: str, num_bars: int = 1) -> ChatOpenAI:
    output_tokens = max(400, num_bars * 80 + 200)
    return ChatOpenAI(  # type: ignore[call-arg]
        model=model,
        temperature=0.3,
        openai_api_key=api_key,
        max_retries=0,
        max_completion_tokens=output_tokens,
    )


__all__ = ["DefectReason", "make_openai_llm"]
