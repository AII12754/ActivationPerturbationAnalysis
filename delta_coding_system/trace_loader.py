"""Trace data loader for compaction_prompt_lite.json.

Loads compression events from the OpenClaw memory trace dataset,
providing prompt_before / prompt_after pairs for delta-coding evaluation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class TraceEvent:
    """A single prompt compaction event from the trace dataset."""
    id: str
    session_id: str
    tokens_before: int
    prompt_before: str
    prompt_after: str


def load_trace_events(json_path: str | Path) -> List[TraceEvent]:
    """Load and parse compaction trace events from a JSON file.

    Parameters
    ----------
    json_path : path to compaction_prompt_lite.json

    Returns
    -------
    List of TraceEvent, one per compaction event.
    """
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raw = [raw]

    events: List[TraceEvent] = []
    for entry in raw:
        event = TraceEvent(
            id=entry["id"],
            session_id=entry.get("session_id", ""),
            tokens_before=entry.get("tokens_before", 0),
            prompt_before=entry["prompt_before"],
            prompt_after=entry["prompt_after"],
        )
        events.append(event)

    logger.info("Loaded %d trace events from %s", len(events), json_path)
    return events
