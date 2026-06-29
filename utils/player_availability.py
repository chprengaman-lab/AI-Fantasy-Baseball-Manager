"""Shared helpers for player availability decisions."""

import re

import pandas as pd


UNAVAILABLE_STATUS_KEYWORDS = {
    "IL",
    "IL10",
    "IL15",
    "IL60",
    "OUT",
    "INJURED",
    "DTD",
    "DAY-TO-DAY",
    "DAY TO DAY",
    "NA",
    "SUSPENDED",
    "PATERNITY",
    "BEREAVEMENT",
}
IGNORED_STATUS_VALUES = {
    "",
    "NAN",
    "NONE",
    "NULL",
    "AVAILABLE",
    "ACTIVE",
    "NORMAL",
    "HEALTHY",
}
AVAILABILITY_STATUS_COLUMNS = [
    "injury_status",
    "status",
    "availability_note",
    "player_notes",
    "roster_status",
    "roster_spot",
    "lineup_status",
    "fantasy_status",
]


def _normalize_status_value(value) -> str:
    """Normalize a status-like value for exact-token matching."""

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    normalized = str(value).strip().strip("\"'").upper()
    if normalized in IGNORED_STATUS_VALUES:
        return ""

    return normalized


def _status_tokens(value: str) -> set[str]:
    """Split a status string into tokens without substring false positives."""

    if not value:
        return set()

    return {
        token
        for token in re.split(r"[\s,/\|()\-]+", value)
        if token and token not in IGNORED_STATUS_VALUES
    }


def classify_player_availability(row) -> dict:
    """Return detailed availability classification for debugging.

    Short markers like NA are dangerous with substring matching because they
    appear inside ordinary words like "Unavailable". We only match exact tokens
    or known full phrases.
    """

    raw_values = []

    for column in AVAILABILITY_STATUS_COLUMNS:
        value = _normalize_status_value(row.get(column, ""))
        if not value:
            continue

        raw_values.append(f"{column}={value}")

        if value in {"DAY TO DAY", "DAY-TO-DAY"}:
            return {
                "is_unavailable": True,
                "raw_status_values_checked": "; ".join(raw_values),
                "unavailable_marker_detected": value,
                "unavailable_source_field": column,
            }

        tokens = _status_tokens(value)
        for marker in UNAVAILABLE_STATUS_KEYWORDS:
            if marker in {"DAY TO DAY", "DAY-TO-DAY"}:
                continue

            if value == marker or marker in tokens:
                return {
                    "is_unavailable": True,
                    "raw_status_values_checked": "; ".join(raw_values),
                    "unavailable_marker_detected": marker,
                    "unavailable_source_field": column,
                }

    return {
        "is_unavailable": False,
        "raw_status_values_checked": "; ".join(raw_values),
        "unavailable_marker_detected": "",
        "unavailable_source_field": "",
    }


def is_player_unavailable(row) -> bool:
    """Return True when roster/status fields say a player is unavailable.

    ESPN and CSV files may use different columns for the same idea. We check all
    common status fields so injured players are excluded from optimizer and
    recommendation decisions by default.
    """

    return bool(classify_player_availability(row)["is_unavailable"])
