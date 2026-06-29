"""Persistent manual roster protection preferences.

These preferences are intentionally simple. Until a real rest-of-season value
model exists, the user can manually mark players as protected, droppable, or
streamers and the optimizer will respect that choice.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from utils.name_matching import normalize_player_name


PREFERENCES_PATH = Path("data/roster_preferences.json")
ROSTER_VALUE_STATUSES = [
    "Do Not Drop",
    "Drop for Strong Upgrade",
    "Droppable",
    "Streamer",
]
STATUS_MIGRATION = {
    "CORE": "Do Not Drop",
    "NORMAL": "Drop for Strong Upgrade",
    "DROP CANDIDATE": "Droppable",
    "STREAMER": "Streamer",
    "DO NOT DROP": "Do Not Drop",
    "DROPPABLE": "Droppable",
    "DROP FOR STRONG UPGRADE": "Drop for Strong Upgrade",
}


def migrate_roster_value_status(value) -> str:
    """Map old saved statuses into the current four-status system."""

    if not isinstance(value, str) or not value.strip():
        return "Drop for Strong Upgrade"

    normalized_value = " ".join(value.strip().upper().split())
    return STATUS_MIGRATION.get(normalized_value, "Drop for Strong Upgrade")


def get_roster_preference_key(player_row) -> str:
    """Use ESPN id when available, otherwise fall back to normalized name."""

    espn_player_id = player_row.get("espn_player_id", "")
    if pd.notna(espn_player_id) and str(espn_player_id).strip():
        return f"espn:{str(espn_player_id).strip()}"

    return f"name:{normalize_player_name(player_row.get('player', ''))}"


def get_default_roster_value_status(player_row) -> str:
    """Return the default manual protection status for one rostered player."""

    roster_spot = str(player_row.get("roster_spot", "")).strip().upper()
    if roster_spot.startswith("IL"):
        return "Do Not Drop"

    return "Drop for Strong Upgrade"


def load_roster_preferences() -> dict:
    """Load saved roster preferences from disk."""

    if not PREFERENCES_PATH.exists():
        return {}

    try:
        with PREFERENCES_PATH.open("r", encoding="utf-8") as preferences_file:
            payload = json.load(preferences_file)
    except (json.JSONDecodeError, OSError):
        return {}

    if not isinstance(payload, dict):
        return {}

    migrated_payload = {}
    changed = False
    for preference_key, preference in payload.items():
        if not isinstance(preference, dict):
            changed = True
            continue

        migrated_status = migrate_roster_value_status(
            preference.get("roster_value_status", "")
        )
        migrated_preference = {
            **preference,
            "roster_value_status": migrated_status,
            "streamer_hold_until": preference.get("streamer_hold_until", ""),
            "streamer_note": preference.get("streamer_note", ""),
            "manual_note": preference.get("manual_note", ""),
            "updated_at": preference.get("updated_at", ""),
        }
        if migrated_preference != preference:
            changed = True
        migrated_payload[preference_key] = migrated_preference

    if changed:
        save_roster_preferences(migrated_payload)

    return migrated_payload


def save_roster_preferences(preferences: dict) -> None:
    """Persist roster preferences to the local data folder."""

    PREFERENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PREFERENCES_PATH.open("w", encoding="utf-8") as preferences_file:
        json.dump(preferences, preferences_file, indent=2, sort_keys=True)


def reset_roster_preferences(roster_df: pd.DataFrame) -> dict:
    """Reset the current roster to default manual statuses."""

    preferences = build_default_preferences(roster_df)
    save_roster_preferences(preferences)
    return preferences


def build_default_preferences(roster_df: pd.DataFrame) -> dict:
    """Build default preferences for the current roster."""

    preferences = {}
    if roster_df is None or roster_df.empty:
        return preferences

    now = datetime.now(timezone.utc).isoformat()
    for _, player_row in roster_df.iterrows():
        preference_key = get_roster_preference_key(player_row)
        preferences[preference_key] = {
            "roster_value_status": get_default_roster_value_status(player_row),
            "manual_note": "",
            "streamer_hold_until": "",
            "streamer_note": "",
            "updated_at": now,
        }

    return preferences


def apply_roster_preferences(
    roster_df: pd.DataFrame,
    preferences: dict | None = None,
) -> pd.DataFrame:
    """Attach saved manual protection settings to roster rows."""

    roster_df = roster_df.copy()
    preferences = preferences if preferences is not None else load_roster_preferences()
    now = datetime.now(timezone.utc).isoformat()

    if roster_df.empty:
        return roster_df

    statuses = []
    notes = []
    updated_values = []
    preference_keys = []
    for _, player_row in roster_df.iterrows():
        preference_key = get_roster_preference_key(player_row)
        saved_preference = preferences.get(preference_key, {})
        statuses.append(
            migrate_roster_value_status(
                saved_preference.get(
                    "roster_value_status",
                    get_default_roster_value_status(player_row),
                )
            )
        )
        notes.append(saved_preference.get("manual_note", ""))
        updated_values.append(saved_preference.get("updated_at", now))
        preference_keys.append(preference_key)

    roster_df["roster_preference_key"] = preference_keys
    roster_df["roster_value_status"] = statuses
    roster_df["manual_note"] = notes
    roster_df["streamer_hold_until"] = [
        preferences.get(key, {}).get("streamer_hold_until", "")
        for key in preference_keys
    ]
    roster_df["streamer_note"] = [
        preferences.get(key, {}).get("streamer_note", "")
        for key in preference_keys
    ]
    roster_df["roster_preference_updated_at"] = updated_values
    return roster_df


def _format_optional_date(value) -> str:
    """Store date-like editor values as YYYY-MM-DD strings."""

    if value is None or pd.isna(value):
        return ""

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return str(value).strip()


def build_preferences_from_editor(
    edited_roster_df: pd.DataFrame,
    existing_preferences: dict | None = None,
) -> dict:
    """Convert edited Streamlit rows into the JSON preference shape."""

    preferences = dict(existing_preferences or {})
    now = datetime.now(timezone.utc).isoformat()

    if edited_roster_df is None or edited_roster_df.empty:
        return preferences

    for _, player_row in edited_roster_df.iterrows():
        preference_key = player_row.get("roster_preference_key") or get_roster_preference_key(
            player_row
        )
        preferences[str(preference_key)] = {
            "roster_value_status": migrate_roster_value_status(
                player_row.get("roster_value_status", "Drop for Strong Upgrade")
            ),
            "manual_note": player_row.get("manual_note", ""),
            "streamer_hold_until": _format_optional_date(
                player_row.get("streamer_hold_until", "")
            ),
            "streamer_note": player_row.get("streamer_note", ""),
            "updated_at": now,
        }

    return preferences
