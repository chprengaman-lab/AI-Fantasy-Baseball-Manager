"""Manual roster protection settings.

This page owns the editable controls for droppable and undroppable players.
Top Pickups reads the saved preferences but stays focused on daily decisions.
"""

from datetime import datetime

import pandas as pd
import streamlit as st

from services.espn_fantasy import (
    ESPNFantasyError,
    connect_espn_baseball_league,
    get_my_espn_roster,
    load_espn_config_from_env,
)
from services.roster_preferences import (
    ROSTER_VALUE_STATUSES,
    apply_roster_preferences,
    build_preferences_from_editor,
    load_roster_preferences,
    reset_roster_preferences,
    save_roster_preferences,
)
from utils.streamlit_dataframe import clean_dataframe_for_streamlit


ROSTER_SETTINGS_CACHE_KEY = "roster_settings_espn_cache"


def load_roster_from_espn() -> dict:
    """Connect to ESPN and return the current roster as a dataframe."""

    config = load_espn_config_from_env()
    league = connect_espn_baseball_league(
        league_id=config["league_id"],
        season=config["season"],
        espn_s2=config["espn_s2"],
        swid=config["swid"],
    )
    roster_rows = get_my_espn_roster(league, team_id=config["team_id"])
    return {
        "roster_dataframe": pd.DataFrame(roster_rows),
        "last_refresh_time": datetime.now(),
    }


def get_cached_roster() -> dict | None:
    """Read cached ESPN roster data from this Streamlit session."""

    return st.session_state.get(ROSTER_SETTINGS_CACHE_KEY)


def cache_roster(payload: dict) -> None:
    """Cache roster data so editing controls do not reconnect on every rerun."""

    st.session_state[ROSTER_SETTINGS_CACHE_KEY] = payload


def prepare_editor_table(roster_df: pd.DataFrame) -> pd.DataFrame:
    """Create the editable table shown on this page."""

    roster_with_preferences = apply_roster_preferences(
        roster_df,
        load_roster_preferences(),
    )
    roster_with_preferences["current_roster_spot"] = roster_with_preferences.get(
        "roster_spot",
        "",
    )
    editor_columns = [
        "roster_preference_key",
        "player",
        "current_roster_spot",
        "eligible_positions",
        "roster_value_status",
        "manual_note",
        "streamer_hold_until",
        "streamer_note",
    ]
    for column in editor_columns:
        if column not in roster_with_preferences.columns:
            roster_with_preferences[column] = ""

    editor_table = roster_with_preferences[editor_columns].copy()
    editor_table["streamer_hold_until"] = pd.to_datetime(
        editor_table["streamer_hold_until"],
        errors="coerce",
    ).dt.date
    return editor_table


st.title("Roster Settings")
st.write(
    "Use this page to manually protect players, mark droppable players, or set "
    "streamer hold dates. Top Pickups uses these saved preferences when evaluating "
    "add/drop moves."
)

cached_roster = get_cached_roster()
refresh_roster = st.button("Refresh ESPN Roster")

if refresh_roster or cached_roster is None:
    try:
        cached_roster = load_roster_from_espn()
        cache_roster(cached_roster)
        st.success("Loaded ESPN roster.")
    except ESPNFantasyError as error:
        cached_roster = {"roster_dataframe": pd.DataFrame(), "error": error}
        cache_roster(cached_roster)
        st.warning(str(error))
        if error.debug_info:
            with st.expander("ESPN Debug Details"):
                st.json(error.debug_info)

if cached_roster is None or cached_roster.get("roster_dataframe", pd.DataFrame()).empty:
    st.info("No ESPN roster is loaded. Check your `.env` ESPN settings, then refresh.")
else:
    last_refresh = cached_roster.get("last_refresh_time")
    if last_refresh:
        st.caption(f"Last ESPN refresh: {last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")

    roster_df = cached_roster["roster_dataframe"].copy()
    editor_table = prepare_editor_table(roster_df)
    st.write(
        "Defaults are conservative: IL players start as Do Not Drop, and no one "
        "defaults to Droppable automatically."
    )
    edited_table = st.data_editor(
        clean_dataframe_for_streamlit(editor_table),
        width="stretch",
        hide_index=True,
        disabled=[
            "roster_preference_key",
            "player",
            "current_roster_spot",
            "eligible_positions",
        ],
        column_config={
            "roster_preference_key": None,
            "roster_value_status": st.column_config.SelectboxColumn(
                "roster_value_status",
                options=ROSTER_VALUE_STATUSES,
                required=True,
            ),
            "manual_note": st.column_config.TextColumn("manual_note"),
            "streamer_hold_until": st.column_config.DateColumn(
                "streamer_hold_until",
                format="YYYY-MM-DD",
            ),
            "streamer_note": st.column_config.TextColumn("streamer_note"),
        },
        key="roster_settings_editor",
    )

    action_columns = st.columns(2)
    if action_columns[0].button("Save Preferences"):
        preferences = build_preferences_from_editor(
            edited_table,
            load_roster_preferences(),
        )
        save_roster_preferences(preferences)
        st.success("Roster preferences saved to `data/roster_preferences.json`.")

    confirm_reset = action_columns[1].checkbox("Confirm reset preferences")
    if action_columns[1].button("Reset Preferences", disabled=not confirm_reset):
        reset_roster_preferences(roster_df)
        st.success("Roster preferences reset to defaults.")
        st.rerun()
