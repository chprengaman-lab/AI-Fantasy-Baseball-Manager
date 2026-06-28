"""Experimental page that uses the reusable ESPN integration layer."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import set_key

from services.espn_fantasy import (
    ESPNFantasyError,
    ESPN_LEAGUE_ID_ENV,
    ESPN_S2_ENV,
    ESPN_SEASON_YEAR_ENV,
    ESPN_SWID_ENV,
    ESPN_TEAM_ID_ENV,
    connect_espn_baseball_league,
    get_espn_free_agents,
    get_espn_teams,
    get_my_espn_roster,
    load_espn_config_from_env,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE_PATH = PROJECT_ROOT / ".env"


def clean_cookie_input(cookie_value: str) -> str:
    """Strip whitespace and accidental wrapping quotes from a cookie value."""

    if not cookie_value:
        return ""

    return cookie_value.strip().strip("\"'")


def save_current_values_to_env(values: dict) -> None:
    """Write ESPN settings to the local .env file only."""

    ENV_FILE_PATH.touch(exist_ok=True)

    for key, value in values.items():
        set_key(str(ENV_FILE_PATH), key, str(value or ""))


def get_league_name(league) -> str:
    """Read league name from the espn-api League object."""

    settings = getattr(league, "settings", None)

    if settings is not None and getattr(settings, "name", ""):
        return settings.name

    return getattr(league, "league_name", "")


def get_selected_team_name(teams: list[dict], selected_team_id) -> str:
    """Find the selected fantasy team name from normalized team rows."""

    selected_team_id = str(selected_team_id)

    for team in teams:
        if str(team.get("team_id", "")) == selected_team_id:
            return team.get("team_name", "")

    return ""


env_config = load_espn_config_from_env()

st.title("ESPN API Integration")
st.warning(
    "This is experimental. It uses the reusable ESPN integration layer powered "
    "by the `espn-api` Python package."
)
st.write(
    "Values are pre-filled from local `.env` when available. You can override "
    "them for this session or save the current values back to your local `.env`."
)
st.caption("`.env` is ignored by git, so ESPN cookies should remain local.")

input_columns = st.columns(3)
league_id = input_columns[0].text_input(
    "ESPN league_id",
    value=str(env_config["league_id"] or ""),
)
season = input_columns[1].number_input(
    "Season year",
    min_value=2000,
    max_value=2100,
    value=int(env_config["season"] or datetime.now().year),
    step=1,
)
team_id = input_columns[2].number_input(
    "Team ID",
    min_value=1,
    value=int(env_config["team_id"] or 1),
    step=1,
)

espn_s2 = st.text_input(
    "espn_s2 cookie",
    value=env_config["espn_s2"],
    type="password",
    help="Private ESPN leagues usually require this cookie.",
)
swid = st.text_input(
    "SWID cookie",
    value=env_config["swid"],
    type="password",
    help="Private ESPN leagues usually require this cookie.",
)

cleaned_espn_s2 = clean_cookie_input(espn_s2)
cleaned_swid = clean_cookie_input(swid)

button_columns = st.columns(2)

if button_columns[0].button("Save current values to local .env"):
    save_current_values_to_env(
        {
            ESPN_LEAGUE_ID_ENV: league_id,
            ESPN_TEAM_ID_ENV: int(team_id),
            ESPN_SEASON_YEAR_ENV: int(season),
            ESPN_SWID_ENV: cleaned_swid,
            ESPN_S2_ENV: cleaned_espn_s2,
        }
    )
    st.success("Saved ESPN settings to local `.env`.")


if button_columns[1].button("Connect to ESPN"):
    try:
        league = connect_espn_baseball_league(
            league_id=league_id,
            season=int(season),
            espn_s2=cleaned_espn_s2 or None,
            swid=cleaned_swid or None,
        )
        league_name = get_league_name(league)
        teams = get_espn_teams(league)
        roster = get_my_espn_roster(league, team_id=int(team_id))
        free_agents = get_espn_free_agents(league, size=100)
        selected_team_name = get_selected_team_name(teams, int(team_id))

        st.success("✅ Connected to ESPN")
        status_columns = st.columns(3)
        status_columns[0].metric("League", league_name or "Unknown")
        status_columns[1].metric("Team", selected_team_name or f"Team {int(team_id)}")
        status_columns[2].metric(
            "Last Refresh",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        st.subheader("My Normalized Roster")

        if roster:
            st.dataframe(pd.DataFrame(roster), width="stretch")
        else:
            st.info("No roster players were returned for this ESPN team.")

        st.subheader("League Teams")

        if teams:
            st.dataframe(pd.DataFrame(teams), width="stretch")
        else:
            st.info("No league teams were returned by ESPN.")

        st.subheader("Available Free Agents")
        st.write("Showing the first 100 free agents returned by espn-api.")

        if free_agents:
            st.dataframe(pd.DataFrame(free_agents), width="stretch")
        else:
            st.info("No free agents were returned by ESPN.")
    except ESPNFantasyError as error:
        st.error(str(error))
    except Exception as error:
        st.error("Unexpected ESPN API integration error.")
        with st.expander("Debug Error"):
            st.write(repr(error))
else:
    st.info("Connect to ESPN to load your normalized roster, teams, and free agents.")
