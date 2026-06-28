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
    get_espn_free_agent_player_objects,
    get_espn_player_position_debug,
    get_espn_teams,
    get_my_espn_roster,
    load_espn_config_from_env,
    normalize_espn_player,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE_PATH = PROJECT_ROOT / ".env"
ESPN_CACHE_KEY = "espn_api_test_cache"


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


def get_raw_roster_players(league, selected_team_id) -> list:
    """Return raw espn-api Player objects for the selected fantasy roster."""

    selected_team_id = str(selected_team_id)

    for team in getattr(league, "teams", []) or []:
        if str(getattr(team, "team_id", "")) == selected_team_id:
            return getattr(team, "roster", []) or []

    return []


def split_positions(eligible_positions: str) -> list[str]:
    """Split a normalized eligibility string into individual positions."""

    if not isinstance(eligible_positions, str) or not eligible_positions.strip():
        return []

    return [
        position.strip()
        for separator_group in eligible_positions.split(",")
        for position in separator_group.split("/")
        if position.strip()
    ]


def filter_free_agents(free_agents_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply simple display filters to normalized ESPN free agents."""

    if free_agents_dataframe.empty:
        return free_agents_dataframe

    position_options = sorted(
        {
            position
            for value in free_agents_dataframe["eligible_positions"].dropna()
            for position in split_positions(value)
        }
    )
    player_type_options = sorted(
        free_agents_dataframe["player_type"].dropna().unique()
    )
    selected_position = st.selectbox(
        "Free agent position filter",
        ["All Positions"] + position_options,
    )
    selected_player_type = st.selectbox(
        "Free agent player type filter",
        ["All Player Types"] + player_type_options,
    )
    player_search = st.text_input("Free agent player search")
    filtered_dataframe = free_agents_dataframe.copy()

    if selected_position != "All Positions":
        filtered_dataframe = filtered_dataframe[
            filtered_dataframe["eligible_positions"].apply(
                lambda value: selected_position in split_positions(value)
            )
        ]

    if selected_player_type != "All Player Types":
        filtered_dataframe = filtered_dataframe[
            filtered_dataframe["player_type"] == selected_player_type
        ]

    if player_search:
        filtered_dataframe = filtered_dataframe[
            filtered_dataframe["player"].str.contains(
                player_search,
                case=False,
                na=False,
            )
        ]

    return filtered_dataframe


def make_json_safe(value):
    """Convert debug values into JSON-friendly values for Streamlit."""

    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return repr(value)


def get_cached_espn_data() -> dict | None:
    """Read cached ESPN data from Streamlit session state."""

    return st.session_state.get(ESPN_CACHE_KEY)


def cache_espn_data(cache_payload: dict) -> None:
    """Store ESPN data so filters can rerun without reconnecting to ESPN."""

    st.session_state[ESPN_CACHE_KEY] = cache_payload


def load_espn_data(
    league_id,
    season,
    team_id,
    cleaned_espn_s2,
    cleaned_swid,
) -> dict:
    """Connect to ESPN once and return normalized tables for this page."""

    league = connect_espn_baseball_league(
        league_id=league_id,
        season=int(season),
        espn_s2=cleaned_espn_s2 or None,
        swid=cleaned_swid or None,
    )
    league_name = get_league_name(league)
    teams = get_espn_teams(league)
    roster = get_my_espn_roster(league, team_id=int(team_id))
    raw_roster_players = get_raw_roster_players(league, int(team_id))
    free_agents = []
    raw_free_agent_players = []
    free_agent_error = None

    try:
        # Free-agent loading is useful, but it should not break the rest of the
        # ESPN test page if ESPN rejects that specific request.
        raw_free_agent_players = get_espn_free_agent_player_objects(league, size=100)
        free_agents = [
            normalize_espn_player(player, roster_spot="Available")
            for player in raw_free_agent_players
        ]
    except ESPNFantasyError as error:
        free_agent_error = error

    selected_team_name = get_selected_team_name(teams, int(team_id))

    return {
        "league": league,
        "league_name": league_name,
        "selected_team_name": selected_team_name,
        "last_refresh_time": datetime.now(),
        "teams_dataframe": pd.DataFrame(teams),
        "roster_dataframe": pd.DataFrame(roster),
        "free_agents_dataframe": pd.DataFrame(free_agents),
        "raw_roster_players": raw_roster_players,
        "raw_free_agent_players": raw_free_agent_players,
        "free_agent_error": free_agent_error,
    }


def render_connection_status(status_text: str, cached_data: dict | None) -> None:
    """Show whether the page is using cached data, refreshing, or disconnected."""

    if status_text == "Refreshing ESPN data":
        st.info("Refreshing ESPN data")
        return

    if cached_data:
        st.success("Using cached ESPN data")
        return

    st.warning("Not connected")


def render_cached_espn_data(cached_data: dict) -> None:
    """Render roster, teams, and free-agent tables from cached ESPN data."""

    league_name = cached_data.get("league_name", "")
    selected_team_name = cached_data.get("selected_team_name", "")
    last_refresh_time = cached_data.get("last_refresh_time")
    teams_dataframe = cached_data.get("teams_dataframe", pd.DataFrame())
    roster_dataframe = cached_data.get("roster_dataframe", pd.DataFrame())
    free_agents_dataframe = cached_data.get("free_agents_dataframe", pd.DataFrame())
    free_agent_error = cached_data.get("free_agent_error")
    raw_roster_players = cached_data.get("raw_roster_players", [])
    raw_free_agent_players = cached_data.get("raw_free_agent_players", [])

    status_columns = st.columns(3)
    status_columns[0].metric("League", league_name or "Unknown")
    status_columns[1].metric("Team", selected_team_name or "Unknown")
    status_columns[2].metric(
        "Last Refresh",
        last_refresh_time.strftime("%Y-%m-%d %H:%M:%S")
        if last_refresh_time
        else "Unknown",
    )

    st.subheader("My Normalized Roster")

    if not roster_dataframe.empty:
        st.dataframe(roster_dataframe, width="stretch")
    else:
        st.info("No roster players were returned for this ESPN team.")

    st.subheader("League Teams")

    if not teams_dataframe.empty:
        st.dataframe(teams_dataframe, width="stretch")
    else:
        st.info("No league teams were returned by ESPN.")

    st.subheader("Available Free Agents")
    st.write("Showing the first 100 free agents returned by espn-api.")

    if free_agent_error is not None:
        st.warning(str(free_agent_error))

        if free_agent_error.debug_info:
            with st.expander("Free Agent Debug Details"):
                st.json(free_agent_error.debug_info)
    elif not free_agents_dataframe.empty:
        filtered_free_agents = filter_free_agents(free_agents_dataframe)
        # Keep this table shaped like the data the Top Pickups page will
        # eventually need when CSV uploads are replaced.
        output_columns = [
            "player",
            "normalized_player_name",
            "player_match_key",
            "espn_player_id",
            "eligible_positions",
            "roster_spot",
            "player_type",
            "pro_team",
            "injury_status",
        ]
        available_columns = [
            column
            for column in output_columns
            if column in filtered_free_agents.columns
        ]

        st.dataframe(
            filtered_free_agents[available_columns],
            width="stretch",
        )
    else:
        st.info("No free agents were returned by ESPN.")

    st.subheader("Player Position Debug")
    st.write(
        "Use this temporary debugger to compare raw espn-api position-like "
        "fields against the cleaned `eligible_positions` used by the optimizer."
    )
    render_player_position_debug(raw_roster_players, raw_free_agent_players)


def render_player_position_debug(
    raw_roster_players: list,
    raw_free_agent_players: list,
) -> None:
    """Show raw position-related fields for one selected ESPN player."""

    player_options = {}

    for player in raw_roster_players:
        player_name = getattr(player, "name", "Unknown Player")
        player_options[f"Roster: {player_name}"] = player

    for player in raw_free_agent_players:
        player_name = getattr(player, "name", "Unknown Player")
        player_options[f"Free Agent: {player_name}"] = player

    if not player_options:
        st.info("No raw ESPN player objects are available for debugging yet.")
        return

    selected_label = st.selectbox(
        "Choose a player to inspect",
        list(player_options.keys()),
    )
    selected_player = player_options[selected_label]

    st.json(
        make_json_safe(get_espn_player_position_debug(selected_player))
    )


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


cached_data = get_cached_espn_data()
refresh_requested = button_columns[1].button("Refresh ESPN Data")
connection_status = (
    "Refreshing ESPN data"
    if refresh_requested
    else "Using cached ESPN data"
    if cached_data
    else "Not connected"
)

render_connection_status(connection_status, cached_data)

if refresh_requested:
    try:
        cached_data = load_espn_data(
            league_id,
            int(season),
            int(team_id),
            cleaned_espn_s2,
            cleaned_swid,
        )
        cache_espn_data(cached_data)
        st.success("✅ Connected to ESPN")
    except ESPNFantasyError as error:
        st.error(str(error))
    except Exception as error:
        st.error("Unexpected ESPN API integration error.")
        with st.expander("Debug Error"):
            st.write(repr(error))

if cached_data:
    render_cached_espn_data(cached_data)
else:
    st.info("Refresh ESPN data to load your normalized roster, teams, and free agents.")
