"""Daily command center for hitter pickup decisions.

This page intentionally keeps the rendering flow simple:
load roster/free-agent data, build projections, then render every section with
an empty-state message when data is missing. One empty table should never stop
the rest of the page from rendering.
"""

from datetime import datetime
import re
from time import perf_counter

import pandas as pd
import streamlit as st

from config import LEAGUE_RULES
from services.espn_fantasy import (
    ESPNFantasyError,
    connect_espn_baseball_league,
    get_espn_free_agents,
    get_my_espn_roster,
    load_espn_config_from_env,
)
from services.lineup_optimizer import (
    calculate_replacement_value_by_position,
    compare_lineups,
    evaluate_multi_add_hitter_scenarios,
    evaluate_single_hitter_pickups,
    get_drop_protection_status,
    get_roster_flexibility_summary,
    is_droppable_player,
    optimize_hitter_lineup,
)
from services.odds_api import MissingOddsAPIKeyError, OddsAPIError, load_cached_hitter_props
from services.player_projection_engine import build_player_projection_table
from utils.name_matching import (
    build_player_match_key,
    fuzzy_match_player_name,
    normalize_player_name,
)
from utils.player_availability import classify_player_availability, is_player_unavailable
from utils.streamlit_dataframe import clean_dataframe_for_streamlit
from utils.streamlit_debug import show_odds_api_error_debug


ESPN_TOP_PICKUPS_CACHE_KEY = "top_pickups_espn_cache"
STARTING_HITTER_POSITIONS = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"}
TOP_PICKUPS_COLUMNS = [
    "player",
    "team",
    "eligible_positions",
    "roster_status",
    "bookmaker",
    "projected_fantasy_points",
    "projection_source",
    "projection_confidence",
    "has_sportsbook_props",
    "is_available_today",
    "availability_note",
    "has_game_today",
    "game_today_note",
    "bookmaker_count",
    "markets_available",
    "matched_sportsbook_player",
    "missing_markets",
    "fallback_markets_used",
    "projected_hits",
    "projected_home_runs",
    "projected_rbi",
    "projected_runs",
    "projected_stolen_bases",
    "projected_total_bases",
    "estimated_singles",
    "estimated_doubles",
    "estimated_triples",
    "batting_average",
    "obp",
    "slg",
    "ops",
]


def split_positions(value: str) -> list[str]:
    """Split comma- or slash-separated ESPN positions into clean tokens."""

    if not isinstance(value, str) or not value.strip():
        return []

    return [item.strip().upper() for item in re.split(r"[,/]", value) if item.strip()]


def row_is_pitcher(row) -> bool:
    """Return True when a row represents a pitcher.

    Roster spot describes current usage, while eligible positions describe where
    ESPN allows the player to be used. Current usage is preferred for grouping.
    """

    roster_spot = str(row.get("roster_spot", "")).strip().upper()
    if roster_spot in {"SP", "RP"}:
        return True

    player_type = str(row.get("player_type", "")).strip().upper()
    if player_type in {"SP", "RP"}:
        return True

    return bool(set(split_positions(str(row.get("eligible_positions", "")))) & {"SP", "RP"})


def row_is_hitter(row) -> bool:
    """Return True for players the hitter optimizer should consider."""

    return not row_is_pitcher(row)


def player_has_position(eligible_positions: str, selected_position: str) -> bool:
    """Return True when a player can fill the selected exact position."""

    if selected_position == "All Positions":
        return True

    return selected_position in split_positions(eligible_positions)


def as_bool_series(series: pd.Series, default: bool = False) -> pd.Series:
    """Convert mixed bool/string/blank values into a reliable boolean Series."""

    def to_bool(value) -> bool:
        if pd.isna(value):
            return default

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return value == 1

        if isinstance(value, str):
            cleaned_value = value.strip().lower()

            if cleaned_value in {"true", "yes", "y", "1"}:
                return True

            if cleaned_value in {"false", "no", "n", "0", ""}:
                return False

        return default

    return series.apply(to_bool)


def get_raw_hitter_prop_match_keys() -> set[str]:
    """Return player match keys found in the cached raw hitter prop rows."""

    cached_payload = load_cached_hitter_props()
    prop_rows = cached_payload.get("hitter_props", [])
    match_keys = set()

    for prop_row in prop_rows:
        player_name = prop_row.get("player", "")

        if player_name:
            match_keys.add(build_player_match_key(player_name))

    return match_keys


def row_has_sportsbook_props(row, raw_prop_match_keys: set[str] | None = None) -> bool:
    """Return True when a player has sportsbook-line evidence today."""

    projection_source = str(row.get("projection_source", ""))
    raw_markets_available = row.get("markets_available", "")
    markets_available = (
        ""
        if pd.isna(raw_markets_available)
        else str(raw_markets_available).strip()
    )
    if markets_available.lower() in {"nan", "<na>", "none"}:
        markets_available = ""
    bookmaker_count = pd.to_numeric(row.get("bookmaker_count", 0), errors="coerce")

    if projection_source in {"Stat-based fallback", "Missing", "Unknown"}:
        return False

    if "Sportsbook" in projection_source:
        return True

    if markets_available:
        return True

    if pd.notna(bookmaker_count) and bookmaker_count > 0:
        return True

    return False


def row_is_actual_sportsbook_projection(row) -> bool:
    """Return True only for rows that already contain sportsbook projection data."""

    projection_source = str(row.get("projection_source", ""))
    raw_markets_available = row.get("markets_available", "")
    markets_available = (
        ""
        if pd.isna(raw_markets_available)
        else str(raw_markets_available).strip()
    )
    if markets_available.lower() in {"nan", "<na>", "none"}:
        markets_available = ""
    bookmaker_count = pd.to_numeric(row.get("bookmaker_count", 0), errors="coerce")

    if "Sportsbook" in projection_source:
        return True

    if pd.notna(bookmaker_count) and bookmaker_count > 0:
        return True

    return bool(markets_available)


def add_sportsbook_prop_flags(
    player_table: pd.DataFrame,
    raw_prop_match_keys: set[str],
) -> pd.DataFrame:
    """Add the has_sportsbook_props column used by daily recommendation filters."""

    player_table = ensure_pickup_table_columns(player_table)

    if "player_match_key" not in player_table.columns:
        player_table["player_match_key"] = player_table["player"].apply(
            build_player_match_key
        )

    player_table["has_sportsbook_props"] = player_table.apply(
        lambda row: row_has_sportsbook_props(row, raw_prop_match_keys),
        axis=1,
    )
    stat_fallback_mask = player_table["projection_source"].isin(
        ["Stat-based fallback", "Missing", "Unknown"]
    )
    player_table.loc[stat_fallback_mask, "has_sportsbook_props"] = False
    return player_table


def apply_fuzzy_sportsbook_projection_matches(
    player_table: pd.DataFrame,
    min_score: int = 92,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copy sportsbook projection fields onto ESPN/CSV rows with near-match names.

    Exact matching happens earlier through normalized names. This function only
    repairs rows that are available/rostered but still lack sportsbook props,
    while a sportsbook row for a very similar name exists elsewhere.
    """

    player_table = player_table.copy()
    player_table["matched_sportsbook_player"] = player_table.get(
        "matched_sportsbook_player",
        "",
    )
    player_table["sportsbook_match_score"] = player_table.get(
        "sportsbook_match_score",
        0,
    )
    player_table["sportsbook_match_method"] = player_table.get(
        "sportsbook_match_method",
        "",
    )
    sportsbook_pool = player_table[
        player_table.apply(row_is_actual_sportsbook_projection, axis=1)
    ].copy()

    if sportsbook_pool.empty:
        return player_table, pd.DataFrame()

    sportsbook_names = sportsbook_pool["player"].dropna().astype(str).unique().tolist()
    sportsbook_by_normalized_name = (
        sportsbook_pool.sort_values("projected_fantasy_points", ascending=False)
        .drop_duplicates(subset=["normalized_player_name"], keep="first")
        .set_index("normalized_player_name")
    )
    sportsbook_by_match_key = (
        sportsbook_pool.sort_values("projected_fantasy_points", ascending=False)
        .drop_duplicates(subset=["player_match_key"], keep="first")
        .set_index("player_match_key")
    )
    projection_columns_to_copy = [
        column
        for column in player_table.columns
        if column
        not in {
            "player",
            "team",
            "eligible_positions",
            "roster_status",
            "normalized_player_name",
            "player_match_key",
            "espn_player_id",
            "roster_spot",
            "player_type",
            "pro_team",
            "injury_status",
            "status",
            "player_notes",
            "lineup_status",
            "fantasy_status",
            "availability_note",
            "is_available_today",
            "has_game_today",
            "game_today_note",
            "droppable",
            "undroppable",
            "long_term_value",
            "keeper_status",
            "notes",
        }
    ]
    diagnostic_rows = []

    for index, player_row in player_table.iterrows():
        if player_row.get("roster_status") not in {"Available", "On My Roster"}:
            continue

        if row_is_actual_sportsbook_projection(player_row):
            diagnostic_rows.append(
                {
                    "espn_player": player_row.get("player", ""),
                    "espn_player_id": player_row.get("espn_player_id", ""),
                    "normalized_player_name": player_row.get("normalized_player_name", ""),
                    "player_match_key": player_row.get("player_match_key", ""),
                    "matched_projection_player": player_row.get("player", ""),
                    "matched_projection_source": player_row.get(
                        "projection_source",
                        "",
                    ),
                    "matched_bookmaker_count": player_row.get("bookmaker_count", 0),
                    "matched_markets_available": player_row.get("markets_available", ""),
                    "has_sportsbook_props": True,
                    "match_method": "already sportsbook row",
                    "match_score": 100,
                }
            )
            continue

        matched_name = ""
        match_score = 0
        match_method = ""
        source_row = None
        normalized_name = player_row.get("normalized_player_name", "")
        player_match_key = player_row.get("player_match_key", "")

        if normalized_name in sportsbook_by_normalized_name.index:
            source_row = sportsbook_by_normalized_name.loc[normalized_name]
            matched_name = source_row.get("player", "")
            match_score = 100
            match_method = "exact normalized name"
        elif player_match_key in sportsbook_by_match_key.index:
            source_row = sportsbook_by_match_key.loc[player_match_key]
            matched_name = source_row.get("player", "")
            match_score = 100
            match_method = "exact match key"
        else:
            fuzzy_name, fuzzy_score = fuzzy_match_player_name(
                player_row.get("player", ""),
                sportsbook_names,
                min_score=min_score,
            )

            if fuzzy_name:
                source_row = sportsbook_pool[
                    sportsbook_pool["player"] == fuzzy_name
                ].iloc[0]
                matched_name = fuzzy_name
                match_score = fuzzy_score
                match_method = "fuzzy"

        matched_source = ""
        matched_bookmaker_count = ""
        matched_markets_available = ""

        if source_row is not None:

            for column in projection_columns_to_copy:
                player_table.at[index, column] = source_row.get(column, "")

            player_table.at[index, "has_sportsbook_props"] = True
            player_table.at[index, "matched_sportsbook_player"] = matched_name
            player_table.at[index, "sportsbook_match_score"] = match_score
            player_table.at[index, "sportsbook_match_method"] = match_method
            matched_source = source_row.get("projection_source", "")
            matched_bookmaker_count = source_row.get("bookmaker_count", 0)
            matched_markets_available = source_row.get("markets_available", "")

        diagnostic_rows.append(
            {
                "espn_player": player_row.get("player", ""),
                "espn_player_id": player_row.get("espn_player_id", ""),
                "normalized_player_name": player_row.get("normalized_player_name", ""),
                "player_match_key": player_row.get("player_match_key", ""),
                "matched_projection_player": matched_name or "",
                "matched_projection_source": matched_source,
                "matched_bookmaker_count": matched_bookmaker_count,
                "matched_markets_available": matched_markets_available,
                "has_sportsbook_props": bool(source_row is not None),
                "match_method": match_method,
                "match_score": match_score,
            }
        )

    matched_projection_players = {
        row.get("matched_projection_player", "")
        for row in diagnostic_rows
        if row.get("matched_projection_player", "")
    }

    for _, sportsbook_row in sportsbook_pool.iterrows():
        sportsbook_player = sportsbook_row.get("player", "")

        if sportsbook_player in matched_projection_players:
            continue

        diagnostic_rows.append(
            {
                "espn_player": "",
                "espn_player_id": "",
                "normalized_player_name": "",
                "player_match_key": "",
                "matched_projection_player": sportsbook_player,
                "matched_projection_source": sportsbook_row.get("projection_source", ""),
                "matched_bookmaker_count": sportsbook_row.get("bookmaker_count", 0),
                "matched_markets_available": sportsbook_row.get("markets_available", ""),
                "has_sportsbook_props": True,
                "match_method": "unmatched sportsbook projection",
                "match_score": 0,
            }
        )

    player_table.loc[
        player_table["projection_source"].isin(["Stat-based fallback", "Missing", "Unknown"]),
        "has_sportsbook_props",
    ] = False

    return player_table, pd.DataFrame(diagnostic_rows)


def get_pickup_tier(projected_fantasy_points: float) -> str:
    """Return a simple label based on today's projected fantasy points."""

    if projected_fantasy_points >= 8:
        return "Elite pickup"
    if projected_fantasy_points >= 6:
        return "Strong pickup"
    if projected_fantasy_points >= 4:
        return "Watchlist"
    return "Low priority"


def add_player_matching_columns(players_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Add normalized names so ESPN, CSV, and sportsbook names can be matched."""

    if players_df is None:
        return None

    players_df = players_df.copy()
    if players_df.empty or "player" not in players_df.columns:
        return players_df

    if "eligible_positions" not in players_df.columns and "position" in players_df.columns:
        players_df["eligible_positions"] = players_df["position"]

    players_df["normalized_player_name"] = players_df["player"].apply(
        normalize_player_name
    )
    players_df["player_match_key"] = players_df["player"].apply(build_player_match_key)
    return players_df


def normalize_espn_dataframe(rows: list[dict], roster_status: str) -> pd.DataFrame:
    """Convert ESPN rows into the columns used by this page."""

    columns = [
        "player",
        "espn_player_id",
        "eligible_positions",
        "roster_spot",
        "player_type",
        "pro_team",
        "injury_status",
        "status",
        "player_notes",
        "lineup_status",
        "fantasy_status",
        "normalized_player_name",
        "player_match_key",
    ]
    dataframe = pd.DataFrame(rows)

    for column in columns:
        if column not in dataframe.columns:
            dataframe[column] = ""

    dataframe = dataframe[columns].copy()
    dataframe["roster_status"] = roster_status
    return dataframe


def load_espn_top_pickups_data() -> dict:
    """Load ESPN roster and free-agent data once, then cache it in Streamlit."""

    env_config = load_espn_config_from_env()
    league = connect_espn_baseball_league(
        league_id=env_config["league_id"],
        season=env_config["season"],
        espn_s2=env_config["espn_s2"],
        swid=env_config["swid"],
    )
    roster_rows = get_my_espn_roster(league, team_id=env_config["team_id"])
    free_agent_rows = get_espn_free_agents(league, size=500)

    return {
        "roster_dataframe": normalize_espn_dataframe(roster_rows, "On My Roster"),
        "available_players_dataframe": normalize_espn_dataframe(
            free_agent_rows,
            "Available",
        ),
        "last_refresh_time": datetime.now(),
        "error": None,
    }


def get_cached_espn_top_pickups_data() -> dict | None:
    """Read cached ESPN data from the current Streamlit session."""

    return st.session_state.get(ESPN_TOP_PICKUPS_CACHE_KEY)


def cache_espn_top_pickups_data(cache_payload: dict) -> None:
    """Store ESPN data so filters do not reconnect to ESPN."""

    st.session_state[ESPN_TOP_PICKUPS_CACHE_KEY] = cache_payload


def build_available_players_template() -> pd.DataFrame:
    """Create a CSV template for waiver/free-agent hitters."""

    return pd.DataFrame(
        [
            {"player": "Example Catcher", "eligible_positions": "C,DH", "player_type": "Hitter"},
            {"player": "Example Left Fielder", "eligible_positions": "LF,DH", "player_type": "Hitter"},
            {"player": "Example Third Baseman", "eligible_positions": "3B,1B", "player_type": "Hitter"},
        ]
    )


def build_roster_template() -> pd.DataFrame:
    """Create a CSV template for current roster uploads."""

    return pd.DataFrame(
        [
            {
                "player": "Example Star Hitter",
                "eligible_positions": "SS,DH",
                "roster_spot": "Starter",
                "player_type": "Hitter",
                "droppable": "FALSE",
                "undroppable": "TRUE",
                "long_term_value": "High",
                "keeper_status": "Core",
                "notes": "Season-long foundation player",
            },
            {
                "player": "Example Bench Hitter",
                "eligible_positions": "1B,DH",
                "roster_spot": "Bench",
                "player_type": "Hitter",
                "droppable": "TRUE",
                "undroppable": "FALSE",
                "long_term_value": "Low",
                "keeper_status": "Streamer",
                "notes": "Daily matchup stream",
            },
            {
                "player": "Example Bench Starter",
                "eligible_positions": "SP",
                "roster_spot": "Bench",
                "player_type": "SP",
                "droppable": "",
                "undroppable": "",
                "long_term_value": "Medium",
                "keeper_status": "Hold",
                "notes": "Pitcher optimization is not built yet",
            },
            {
                "player": "Example Injured Player",
                "eligible_positions": "CF",
                "roster_spot": "IL",
                "player_type": "Hitter",
                "droppable": "",
                "undroppable": "",
                "long_term_value": "High",
                "keeper_status": "Hold",
                "notes": "IL players are protected from recommendations",
            },
        ]
    )


def ensure_pickup_table_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Guarantee the page has the columns it needs even with no projections."""

    table = table.copy()
    expected_columns = set(TOP_PICKUPS_COLUMNS) | {
        "bookmaker",
        "tier",
        "has_sportsbook_props",
        "matched_sportsbook_player",
        "sportsbook_match_score",
        "sportsbook_match_method",
        "normalized_player_name",
        "player_type",
        "roster_spot",
        "injury_status",
        "status",
        "pro_team",
        "player_notes",
        "lineup_status",
        "fantasy_status",
        "has_game_today",
        "game_today_note",
        "espn_player_id",
        "player_match_key",
        "droppable",
        "undroppable",
        "long_term_value",
        "keeper_status",
        "notes",
    }

    for column in expected_columns:
        if column not in table.columns:
            table[column] = ""

    table["projected_fantasy_points"] = pd.to_numeric(
        table["projected_fantasy_points"],
        errors="coerce",
    ).fillna(0.0)
    return table


def get_actual_sportsbook_projection_table(
    projection_table: pd.DataFrame,
) -> pd.DataFrame:
    """Return only rows that came from real sportsbook prop data.

    This prevents stat fallback rows from being treated like sportsbook rows
    just because they have a game today or season stats.
    """

    projection_table = ensure_pickup_table_columns(projection_table)
    if projection_table.empty:
        return projection_table

    projection_table = add_player_matching_columns(projection_table)
    sportsbook_table = projection_table[
        projection_table.apply(row_is_actual_sportsbook_projection, axis=1)
    ].copy()

    if sportsbook_table.empty:
        return sportsbook_table

    sportsbook_table["projected_fantasy_points"] = pd.to_numeric(
        sportsbook_table["projected_fantasy_points"],
        errors="coerce",
    ).fillna(0.0)
    return sportsbook_table.sort_values(
        "projected_fantasy_points",
        ascending=False,
    )


def merge_projection_data_onto_player_source(
    source_df: pd.DataFrame | None,
    projection_table: pd.DataFrame,
    roster_status: str,
    min_score: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach sportsbook projections to ESPN/CSV player rows.

    The source table is the truth for who is rostered or available. Projection
    data is copied onto those rows after matching by player_match_key,
    normalized name, then fuzzy name. Fallback rows are only used when no
    sportsbook projection matched.
    """

    output_columns = ensure_pickup_table_columns(pd.DataFrame()).columns.tolist()
    if source_df is None or source_df.empty or "player" not in source_df.columns:
        return pd.DataFrame(columns=output_columns), pd.DataFrame()

    source_df = ensure_pickup_table_columns(add_player_matching_columns(source_df))
    source_df["roster_status"] = roster_status

    projection_table = ensure_pickup_table_columns(add_player_matching_columns(projection_table))
    sportsbook_table = get_actual_sportsbook_projection_table(projection_table)

    sportsbook_by_key = (
        sportsbook_table.drop_duplicates(subset=["player_match_key"], keep="first")
        .set_index("player_match_key")
        if not sportsbook_table.empty
        else pd.DataFrame()
    )
    sportsbook_by_normalized = (
        sportsbook_table.drop_duplicates(subset=["normalized_player_name"], keep="first")
        .set_index("normalized_player_name")
        if not sportsbook_table.empty
        else pd.DataFrame()
    )
    sportsbook_names = (
        sportsbook_table["player"].dropna().astype(str).unique().tolist()
        if not sportsbook_table.empty
        else []
    )

    fallback_table = projection_table[
        ~projection_table.apply(row_is_actual_sportsbook_projection, axis=1)
    ].copy()
    fallback_by_key = (
        fallback_table.sort_values("projected_fantasy_points", ascending=False)
        .drop_duplicates(subset=["player_match_key"], keep="first")
        .set_index("player_match_key")
        if not fallback_table.empty
        else pd.DataFrame()
    )
    fallback_by_normalized = (
        fallback_table.sort_values("projected_fantasy_points", ascending=False)
        .drop_duplicates(subset=["normalized_player_name"], keep="first")
        .set_index("normalized_player_name")
        if not fallback_table.empty
        else pd.DataFrame()
    )

    # These fields identify the ESPN/CSV player. We preserve them and copy only
    # projection values from the sportsbook/fallback row.
    source_identity_columns = {
        "player",
        "team",
        "eligible_positions",
        "roster_status",
        "normalized_player_name",
        "player_match_key",
        "espn_player_id",
        "roster_spot",
        "player_type",
        "pro_team",
        "injury_status",
        "status",
        "player_notes",
        "lineup_status",
        "fantasy_status",
        "availability_note",
        "is_available_today",
        "has_game_today",
        "game_today_note",
        "droppable",
        "undroppable",
        "long_term_value",
        "keeper_status",
        "notes",
    }
    projection_columns_to_copy = [
        column for column in output_columns if column not in source_identity_columns
    ]

    merged_rows = []
    diagnostic_rows = []

    for _, source_row in source_df.iterrows():
        merged_row = source_row.to_dict()
        if not str(merged_row.get("player_type", "")).strip():
            merged_row["player_type"] = "Pitcher" if row_is_pitcher(source_row) else "Hitter"
        player_match_key = str(source_row.get("player_match_key", ""))
        normalized_name = str(source_row.get("normalized_player_name", ""))
        matched_projection_row = None
        matched_projection_player = ""
        match_method = ""
        match_score = 0

        if player_match_key and not sportsbook_by_key.empty and player_match_key in sportsbook_by_key.index:
            matched_projection_row = sportsbook_by_key.loc[player_match_key]
            matched_projection_player = matched_projection_row.get("player", "")
            match_method = "exact match key"
            match_score = 100
        elif (
            normalized_name
            and not sportsbook_by_normalized.empty
            and normalized_name in sportsbook_by_normalized.index
        ):
            matched_projection_row = sportsbook_by_normalized.loc[normalized_name]
            matched_projection_player = matched_projection_row.get("player", "")
            match_method = "exact normalized name"
            match_score = 100
        elif sportsbook_names:
            fuzzy_name, fuzzy_score = fuzzy_match_player_name(
                source_row.get("player", ""),
                sportsbook_names,
                min_score=min_score,
            )
            if fuzzy_name:
                matched_projection_row = sportsbook_table[
                    sportsbook_table["player"] == fuzzy_name
                ].iloc[0]
                matched_projection_player = fuzzy_name
                match_method = "fuzzy"
                match_score = fuzzy_score

        has_sportsbook_match = matched_projection_row is not None

        if not has_sportsbook_match:
            if player_match_key and not fallback_by_key.empty and player_match_key in fallback_by_key.index:
                matched_projection_row = fallback_by_key.loc[player_match_key]
                matched_projection_player = matched_projection_row.get("player", "")
                match_method = "fallback exact match key"
                match_score = 100
            elif (
                normalized_name
                and not fallback_by_normalized.empty
                and normalized_name in fallback_by_normalized.index
            ):
                matched_projection_row = fallback_by_normalized.loc[normalized_name]
                matched_projection_player = matched_projection_row.get("player", "")
                match_method = "fallback exact normalized name"
                match_score = 100

        if matched_projection_row is not None:
            for column in projection_columns_to_copy:
                merged_row[column] = matched_projection_row.get(column, merged_row.get(column, ""))
            for context_column in [
                "team",
                "has_game_today",
                "game_today_note",
                "is_available_today",
                "availability_note",
            ]:
                current_value = merged_row.get(context_column, "")
                if pd.isna(current_value) or str(current_value).strip() == "":
                    merged_row[context_column] = matched_projection_row.get(
                        context_column,
                        current_value,
                    )
            merged_row["matched_sportsbook_player"] = (
                matched_projection_player if has_sportsbook_match else ""
            )
            merged_row["sportsbook_match_method"] = match_method
            merged_row["sportsbook_match_score"] = match_score
        else:
            merged_row["projected_fantasy_points"] = 0.0
            merged_row["projection_source"] = "Missing"
            merged_row["projection_confidence"] = "Unknown"
            merged_row["matched_sportsbook_player"] = ""
            merged_row["sportsbook_match_method"] = "unmatched"
            merged_row["sportsbook_match_score"] = 0

        merged_row["has_sportsbook_props"] = (
            has_sportsbook_match or row_has_sportsbook_props(merged_row)
        )
        diagnostic_rows.append(
            {
                "espn_player": source_row.get("player", ""),
                "espn_player_id": source_row.get("espn_player_id", ""),
                "normalized_player_name": normalized_name,
                "player_match_key": player_match_key,
                "matched_projection_player": matched_projection_player,
                "matched_projection_source": (
                    matched_projection_row.get("projection_source", "")
                    if matched_projection_row is not None
                    else ""
                ),
                "matched_bookmaker_count": (
                    matched_projection_row.get("bookmaker_count", 0)
                    if matched_projection_row is not None
                    else 0
                ),
                "matched_markets_available": (
                    matched_projection_row.get("markets_available", "")
                    if matched_projection_row is not None
                    else ""
                ),
                "has_sportsbook_props": bool(merged_row["has_sportsbook_props"]),
                "match_method": match_method or "unmatched",
                "match_score": match_score,
            }
        )
        merged_rows.append(merged_row)

    merged_table = ensure_pickup_table_columns(pd.DataFrame(merged_rows))
    merged_table["has_sportsbook_props"] = merged_table.apply(
        row_has_sportsbook_props,
        axis=1,
    )
    merged_table.loc[
        merged_table["projection_source"].isin(["Stat-based fallback", "Missing", "Unknown"]),
        "has_sportsbook_props",
    ] = False
    return merged_table, pd.DataFrame(diagnostic_rows)


def build_position_enrichment_dataframe(
    available_df: pd.DataFrame | None,
    roster_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Build player metadata used to enrich projection rows."""

    frames = []
    for source_df in [available_df, roster_df]:
        if source_df is None or source_df.empty or "player" not in source_df.columns:
            continue

        source_df = add_player_matching_columns(source_df)
        columns = [
            column
            for column in [
                "player",
                "eligible_positions",
                "position",
                "injury_status",
                "availability_note",
                "roster_spot",
                "status",
                "pro_team",
                "player_notes",
                "lineup_status",
                "fantasy_status",
                "has_game_today",
                "game_today_note",
                "is_available_today",
                "espn_player_id",
                "player_match_key",
                "normalized_player_name",
                "player_type",
            ]
            if column in source_df.columns
        ]
        enrichment = source_df[columns].copy()
        if "eligible_positions" not in enrichment.columns and "position" in enrichment.columns:
            enrichment = enrichment.rename(columns={"position": "eligible_positions"})
        frames.append(enrichment)

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["normalized_player_name"],
        keep="first",
    )


def append_missing_player_rows(
    projection_table: pd.DataFrame,
    source_df: pd.DataFrame | None,
    roster_status: str,
) -> pd.DataFrame:
    """Keep ESPN/CSV players visible even when no projection row matched."""

    output = ensure_pickup_table_columns(projection_table)
    if source_df is None or source_df.empty or "player" not in source_df.columns:
        return output

    source_df = add_player_matching_columns(source_df)
    existing_names = set(output["normalized_player_name"].dropna())
    missing_rows = source_df[
        ~source_df["normalized_player_name"].isin(existing_names)
    ].copy()

    if missing_rows.empty:
        return output

    for column in output.columns:
        if column not in missing_rows.columns:
            missing_rows[column] = ""

    missing_rows["roster_status"] = roster_status
    missing_rows["projected_fantasy_points"] = 0.0
    missing_rows["projection_source"] = "Missing"
    missing_rows["projection_confidence"] = "Unknown"
    missing_rows["is_available_today"] = ~missing_rows.apply(
        is_player_unavailable,
        axis=1,
    )
    if "has_game_today" not in missing_rows.columns:
        missing_rows["has_game_today"] = False
    no_game_mask = ~as_bool_series(missing_rows["has_game_today"])
    missing_rows.loc[no_game_mask, "projection_source"] = (
        "No game today"
    )
    missing_rows.loc[no_game_mask, "projection_confidence"] = (
        "Unavailable"
    )
    missing_rows.loc[no_game_mask, "game_today_note"] = (
        "No game today: MLB team could not be identified."
    )
    return pd.concat([output, missing_rows[output.columns]], ignore_index=True)


def add_roster_metadata(
    projection_table: pd.DataFrame,
    roster_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach roster fields like droppable/keeper status to projection rows."""

    projection_table = ensure_pickup_table_columns(projection_table)
    if roster_df is None or roster_df.empty or "player" not in roster_df.columns:
        return projection_table

    roster_df = add_player_matching_columns(roster_df)
    metadata_columns = [
        column
        for column in [
            "normalized_player_name",
            "roster_spot",
            "droppable",
            "undroppable",
            "long_term_value",
            "keeper_status",
            "injury_status",
            "availability_note",
            "status",
            "pro_team",
            "player_notes",
            "lineup_status",
            "fantasy_status",
            "has_game_today",
            "game_today_note",
            "is_available_today",
            "espn_player_id",
            "player_match_key",
            "notes",
            "player_type",
        ]
        if column in roster_df.columns
    ]
    roster_metadata = roster_df[metadata_columns].drop_duplicates(
        subset=["normalized_player_name"],
        keep="first",
    )
    merged = projection_table.merge(
        roster_metadata,
        on="normalized_player_name",
        how="left",
        suffixes=("", "_roster"),
    )

    for column in metadata_columns:
        if column == "normalized_player_name":
            continue
        roster_column = f"{column}_roster"
        if roster_column in merged.columns:
            merged[column] = merged[roster_column].combine_first(merged[column])
            merged = merged.drop(columns=[roster_column])

    return merged


def filter_hitter_optimizer_pool(
    table: pd.DataFrame,
    include_unavailable_players: bool,
    include_players_without_games: bool,
    include_stat_fallback_only_hitters: bool,
) -> pd.DataFrame:
    """Keep healthy hitters for lineup optimization and pickup simulations."""

    if table is None or table.empty:
        return pd.DataFrame()

    filtered = table.copy()
    if not include_unavailable_players:
        filtered = filtered[~filtered.apply(is_player_unavailable, axis=1)].copy()

    if not include_players_without_games and "has_game_today" in filtered.columns:
        filtered = filtered[as_bool_series(filtered["has_game_today"])].copy()

    if (
        not include_stat_fallback_only_hitters
        and "has_sportsbook_props" in filtered.columns
    ):
        filtered = filtered[as_bool_series(filtered["has_sportsbook_props"])].copy()

    if "projection_source" in filtered.columns:
        filtered = filtered[
            ~filtered["projection_source"].isin(["Missing", "Unknown"])
        ].copy()

    return filtered[filtered.apply(row_is_hitter, axis=1)].copy()


def get_available_hitter_filter_reason(
    row,
    include_unavailable_players: bool,
    include_players_without_games: bool,
    include_stat_fallback_only_hitters: bool,
) -> str:
    """Explain why an available hitter is excluded from recommendation tables."""

    if row_is_pitcher(row):
        return "Filtered out because this player is a pitcher."

    if not include_unavailable_players and is_player_unavailable(row):
        return "Filtered out because the player is injured or unavailable."

    if (
        not include_players_without_games
        and "has_game_today" in row
        and not bool(as_bool_series(pd.Series([row.get("has_game_today")])).iloc[0])
    ):
        return "Filtered out because the player's MLB team does not play today."

    if (
        not include_stat_fallback_only_hitters
        and not bool(as_bool_series(pd.Series([row.get("has_sportsbook_props")])).iloc[0])
    ):
        return "Filtered out because the player has no sportsbook hitter props."

    if row.get("projection_source", "") in {"Missing", "Unknown"}:
        return "Filtered out because no usable projection matched."

    return "Included in the recommendation candidate pool."


def is_ranked_sportsbook_table_candidate(row) -> bool:
    """Return True when an available player belongs in the main sportsbook table."""

    player_type = str(row.get("player_type", "")).strip().lower()
    projected_points = pd.to_numeric(
        row.get("projected_fantasy_points", 0),
        errors="coerce",
    )
    return (
        player_type == "hitter"
        and bool(as_bool_series(pd.Series([row.get("has_sportsbook_props")])).iloc[0])
        and bool(as_bool_series(pd.Series([row.get("has_game_today")])).iloc[0])
        and bool(as_bool_series(pd.Series([row.get("is_available_today")])).iloc[0])
        and pd.notna(projected_points)
    )


def get_ranked_sportsbook_exclusion_reason(row, player_search: str = "") -> str:
    """Explain why a matched sportsbook free agent is not visible in the ranked table."""

    if str(row.get("player_type", "")).strip().lower() != "hitter":
        return "not hitter"

    if not bool(as_bool_series(pd.Series([row.get("has_sportsbook_props")])).iloc[0]):
        return "no sportsbook props flag"

    if not bool(as_bool_series(pd.Series([row.get("has_game_today")])).iloc[0]):
        return "no game today"

    if not bool(as_bool_series(pd.Series([row.get("is_available_today")])).iloc[0]):
        return "unavailable/injured"

    projected_points = pd.to_numeric(
        row.get("projected_fantasy_points", 0),
        errors="coerce",
    )
    if pd.isna(projected_points):
        return "missing projected points"

    if player_search and player_search.lower() not in str(row.get("player", "")).lower():
        return "filtered by user search"

    return ""


def reduce_available_hitter_candidates(
    available_table: pd.DataFrame,
    per_position_limit: int = 8,
    total_limit: int = 60,
) -> pd.DataFrame:
    """Keep the strongest available hitters before expensive simulations.

    The optimizer still evaluates add/drop moves the same way. This only
    prevents the brute-force simulator from spending time on hundreds of
    low-projection free agents who are unlikely to affect the starting lineup.
    We keep top players by exact position so scarce slots like C, CF, RF, and
    DH still have candidates.
    """

    if available_table is None or available_table.empty:
        return pd.DataFrame()

    candidates = available_table.copy()
    candidates["projected_fantasy_points"] = pd.to_numeric(
        candidates["projected_fantasy_points"],
        errors="coerce",
    ).fillna(0)
    selected_indexes = set()

    for position in sorted(STARTING_HITTER_POSITIONS):
        position_candidates = candidates[
            candidates["eligible_positions"].apply(
                lambda value: player_has_position(value, position)
            )
        ].sort_values("projected_fantasy_points", ascending=False)
        selected_indexes.update(position_candidates.head(per_position_limit).index)

    if not selected_indexes:
        return candidates.sort_values(
            "projected_fantasy_points",
            ascending=False,
        ).head(total_limit)

    reduced = candidates.loc[list(selected_indexes)].copy()
    reduced = reduced.sort_values("projected_fantasy_points", ascending=False)

    if len(reduced) > total_limit:
        reduced = reduced.head(total_limit)

    return reduced


def get_player_name_set(table: pd.DataFrame | None) -> set[str]:
    """Return normalized names from a roster or available-player table."""

    if table is None or table.empty or "player" not in table.columns:
        return set()

    table = add_player_matching_columns(table)
    return set(table["normalized_player_name"].dropna())


def assign_roster_status(
    projection_table: pd.DataFrame,
    roster_df: pd.DataFrame | None,
    available_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Mark whether each projection row is rostered, available, or unknown."""

    projection_table = ensure_pickup_table_columns(projection_table)
    if "normalized_player_name" not in projection_table.columns:
        projection_table["normalized_player_name"] = projection_table["player"].apply(
            normalize_player_name
        )

    roster_names = get_player_name_set(roster_df)
    available_names = get_player_name_set(available_df)

    def get_status(player_name: str) -> str:
        if player_name in roster_names:
            return "On My Roster"
        if player_name in available_names:
            return "Available"
        return "Unknown"

    projection_table["roster_status"] = projection_table[
        "normalized_player_name"
    ].apply(get_status)
    return projection_table


def build_roster_protection_review(roster_df: pd.DataFrame) -> pd.DataFrame:
    """Show exactly why each rostered player is protected or droppable."""

    columns = [
        "player",
        "eligible_positions",
        "roster_spot",
        "player_type",
        "droppable",
        "undroppable",
        "long_term_value",
        "keeper_status",
        "notes",
        "inferred_drop_status",
        "drop_reason",
    ]
    if roster_df is None or roster_df.empty:
        return pd.DataFrame(columns=columns)

    review = roster_df.copy()
    for column in columns:
        if column not in review.columns:
            review[column] = ""

    drop_details = review.apply(get_drop_protection_status, axis=1, result_type="expand")
    review["inferred_drop_status"] = drop_details[0]
    review["drop_reason"] = drop_details[1]
    return review[columns]


def build_projection_coverage_review(master_table: pd.DataFrame) -> pd.DataFrame:
    """Show whether each roster/free-agent player has a usable projection."""

    columns = [
        "player",
        "roster_status",
        "eligible_positions",
        "projected_fantasy_points",
        "projection_source",
        "projection_confidence",
    ]
    master_table = ensure_pickup_table_columns(master_table)
    return master_table[columns].sort_values(["roster_status", "player"])


def build_projection_coverage_summary(
    roster_projection_table: pd.DataFrame,
    available_projection_table: pd.DataFrame,
) -> dict:
    """Count matched and missing projections for roster and free-agent pools."""

    def count_pool(table: pd.DataFrame) -> tuple[int, int]:
        if table is None or table.empty:
            return 0, 0
        missing = table["projection_source"].isin(
            ["Missing", "Unavailable", "Unknown", "No game today"]
        )
        return int((~missing).sum()), int(missing.sum())

    roster_matched, roster_missing = count_pool(roster_projection_table)
    available_matched, available_missing = count_pool(available_projection_table)
    return {
        "roster_players_matched": roster_matched,
        "roster_players_missing": roster_missing,
        "free_agents_matched": available_matched,
        "free_agents_missing": available_missing,
    }


def show_projection_coverage_summary(summary: dict) -> None:
    """Display quick projection coverage metrics."""

    st.subheader("Projection Coverage Summary")
    columns = st.columns(4)
    columns[0].metric("Roster Matched", summary["roster_players_matched"])
    columns[1].metric("Roster Missing", summary["roster_players_missing"])
    columns[2].metric("Free Agents Matched", summary["free_agents_matched"])
    columns[3].metric("Free Agents Missing", summary["free_agents_missing"])


def show_weekly_sp_start_tracker() -> dict:
    """Display SP start context without changing optimizer behavior."""

    weekly_cap = LEAGUE_RULES["weekly_sp_start_cap"]
    st.header("Weekly SP Start Tracker")
    st.write(
        "This is context only. It helps balance hitter streaming against future "
        "starting pitcher needs."
    )
    columns = st.columns(2)
    starts_used = columns[0].number_input(
        "SP starts already used this week",
        min_value=0,
        value=0,
        step=1,
    )
    starts_planned = columns[1].number_input(
        "SP starts currently planned for rest of week",
        min_value=0,
        value=0,
        step=1,
    )
    total_projected_starts = starts_used + starts_planned
    starts_remaining = weekly_cap - total_projected_starts

    metrics = st.columns(3)
    metrics[0].metric("SP Starts Used", starts_used)
    metrics[1].metric("SP Starts Planned", starts_planned)
    metrics[2].metric("SP Starts Remaining", starts_remaining)

    if total_projected_starts > weekly_cap:
        st.warning("Your planned SP starts exceed the weekly cap. Some starts may not count.")
    elif starts_remaining == 0:
        st.info("You have filled your weekly SP start capacity, so hitter streaming is especially attractive.")
    else:
        st.info("You still have SP starts available, so preserve enough SP depth for later starts.")

    return {"starts_remaining": starts_remaining}


def show_roster_flexibility_summary(
    roster_df: pd.DataFrame,
    sp_start_context: dict | None,
) -> dict:
    """Display active roster space, IL count, and pitcher count context."""

    summary = get_roster_flexibility_summary(roster_df)
    st.header("Roster Flexibility Summary")
    st.write(
        "This explains whether you can add a hitter without dropping anyone. "
        "Carrying fewer than the SP maximum can create hitter streaming flexibility."
    )
    columns = st.columns(5)
    columns[0].metric(
        "Active Roster Spots Used",
        f"{summary['active_roster_count']} / {summary['active_roster_limit']}",
    )
    columns[1].metric("Open Active Spots", summary["open_active_spots"])
    columns[2].metric("SP Count", f"{summary['sp_count']} / {summary['sp_limit']}")
    columns[3].metric("RP Count", f"{summary['rp_count']} / {summary['rp_limit']}")
    columns[4].metric("IL Count", f"{summary['il_count']} / {summary['il_limit']}")

    note = summary["flexibility_note"]
    if sp_start_context is not None:
        if sp_start_context["starts_remaining"] == 0 and summary["open_active_spots"] > 0:
            note += " Your weekly SP start capacity is filled, making hitter streaming more attractive."
        elif sp_start_context["starts_remaining"] > 0:
            note += " Balance hitter streaming against future SP start needs."
    st.info(note)
    return summary


def show_data_source_status(
    selected_data_source: str,
    roster_df: pd.DataFrame | None,
    available_df: pd.DataFrame | None,
    espn_cache: dict | None,
) -> None:
    """Show a compact status box for loaded player data."""

    roster_count = 0 if roster_df is None else len(roster_df)
    available_count = 0 if available_df is None else len(available_df)
    last_refresh = ""
    if espn_cache is not None and espn_cache.get("last_refresh_time"):
        last_refresh = espn_cache["last_refresh_time"].strftime("%Y-%m-%d %H:%M:%S")

    with st.container(border=True):
        st.write(f"Data source: **{selected_data_source}**")
        columns = st.columns(3)
        columns[0].metric("Roster Players Loaded", roster_count)
        columns[1].metric("Free Agents Loaded", available_count)
        columns[2].metric("Last ESPN Refresh", last_refresh or "Not refreshed")


def get_best_positive_move(
    pickup_recommendations: pd.DataFrame,
    multi_add_scenarios: pd.DataFrame,
):
    """Choose the best move by risk-adjusted score.

    If scores are close, no-drop moves are preferred because they preserve
    season-long roster value.
    """

    tables = [
        table
        for table in [pickup_recommendations, multi_add_scenarios]
        if table is not None and not table.empty
    ]
    if not tables:
        return None

    moves = pd.concat(tables, ignore_index=True, sort=False)
    moves = moves[
        (moves["projected_gain"] > 0) & (moves["risk_adjusted_score"] > 0)
    ].copy()
    if moves.empty:
        return None

    moves["move_type_priority"] = moves["move_type"].map(
        {"Multi Add": 0, "Add Only": 1, "Add/Drop": 2}
    ).fillna(99)
    best_score = moves["risk_adjusted_score"].max()
    close_moves = moves[best_score - moves["risk_adjusted_score"] <= 0.5].copy()
    return close_moves.sort_values(
        ["move_type_priority", "risk_adjusted_score"],
        ascending=[True, False],
    ).iloc[0]


def show_todays_action_plan(best_move) -> None:
    """Show the first plain-English recommendation on the page."""

    st.header("Today's Action Plan")
    if best_move is None:
        st.warning("No recommended pickup move today. Your best move may be to hold your roster.")
        return

    move_type = best_move.get("move_type", "")
    projected_gain = float(best_move.get("projected_gain", 0))
    score = float(best_move.get("risk_adjusted_score", 0))
    label = best_move.get("risk_adjusted_label", "")
    drop_risk = best_move.get("drop_risk", "")

    if move_type == "Multi Add":
        message = f"Best move today: Add {best_move.get('add_players', '')} without dropping anyone."
    elif move_type == "Add Only":
        message = f"Best move today: Add {best_move.get('add_player', '')} without dropping anyone."
    else:
        message = (
            f"Best move today: Add {best_move.get('add_player', '')} and "
            f"drop {best_move.get('drop_player', '')}."
        )

    detail = (
        f"Why: projected gain is +{projected_gain:.2f} points, and the "
        f"risk-adjusted score is {score:.2f} ({label})."
    )
    confidence = best_move.get("add_projection_confidence") or best_move.get(
        "projection_confidence",
        "Unknown",
    )
    source = best_move.get("add_projection_source") or best_move.get(
        "projection_source",
        "unknown source",
    )

    if move_type == "Add/Drop" and drop_risk in {"Medium", "High"}:
        st.warning(f"{message}\n\n{detail}\n\nDrop risk: {drop_risk}.")
    elif label in {"Excellent", "Good"}:
        st.success(f"{message}\n\n{detail}")
    else:
        st.info(f"{message}\n\n{detail}")

    st.caption(f"Confidence: {confidence} from {source}.")


def show_lineup_impact(
    roster_projection_table: pd.DataFrame,
    available_projection_table: pd.DataFrame,
    best_move,
) -> None:
    """Show how the optimized starting lineup changes for the best move."""

    st.subheader("Lineup Impact")
    if best_move is None:
        st.info("No positive move was selected, so there is no lineup impact to show.")
        return

    current_lineup, _, _ = optimize_hitter_lineup(roster_projection_table)
    move_type = best_move.get("move_type", "")
    new_roster = roster_projection_table.copy()

    if move_type == "Multi Add":
        add_names = [
            normalize_player_name(name)
            for name in str(best_move.get("add_players", "")).split(",")
        ]
        add_rows = available_projection_table[
            available_projection_table["normalized_player_name"].isin(add_names)
        ]
        new_roster = pd.concat([new_roster, add_rows], ignore_index=True)
    elif move_type == "Add Only":
        add_name = normalize_player_name(best_move.get("add_player", ""))
        add_rows = available_projection_table[
            available_projection_table["normalized_player_name"] == add_name
        ]
        new_roster = pd.concat([new_roster, add_rows], ignore_index=True)
    elif move_type == "Add/Drop":
        add_name = normalize_player_name(best_move.get("add_player", ""))
        drop_name = normalize_player_name(best_move.get("drop_player", ""))
        new_roster = new_roster[new_roster["normalized_player_name"] != drop_name]
        add_rows = available_projection_table[
            available_projection_table["normalized_player_name"] == add_name
        ]
        new_roster = pd.concat([new_roster, add_rows], ignore_index=True)

    new_lineup, _, _ = optimize_hitter_lineup(new_roster)
    impact = compare_lineups(current_lineup, new_lineup)
    columns = st.columns(3)
    columns[0].metric(
        "Current Starting Total",
        round(current_lineup["projected_fantasy_points"].sum(), 2),
    )
    columns[1].metric(
        "New Starting Total",
        round(new_lineup["projected_fantasy_points"].sum(), 2),
    )
    columns[2].metric(
        "Projected Gain",
        round(new_lineup["projected_fantasy_points"].sum() - current_lineup["projected_fantasy_points"].sum(), 2),
    )

    if impact.empty:
        st.info("The optimized lineup slots did not change.")
    else:
        st.dataframe(clean_dataframe_for_streamlit(impact.round(3)), width="stretch")


def show_grouped_roster_tables(
    lineup_df: pd.DataFrame,
    roster_group_table: pd.DataFrame,
) -> None:
    """Group roster players so pitchers do not appear under bench hitters."""

    if roster_group_table.empty:
        st.info("No roster players are available to group.")
        return

    starters = set(lineup_df["player"].dropna()) if not lineup_df.empty else set()
    roster = roster_group_table.copy()
    unavailable = roster.apply(is_player_unavailable, axis=1)
    pitchers = roster.apply(row_is_pitcher, axis=1)
    sp_mask = roster["roster_spot"].astype(str).str.upper().eq("SP") | (
        roster["player_type"].astype(str).str.upper().eq("SP")
    )
    rp_mask = roster["roster_spot"].astype(str).str.upper().eq("RP") | (
        roster["player_type"].astype(str).str.upper().eq("RP")
    )
    hitter_mask = ~pitchers & ~unavailable

    groups = {
        "Starting Hitters": roster[hitter_mask & roster["player"].isin(starters)],
        "Bench Hitters": roster[hitter_mask & ~roster["player"].isin(starters)],
        "Starting Pitchers": roster[~unavailable & sp_mask],
        "Relief Pitchers": roster[~unavailable & rp_mask],
        "Injured / Unavailable": roster[unavailable],
    }
    display_columns = [
        "player",
        "eligible_positions",
        "roster_spot",
        "player_type",
        "projected_fantasy_points",
        "projection_source",
        "projection_confidence",
        "injury_status",
        "availability_note",
    ]

    for title, group in groups.items():
        st.subheader(title)
        if group.empty:
            st.info(f"No players in {title.lower()}.")
        else:
            columns = [column for column in display_columns if column in group.columns]
            st.dataframe(
                clean_dataframe_for_streamlit(group[columns].round(3)),
                width="stretch",
            )


def show_split_pickup_recommendations(
    recommendations: pd.DataFrame,
    selected_min_gain: float,
    show_no_gain_moves: bool,
) -> None:
    """Show Add Only and Add/Drop moves in separate tables."""

    st.write(
        "Add-only moves are safer because you do not lose a rostered player. "
        "Add/drop moves can improve today but carry season-long roster risk."
    )
    if recommendations.empty:
        st.info("No one-move pickup recommendation rows were created.")
        return

    filtered = recommendations[
        recommendations["projected_gain"] >= selected_min_gain
    ].copy()
    if not show_no_gain_moves:
        filtered = filtered[filtered["projected_gain"] > 0]

    add_only = filtered[filtered["move_type"] == "Add Only"].sort_values(
        "risk_adjusted_score",
        ascending=False,
    )
    add_drop = filtered[filtered["move_type"] == "Add/Drop"].sort_values(
        "risk_adjusted_score",
        ascending=False,
    )

    st.subheader("Best Add-Only Moves")
    if add_only.empty:
        st.info("No positive add-only moves found.")
    else:
        columns = [
            "add_player",
            "add_eligible_positions",
            "projected_gain",
            "risk_adjusted_score",
            "risk_adjusted_label",
            "new_starting_total",
            "recommendation",
            "add_projection_source",
            "add_projection_confidence",
        ]
        st.dataframe(clean_dataframe_for_streamlit(add_only[columns].round(3)), width="stretch")

    st.subheader("Best Add/Drop Moves")
    if add_drop.empty:
        st.info("No positive add/drop moves found.")
    else:
        columns = [
            "add_player",
            "add_eligible_positions",
            "drop_player",
            "drop_eligible_positions",
            "drop_long_term_value",
            "drop_keeper_status",
            "drop_risk",
            "projected_gain",
            "risk_adjusted_score",
            "risk_adjusted_label",
            "new_starting_total",
            "recommendation",
        ]
        visible_columns = [column for column in columns if column in add_drop.columns]
        st.dataframe(
            clean_dataframe_for_streamlit(add_drop[visible_columns].round(3)),
            width="stretch",
        )


def show_multi_add_scenarios(
    scenarios: pd.DataFrame,
    show_no_gain_moves: bool,
) -> None:
    """Display multi-add combinations when multiple active spots are open."""

    st.subheader("Best Multi-Add Scenarios")
    st.write(
        "Multi-add scenarios are only evaluated when you have more than one "
        "open active roster spot. These moves do not require dropping a player."
    )
    if scenarios.empty:
        st.info("No multi-add scenarios were found.")
        return

    if not show_no_gain_moves:
        scenarios = scenarios[scenarios["projected_gain"] > 0]

    if scenarios.empty:
        st.info("No positive multi-add scenarios found.")
        return

    columns = [
        "add_players",
        "add_eligible_positions",
        "number_of_adds",
        "projected_gain",
        "risk_adjusted_score",
        "risk_adjusted_label",
        "recommendation",
    ]
    st.dataframe(clean_dataframe_for_streamlit(scenarios[columns].round(3)), width="stretch")


def show_replacement_value_by_position(
    roster_projection_table: pd.DataFrame,
    available_projection_table: pd.DataFrame,
    master_pickup_table: pd.DataFrame,
) -> None:
    """Show optimized roster-versus-waiver value by exact hitter slot."""

    st.subheader("Optimized Replacement Value by Position")
    st.write(
        "This compares your optimized current hitter lineup against an optimized "
        "available-player lineup. Each available player can only be used once."
    )
    available_hitter_count = len(available_projection_table)
    available_with_projection_count = int(
        (
            ~available_projection_table["projection_source"].isin(
                ["Missing", "Unavailable"]
            )
        ).sum()
    ) if not available_projection_table.empty else 0
    eligible_count = int(
        available_projection_table["eligible_positions"].apply(
            lambda value: bool(set(split_positions(value)) & STARTING_HITTER_POSITIONS)
        ).sum()
    ) if not available_projection_table.empty else 0

    columns = st.columns(3)
    columns[0].metric("Available Hitters Loaded", available_hitter_count)
    columns[1].metric("Available Hitters With Projections", available_with_projection_count)
    columns[2].metric("Eligible For Starting Slots", eligible_count)

    if available_projection_table.empty:
        st.info("No available hitters are loaded for replacement-value analysis.")
        return

    replacement_table = calculate_replacement_value_by_position(
        roster_projection_table,
        available_projection_table,
        master_pickup_table,
    )
    st.dataframe(clean_dataframe_for_streamlit(replacement_table.round(3)), width="stretch")


def log_timing(timing_rows: list[dict], label: str, start_time: float) -> float:
    """Record elapsed time for one Top Pickups page phase."""

    elapsed_seconds = perf_counter() - start_time
    timing_rows.append(
        {
            "phase": label,
            "seconds": round(elapsed_seconds, 3),
        }
    )
    return perf_counter()


@st.cache_data(ttl=60 * 60)
def load_cached_player_projection_table(
    availability_dataframe=None,
    force_refresh_odds: bool = False,
    include_unavailable_players: bool = False,
    include_players_without_games: bool = False,
) -> pd.DataFrame:
    """Build and cache the unified projection table for one hour."""

    return build_player_projection_table(
        availability_dataframe,
        force_refresh_odds=force_refresh_odds,
        include_unavailable_players=include_unavailable_players,
        include_players_without_games=include_players_without_games,
    )


st.title("Top Pickups Command Center")
st.caption(
    "Start with the action plan, then review lineup impact, pickup options, "
    "and data quality below."
)
action_plan_container = st.container()

selected_data_source = st.radio(
    "Player data source",
    ["ESPN League Data", "CSV Uploads"],
    horizontal=True,
)
uploaded_available_players = None
uploaded_roster = None
refresh_espn_data = False

with st.expander("Uploads and Settings", expanded=True):
    if selected_data_source == "CSV Uploads":
        st.write(
            "Upload your available-player list and roster, or download the "
            "templates and fill them out first."
        )
        st.warning(
            "LF, CF, RF, and DH are treated as exact positions. Do not enter OF "
            "unless your league actually uses OF."
        )
        template_columns = st.columns(2)
        template_columns[0].download_button(
            "Download available players CSV template",
            data=build_available_players_template().to_csv(index=False),
            file_name="available_players_template.csv",
            mime="text/csv",
        )
        template_columns[1].download_button(
            "Download roster CSV template",
            data=build_roster_template().to_csv(index=False),
            file_name="roster_template.csv",
            mime="text/csv",
        )
        upload_columns = st.columns(2)
        uploaded_available_players = upload_columns[0].file_uploader(
            "Upload available players CSV",
            type=["csv"],
        )
        uploaded_roster = upload_columns[1].file_uploader(
            "Upload your current roster CSV",
            type=["csv"],
        )
    else:
        st.write(
            "Use ESPN League Data to populate your current roster and available "
            "free agents automatically. Filters use cached ESPN data until you refresh."
        )
        refresh_espn_data = st.button("Refresh ESPN Data")


try:
    page_start_time = perf_counter()
    timing_rows = []
    profile_counts = {
        "free_agents_loaded": 0,
        "free_agents_after_player_type_filter": 0,
        "available_hitters": 0,
        "free_agents_after_injury_filter": 0,
        "free_agents_after_game_today_filter": 0,
        "free_agents_after_no_props_filter": 0,
        "free_agents_sent_to_projection_matching": 0,
        "free_agents_matched_to_projections": 0,
        "free_agents_unmatched_to_projections": 0,
        "hitters_with_game_today": 0,
        "hitters_with_sportsbook_props": 0,
        "stat_fallback_only_hitters": 0,
        "unmatched_sportsbook_projection_players": 0,
        "espn_free_agents_matched_to_sportsbook": 0,
        "hitters_excluded_no_game": 0,
        "hitters_excluded_unavailable": 0,
        "available_hitters_with_projections": 0,
        "hitters_eligible_for_optimization": 0,
        "available_hitter_candidates_simulated": 0,
        "add_only_simulations_executed": 0,
        "add_drop_simulations_executed": 0,
        "multi_add_simulations_executed": 0,
    }
    roster_dataframe = None
    available_players_dataframe = None
    has_roster_data = False
    has_available_data = False
    show_roster_protection_review = False
    espn_cache = get_cached_espn_top_pickups_data()

    st.sidebar.warning("Live refresh may consume API credits.")
    refresh_live_odds_data = st.sidebar.button("Refresh live odds data")
    include_unavailable_players = st.sidebar.checkbox(
        "Include injured/unavailable players",
        value=False,
    )
    include_players_without_games = st.sidebar.checkbox(
        "Include players without games today",
        value=False,
    )
    include_stat_fallback_only_hitters = st.sidebar.checkbox(
        "Include stat-fallback-only hitters in recommendations",
        value=False,
    )
    treat_bench_hitters_as_droppable = st.sidebar.checkbox(
        "Treat bench hitters as droppable",
        value=True,
        help=(
            "Development safety setting: healthy bench hitters without Core or "
            "undroppable protection can be evaluated as drop candidates."
        ),
    )

    if refresh_live_odds_data:
        load_cached_player_projection_table.clear()

    if selected_data_source == "ESPN League Data":
        if refresh_espn_data or espn_cache is None:
            try:
                phase_start = perf_counter()
                espn_cache = load_espn_top_pickups_data()
                log_timing(timing_rows, "ESPN load", phase_start)
                cache_espn_top_pickups_data(espn_cache)
                st.success("Loaded ESPN roster and free-agent data.")
            except ESPNFantasyError as error:
                espn_cache = {
                    "roster_dataframe": pd.DataFrame(),
                    "available_players_dataframe": pd.DataFrame(),
                    "last_refresh_time": None,
                    "error": error,
                }
                cache_espn_top_pickups_data(espn_cache)
                st.warning("ESPN data could not load. Check `.env`, then refresh ESPN data.")
                if error.debug_info:
                    with st.expander("ESPN Debug Details"):
                        st.json(error.debug_info)

        if espn_cache is not None and espn_cache.get("error") is None:
            roster_dataframe = espn_cache["roster_dataframe"].copy()
            available_players_dataframe = espn_cache["available_players_dataframe"].copy()
            profile_counts["free_agents_loaded"] = len(available_players_dataframe)
            has_roster_data = not roster_dataframe.empty
            has_available_data = not available_players_dataframe.empty
            show_roster_protection_review = st.sidebar.checkbox(
                "Show roster protection review",
                value=True,
            )
        elif espn_cache is not None and espn_cache.get("error") is not None:
            st.warning(str(espn_cache["error"]))
        elif espn_cache is not None:
            timing_rows.append({"phase": "ESPN load", "seconds": 0.0})

    if selected_data_source == "CSV Uploads" and uploaded_available_players is not None:
        phase_start = perf_counter()
        available_players_dataframe = pd.read_csv(uploaded_available_players)
        log_timing(timing_rows, "CSV available-player load", phase_start)
        if "player" not in available_players_dataframe.columns:
            st.warning("The available-player CSV must include a `player` column.")
            available_players_dataframe = None
        else:
            available_players_dataframe = add_player_matching_columns(available_players_dataframe)
            profile_counts["free_agents_loaded"] = len(available_players_dataframe)
            has_available_data = True

    if selected_data_source == "CSV Uploads" and uploaded_roster is not None:
        phase_start = perf_counter()
        roster_dataframe = pd.read_csv(uploaded_roster)
        log_timing(timing_rows, "CSV roster load", phase_start)
        if "player" not in roster_dataframe.columns:
            st.warning("The roster CSV must include a `player` column.")
            roster_dataframe = None
        else:
            roster_dataframe = add_player_matching_columns(roster_dataframe)
            has_roster_data = True
            show_roster_protection_review = st.sidebar.checkbox(
                "Show roster protection review",
                value=True,
            )

    show_data_source_status(
        selected_data_source,
        roster_dataframe,
        available_players_dataframe,
        espn_cache if selected_data_source == "ESPN League Data" else None,
    )

    roster_summary = None
    if has_roster_data:
        sp_start_context = show_weekly_sp_start_tracker()
        roster_summary = show_roster_flexibility_summary(
            roster_dataframe,
            sp_start_context,
        )
    else:
        with action_plan_container:
            st.header("Today's Action Plan")
            st.info("Load a roster and available-player source to generate pickup recommendations.")

    position_enrichment_dataframe = build_position_enrichment_dataframe(
        available_players_dataframe,
        roster_dataframe,
    )
    phase_start = perf_counter()
    pickup_table = load_cached_player_projection_table(
        position_enrichment_dataframe,
        refresh_live_odds_data,
        True,
        include_players_without_games,
    )
    log_timing(timing_rows, "Projection matching", phase_start)
    odds_data_source = pickup_table.attrs.get("odds_data_source", {})
    pickup_table = ensure_pickup_table_columns(pickup_table)

    st.caption(
        "Data source: "
        f"events = {odds_data_source.get('events') or 'Unknown'}, "
        f"hitter props = {odds_data_source.get('hitter_props') or 'Unknown'}"
    )

    if pickup_table.empty:
        st.info(
            "No sportsbook projection rows are available right now. The page "
            "will still show roster/free-agent review sections."
        )

    projection_base_table = ensure_pickup_table_columns(
        add_player_matching_columns(pickup_table)
    )
    sportsbook_projection_table = get_actual_sportsbook_projection_table(
        projection_base_table
    )

    # ESPN/CSV rows are the source of truth for roster and waiver status. We
    # attach sportsbook projections to those rows before fallback/watchlist
    # logic runs so a player like Heriberto Hernandez is not split into two
    # separate rows.
    roster_merged_table, roster_match_diagnostics = merge_projection_data_onto_player_source(
        roster_dataframe,
        projection_base_table,
        "On My Roster",
        min_score=90,
    )
    available_merged_table, available_match_diagnostics = merge_projection_data_onto_player_source(
        available_players_dataframe,
        projection_base_table,
        "Available",
        min_score=90,
    )
    profile_counts["free_agents_sent_to_projection_matching"] = (
        len(available_players_dataframe)
        if available_players_dataframe is not None
        else 0
    )

    source_match_keys = set()
    for source_table in [roster_merged_table, available_merged_table]:
        if not source_table.empty:
            source_match_keys.update(source_table["player_match_key"].dropna())

    projection_only_table = projection_base_table[
        ~projection_base_table["player_match_key"].isin(source_match_keys)
    ].copy()
    if not projection_only_table.empty:
        projection_only_table["roster_status"] = "Unknown"

    master_pickup_table = ensure_pickup_table_columns(
        pd.concat(
            [roster_merged_table, available_merged_table, projection_only_table],
            ignore_index=True,
            sort=False,
        )
    )
    master_pickup_table["sportsbook_match_score"] = pd.to_numeric(
        master_pickup_table["sportsbook_match_score"],
        errors="coerce",
    ).fillna(0)
    sportsbook_match_diagnostics = pd.concat(
        [roster_match_diagnostics, available_match_diagnostics],
        ignore_index=True,
        sort=False,
    )
    if not sportsbook_match_diagnostics.empty:
        sportsbook_match_diagnostics["match_score"] = pd.to_numeric(
            sportsbook_match_diagnostics["match_score"],
            errors="coerce",
        ).fillna(0)

    matched_sportsbook_players = set()
    if not sportsbook_match_diagnostics.empty:
        matched_sportsbook_players = set(
            sportsbook_match_diagnostics.loc[
                as_bool_series(sportsbook_match_diagnostics["has_sportsbook_props"]),
                "matched_projection_player",
            ]
            .dropna()
            .astype(str)
        )

    unmatched_sportsbook_rows = sportsbook_projection_table[
        ~sportsbook_projection_table["player"].isin(matched_sportsbook_players)
    ].copy()
    if not unmatched_sportsbook_rows.empty:
        unmatched_diagnostics = pd.DataFrame(
            [
                {
                    "espn_player": "",
                    "espn_player_id": "",
                    "normalized_player_name": "",
                    "player_match_key": "",
                    "matched_projection_player": row.get("player", ""),
                    "matched_projection_source": row.get("projection_source", ""),
                    "matched_bookmaker_count": row.get("bookmaker_count", 0),
                    "matched_markets_available": row.get("markets_available", ""),
                    "has_sportsbook_props": True,
                    "match_method": "unmatched sportsbook projection",
                    "match_score": 0,
                }
                for _, row in unmatched_sportsbook_rows.iterrows()
            ]
        )
        sportsbook_match_diagnostics = pd.concat(
            [sportsbook_match_diagnostics, unmatched_diagnostics],
            ignore_index=True,
            sort=False,
        )

    if not sportsbook_match_diagnostics.empty:
        available_match_mask = (
            sportsbook_match_diagnostics["espn_player"].astype(str).str.strip() != ""
        ) & as_bool_series(sportsbook_match_diagnostics["has_sportsbook_props"])
        profile_counts["espn_free_agents_matched_to_sportsbook"] = int(
            (
                available_match_mask
                & sportsbook_match_diagnostics["espn_player"].isin(
                    available_merged_table["player"]
                    if not available_merged_table.empty
                    else []
                )
            ).sum()
        )
        profile_counts["unmatched_sportsbook_projection_players"] = int(
            (
                sportsbook_match_diagnostics["match_method"]
                == "unmatched sportsbook projection"
            ).sum()
        )

    roster_group_table = pd.DataFrame()
    roster_projection_table = pd.DataFrame()
    available_projection_table = pd.DataFrame()
    pickup_recommendations = pd.DataFrame()
    multi_add_scenarios = pd.DataFrame()
    projection_coverage_summary = {
        "roster_players_matched": 0,
        "roster_players_missing": 0,
        "free_agents_matched": 0,
        "free_agents_missing": 0,
    }

    if has_roster_data:
        roster_group_table = master_pickup_table[
            master_pickup_table["roster_status"] == "On My Roster"
        ].copy()
        roster_projection_table = filter_hitter_optimizer_pool(
            roster_group_table,
            include_unavailable_players,
            include_players_without_games,
            include_stat_fallback_only_hitters,
        )

    if has_available_data:
        phase_start = perf_counter()
        raw_available_projection_table = master_pickup_table[
            master_pickup_table["roster_status"] == "Available"
        ].copy()
        strict_player_type_hitter_mask = (
            raw_available_projection_table["player_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("hitter")
        )
        profile_counts["free_agents_after_player_type_filter"] = int(
            strict_player_type_hitter_mask.sum()
        )
        raw_available_hitters = raw_available_projection_table[
            raw_available_projection_table.apply(row_is_hitter, axis=1)
        ].copy()
        has_game_today_mask = as_bool_series(raw_available_hitters["has_game_today"])
        profile_counts["available_hitters"] = len(raw_available_hitters)
        unavailable_mask = (
            raw_available_hitters.apply(is_player_unavailable, axis=1)
            if not raw_available_hitters.empty
            else pd.Series(dtype=bool)
        )
        profile_counts["hitters_with_game_today"] = int(
            has_game_today_mask.sum()
        ) if not raw_available_hitters.empty else 0
        profile_counts["free_agents_after_injury_filter"] = int(
            (~unavailable_mask).sum()
        ) if not raw_available_hitters.empty else 0
        profile_counts["free_agents_after_game_today_filter"] = int(
            (has_game_today_mask & ~unavailable_mask).sum()
        ) if not raw_available_hitters.empty else 0
        sportsbook_props_mask = as_bool_series(
            raw_available_hitters["has_sportsbook_props"]
        )
        profile_counts["hitters_with_sportsbook_props"] = int(
            sportsbook_props_mask.sum()
        ) if not raw_available_hitters.empty else 0
        stat_fallback_mask = (
            has_game_today_mask
            & ~unavailable_mask
            & ~sportsbook_props_mask
        )
        no_props_filter_mask = has_game_today_mask & ~unavailable_mask & (
            sportsbook_props_mask
            | (include_stat_fallback_only_hitters & stat_fallback_mask)
        )
        profile_counts["free_agents_after_no_props_filter"] = int(
            no_props_filter_mask.sum()
        ) if not raw_available_hitters.empty else 0
        profile_counts["stat_fallback_only_hitters"] = int(
            stat_fallback_mask.sum()
        ) if not raw_available_hitters.empty else 0
        profile_counts["hitters_excluded_no_game"] = int(
            (~has_game_today_mask).sum()
        ) if not raw_available_hitters.empty else 0
        profile_counts["hitters_excluded_unavailable"] = int(
            unavailable_mask.sum()
        ) if not raw_available_hitters.empty else 0
        unmatched_projection_mask = raw_available_hitters["projection_source"].isin(
            ["Missing", "Unknown"]
        )
        profile_counts["free_agents_unmatched_to_projections"] = int(
            unmatched_projection_mask.sum()
        ) if not raw_available_hitters.empty else 0
        profile_counts["free_agents_matched_to_projections"] = int(
            (~unmatched_projection_mask).sum()
        ) if not raw_available_hitters.empty else 0
        profile_counts["available_hitters_with_projections"] = int(
            (
                ~raw_available_hitters["projection_source"].isin(
                    ["Missing", "Unavailable", "No game today"]
                )
            ).sum()
        ) if not raw_available_hitters.empty else 0

        available_projection_table = raw_available_projection_table.copy()
        available_projection_table = filter_hitter_optimizer_pool(
            available_projection_table,
            include_unavailable_players,
            include_players_without_games,
            include_stat_fallback_only_hitters,
        )
        profile_counts["hitters_eligible_for_optimization"] = len(
            available_projection_table
        )
        reduced_available_projection_table = reduce_available_hitter_candidates(
            available_projection_table,
        )
        multi_add_candidate_table = reduce_available_hitter_candidates(
            available_projection_table,
            per_position_limit=5,
            total_limit=25,
        )
        profile_counts["available_hitter_candidates_simulated"] = len(
            reduced_available_projection_table
        )
        log_timing(timing_rows, "Available hitter filtering", phase_start)
    else:
        raw_available_hitters = pd.DataFrame()
        reduced_available_projection_table = pd.DataFrame()
        multi_add_candidate_table = pd.DataFrame()

    if has_roster_data and has_available_data:
        progress_placeholder = st.empty()

        def show_optimizer_progress(progress_payload: dict) -> None:
            """Update the page every 50 simulations during brute-force runs."""

            progress_placeholder.caption(
                "Optimizer progress: "
                f"{progress_payload.get('phase', 'simulation')} "
                f"{progress_payload.get('simulation_count', 0)} simulations"
            )

        phase_start = perf_counter()
        pickup_recommendations = evaluate_single_hitter_pickups(
            roster_projection_table,
            reduced_available_projection_table,
            active_roster_df=roster_group_table,
            treat_bench_hitters_as_droppable=treat_bench_hitters_as_droppable,
            progress_callback=show_optimizer_progress,
        )
        profile_counts["add_drop_simulations_executed"] = pickup_recommendations.attrs.get(
            "add_drop_simulation_count",
            0,
        )
        profile_counts["add_only_simulations_executed"] = pickup_recommendations.attrs.get(
            "add_only_simulation_count",
            0,
        )
        log_timing(timing_rows, "Recommendation generation", phase_start)
        if roster_summary is None:
            roster_summary = get_roster_flexibility_summary(roster_group_table)
        if roster_summary["open_active_spots"] > 1:
            phase_start = perf_counter()
            multi_add_scenarios = evaluate_multi_add_hitter_scenarios(
                roster_projection_table,
                multi_add_candidate_table,
                active_roster_df=roster_group_table,
                progress_callback=show_optimizer_progress,
            )
            profile_counts["multi_add_simulations_executed"] = multi_add_scenarios.attrs.get(
                "simulation_count",
                0,
            )
            log_timing(timing_rows, "Multi-add generation", phase_start)
        projection_coverage_summary = build_projection_coverage_summary(
            roster_projection_table,
            raw_available_hitters,
        )
        progress_placeholder.empty()

    best_move = get_best_positive_move(pickup_recommendations, multi_add_scenarios)
    if has_roster_data and has_available_data:
        with action_plan_container:
            show_todays_action_plan(best_move)

    with st.expander("Top Pickups Performance Profile", expanded=True):
        st.write(
            "These timings identify where the page spends time. The optimizer "
            "uses a reduced candidate pool before simulations so it does not "
            "test every free agent against every roster player."
        )
        metric_columns = st.columns(4)
        metric_columns[0].metric(
            "Free Agents Loaded",
            profile_counts["free_agents_loaded"],
        )
        metric_columns[1].metric(
            "Available Hitters",
            profile_counts["available_hitters"],
        )
        metric_columns[2].metric(
            "Hitters With Projections",
            profile_counts["available_hitters_with_projections"],
        )
        metric_columns[3].metric(
            "Candidates Simulated",
            profile_counts["available_hitter_candidates_simulated"],
        )
        daily_columns = st.columns(6)
        daily_columns[0].metric(
            "Hitters With Game Today",
            profile_counts["hitters_with_game_today"],
        )
        daily_columns[1].metric(
            "Hitters With Sportsbook Props",
            profile_counts["hitters_with_sportsbook_props"],
        )
        daily_columns[2].metric(
            "Stat Fallback Only",
            profile_counts["stat_fallback_only_hitters"],
        )
        daily_columns[3].metric(
            "Excluded: No Game",
            profile_counts["hitters_excluded_no_game"],
        )
        daily_columns[4].metric(
            "Excluded: Injury/Unavailable",
            profile_counts["hitters_excluded_unavailable"],
        )
        daily_columns[5].metric(
            "Eligible For Optimization",
            profile_counts["hitters_eligible_for_optimization"],
        )
        pipeline_columns = st.columns(4)
        pipeline_columns[0].metric(
            "After Player Type Filter",
            profile_counts["free_agents_after_player_type_filter"],
        )
        pipeline_columns[1].metric(
            "After Injury Filter",
            profile_counts["free_agents_after_injury_filter"],
        )
        pipeline_columns[2].metric(
            "After Game-Today Filter",
            profile_counts["free_agents_after_game_today_filter"],
        )
        pipeline_columns[3].metric(
            "After No-Props Filter",
            profile_counts["free_agents_after_no_props_filter"],
        )
        match_pipeline_columns = st.columns(3)
        match_pipeline_columns[0].metric(
            "Sent To Projection Matching",
            profile_counts["free_agents_sent_to_projection_matching"],
        )
        match_pipeline_columns[1].metric(
            "Matched To Projections",
            profile_counts["free_agents_matched_to_projections"],
        )
        match_pipeline_columns[2].metric(
            "Unmatched To Projections",
            profile_counts["free_agents_unmatched_to_projections"],
        )
        simulation_columns = st.columns(3)
        simulation_columns[0].metric(
            "Add-Only Simulations",
            profile_counts["add_only_simulations_executed"],
        )
        simulation_columns[1].metric(
            "Add/Drop Simulations",
            profile_counts["add_drop_simulations_executed"],
        )
        simulation_columns[2].metric(
            "Multi-Add Simulations",
            profile_counts["multi_add_simulations_executed"],
        )
        match_columns = st.columns(2)
        match_columns[0].metric(
            "ESPN Free Agents Matched To Sportsbook",
            profile_counts["espn_free_agents_matched_to_sportsbook"],
        )
        match_columns[1].metric(
            "Unmatched Sportsbook Projection Players",
            profile_counts["unmatched_sportsbook_projection_players"],
        )
        st.dataframe(
            clean_dataframe_for_streamlit(pd.DataFrame(timing_rows)),
            width="stretch",
        )
    st.header("Best Current Hitter Lineup")
    st.write("This optimizes only your current roster. Available players are not used here.")
    if not has_roster_data:
        st.info("Load your roster to optimize the current hitter lineup.")
    elif roster_projection_table.empty:
        st.info("No healthy rostered hitters with exact hitter eligibility were found.")
    else:
        lineup_df, bench_df, warning_message = optimize_hitter_lineup(roster_projection_table)
        if warning_message:
            st.warning(warning_message)
        st.metric(
            "Projected Starting Hitter Total",
            round(lineup_df["projected_fantasy_points"].sum(), 2),
        )
        slot_order = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
        lineup_display = lineup_df.copy()
        lineup_display["optimized_slot"] = lineup_display["roster_slot"]
        lineup_display["current_roster_spot"] = lineup_display.get("roster_spot", "")
        lineup_display["slot_sort_order"] = lineup_display["optimized_slot"].map(
            {slot: index for index, slot in enumerate(slot_order)}
        )
        lineup_display = lineup_display.sort_values(
            "slot_sort_order",
            na_position="last",
        )
        lineup_columns = [
            column
            for column in [
                "optimized_slot",
                "player",
                "eligible_positions",
                "current_roster_spot",
                "projected_fantasy_points",
                "projection_source",
                "projection_confidence",
                "injury_status",
                "availability_note",
            ]
            if column in lineup_display.columns
        ]
        st.subheader("Optimized Starting Lineup")
        st.write(
            "Optimized slot is where the app recommends starting the player "
            "today. Current roster spot is where ESPN currently has him."
        )
        st.dataframe(
            clean_dataframe_for_streamlit(lineup_display[lineup_columns].round(3)),
            width="stretch",
        )
        show_grouped_roster_tables(lineup_df, roster_group_table)
    st.header("Pickup Recommendations")
    if not (has_roster_data and has_available_data):
        st.info("Load both roster and available-player data to calculate recommendations.")
    else:
        selected_min_gain = st.sidebar.slider(
            "Minimum Projected Gain",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
        )
        show_no_gain_moves = st.sidebar.checkbox("Show no-gain moves", value=False)
        show_projection_coverage_summary(projection_coverage_summary)
        if roster_summary and roster_summary["open_active_spots"] == 0:
            st.info(
                "Your active roster is full, so pickup recommendations are "
                "evaluated as add/drop moves."
            )

        droppable_exists = (
            not roster_projection_table.empty
            and roster_projection_table.apply(
                lambda row: is_droppable_player(
                    row,
                    treat_bench_hitters_as_droppable=treat_bench_hitters_as_droppable,
                ),
                axis=1,
            ).any()
        )
        if not droppable_exists:
            st.info(
                "No add/drop moves can be evaluated because no rostered players "
                "are marked droppable or streamable."
            )

        if pickup_recommendations.empty and multi_add_scenarios.empty:
            st.info(
                "No pickup recommendation rows were created. Check projection "
                "coverage and droppable candidate review below for the exact data issue."
            )
        else:
            show_lineup_impact(
                roster_projection_table,
                available_projection_table,
                best_move,
            )
            if roster_summary and roster_summary["open_active_spots"] > 1:
                show_multi_add_scenarios(multi_add_scenarios, show_no_gain_moves)
            show_split_pickup_recommendations(
                pickup_recommendations,
                selected_min_gain,
                show_no_gain_moves,
            )
    st.header("Ranked Available Hitters with Sportsbook Props")
    st.write(
        "This table prioritizes available hitters with sportsbook prop lines. "
        "Those lines are the strongest daily signal that the player is expected "
        "to have a role today."
    )
    ranked_available = master_pickup_table[
        master_pickup_table["roster_status"] == "Available"
    ].copy()
    available_pitchers = ranked_available[ranked_available.apply(row_is_pitcher, axis=1)].copy()
    ranked_available = ranked_available[
        ranked_available["player_type"].astype(str).str.strip().str.lower().eq("hitter")
    ].copy()
    stat_fallback_watchlist = ranked_available[
        as_bool_series(ranked_available["has_game_today"])
        & ~ranked_available.apply(is_player_unavailable, axis=1)
        & ~as_bool_series(ranked_available["has_sportsbook_props"])
        & ranked_available["projection_source"].isin(
            ["Stat-based fallback", "Unknown"]
        )
    ].copy()
    player_search = ""
    ranked_available["included_in_ranked_sportsbook_table"] = ranked_available.apply(
        is_ranked_sportsbook_table_candidate,
        axis=1,
    )
    ranked_available["exclusion_reason"] = ranked_available.apply(
        get_ranked_sportsbook_exclusion_reason,
        axis=1,
    )
    ranked_available = ranked_available[
        as_bool_series(ranked_available["included_in_ranked_sportsbook_table"])
    ].copy()
    ranked_available = ranked_available[
        ~ranked_available["projection_source"].isin(["Missing", "Unknown"])
    ].copy()

    if ranked_available.empty:
        st.info("No available hitters with sportsbook props match the current filters.")
    else:
        ranked_available["tier"] = ranked_available["projected_fantasy_points"].apply(
            get_pickup_tier
        )
        bookmaker_options = sorted(ranked_available["bookmaker"].dropna().unique())
        selected_bookmaker = st.sidebar.selectbox(
            "Bookmaker",
            ["All Bookmakers"] + bookmaker_options,
        )
        max_points = float(ranked_available["projected_fantasy_points"].max())
        selected_min_points = st.sidebar.slider(
            "Minimum Projected Fantasy Points",
            min_value=0.0,
            max_value=max(max_points, 0.0),
            value=0.0,
            step=0.5,
        )
        player_search = st.sidebar.text_input("Player Search")
        position_options = sorted(
            {
                position
                for value in ranked_available["eligible_positions"].dropna()
                for position in split_positions(value)
                if position in STARTING_HITTER_POSITIONS
            }
        )
        selected_position = st.sidebar.selectbox(
            "Position",
            ["All Positions"] + position_options,
        )

        if selected_bookmaker != "All Bookmakers":
            ranked_available = ranked_available[
                ranked_available["bookmaker"] == selected_bookmaker
            ]
        ranked_available = ranked_available[
            ranked_available["projected_fantasy_points"] >= selected_min_points
        ]
        if player_search:
            ranked_available = ranked_available[
                ranked_available["player"].str.contains(player_search, case=False, na=False)
            ]
            if ranked_available.empty:
                st.info("No available hitters with sportsbook props match the current player search.")
        if selected_position != "All Positions":
            ranked_available = ranked_available[
                ranked_available["eligible_positions"].apply(
                    lambda value: player_has_position(value, selected_position)
                )
            ]

        ranked_available = ranked_available.sort_values(
            "projected_fantasy_points",
            ascending=False,
        )
        if ranked_available.empty:
            st.info("No available hitters with sportsbook props match the current filters.")
        else:
            visible_columns = [
                column for column in TOP_PICKUPS_COLUMNS if column in ranked_available.columns
            ]
            st.dataframe(
                clean_dataframe_for_streamlit(ranked_available[visible_columns].round(3)),
                width="stretch",
            )

    st.header("Stat Fallback Watchlist")
    st.write(
        "These players have a game today but no sportsbook hitter props. They "
        "may not be expected to start, so they are excluded from main "
        "recommendations by default."
    )
    if stat_fallback_watchlist.empty:
        st.info("No stat-fallback-only available hitters found.")
    else:
        stat_fallback_watchlist = stat_fallback_watchlist.sort_values(
            "projected_fantasy_points",
            ascending=False,
        )
        watchlist_columns = [
            column
            for column in [
                "player",
                "eligible_positions",
                "pro_team",
                "projected_fantasy_points",
                "projection_source",
                "projection_confidence",
                "has_game_today",
                "game_today_note",
                "is_available_today",
                "availability_note",
                "fallback_projection_note",
            ]
            if column in stat_fallback_watchlist.columns
        ]
        st.dataframe(
            clean_dataframe_for_streamlit(
                stat_fallback_watchlist[watchlist_columns].round(3)
            ),
            width="stretch",
        )

    if not available_pitchers.empty:
        with st.expander("Available Pitchers"):
            st.write("Pitchers are shown for review but are not used by the hitter optimizer.")
            pitcher_columns = [
                column
                for column in [
                    "player",
                    "eligible_positions",
                    "player_type",
                    "roster_status",
                    "projection_source",
                    "projection_confidence",
                ]
                if column in available_pitchers.columns
            ]
            st.dataframe(
                clean_dataframe_for_streamlit(available_pitchers[pitcher_columns]),
                width="stretch",
            )

    unavailable_free_agents = master_pickup_table[
        (master_pickup_table["roster_status"] == "Available")
        & master_pickup_table.apply(row_is_hitter, axis=1)
        & (
            master_pickup_table.apply(is_player_unavailable, axis=1)
            | ~as_bool_series(master_pickup_table["has_game_today"])
            | ~as_bool_series(master_pickup_table["has_sportsbook_props"])
        )
    ].copy()
    if not unavailable_free_agents.empty:
        with st.expander("Unavailable / No Game / No Props Review"):
            st.write(
                "These free-agent hitters are excluded by default because they "
                "are injured/unavailable, their MLB team does not play today, "
                "or they do not have sportsbook hitter props."
            )
            review_columns = [
                "player",
                "eligible_positions",
                "pro_team",
                "has_sportsbook_props",
                "injury_status",
                "status",
                "player_notes",
                "lineup_status",
                "fantasy_status",
                "has_game_today",
                "game_today_note",
                "is_available_today",
                "availability_note",
                "projection_source",
                "projection_confidence",
                "fallback_projection_note",
            ]
            visible_columns = [
                column
                for column in review_columns
                if column in unavailable_free_agents.columns
            ]
            st.dataframe(
                clean_dataframe_for_streamlit(
                    unavailable_free_agents[visible_columns].sort_values("player")
                ),
                width="stretch",
            )

    with st.expander("Availability Classification Debug", expanded=True):
        st.write(
            "This shows the exact status fields checked for available hitters. "
            "Blank, null, nan, Active, Available, Normal, and Healthy values are ignored."
        )
        availability_debug_source = (
            raw_available_hitters.copy()
            if "raw_available_hitters" in locals() and not raw_available_hitters.empty
            else pd.DataFrame()
        )
        if availability_debug_source.empty:
            st.info("No available hitters are loaded for availability debugging.")
        else:
            availability_details = availability_debug_source.apply(
                classify_player_availability,
                axis=1,
                result_type="expand",
            )
            availability_debug_source["raw_status_values_checked"] = availability_details[
                "raw_status_values_checked"
            ]
            availability_debug_source["unavailable_marker_detected"] = availability_details[
                "unavailable_marker_detected"
            ]
            availability_debug_source["is_available_today"] = ~availability_details[
                "is_unavailable"
            ]
            debug_columns = [
                column
                for column in [
                    "player",
                    "injury_status",
                    "status",
                    "roster_status",
                    "roster_spot",
                    "availability_note",
                    "raw_status_values_checked",
                    "unavailable_marker_detected",
                    "is_available_today",
                    "projected_fantasy_points",
                    "projection_source",
                ]
                if column in availability_debug_source.columns
            ]
            st.dataframe(
                clean_dataframe_for_streamlit(
                    availability_debug_source[debug_columns].head(50).round(3)
                ),
                width="stretch",
            )

    with st.expander("Sportsbook Match Debug", expanded=True):
        st.write(
            "This verifies whether ESPN free agents are joining to sportsbook "
            "projection rows. Matching tries player_match_key first, then "
            "normalized player name, then fuzzy matching at a safe threshold."
        )
        available_source_table = (
            add_player_matching_columns(available_players_dataframe)
            if available_players_dataframe is not None
            else pd.DataFrame()
        )
        available_source_hitters = (
            available_source_table[available_source_table.apply(row_is_hitter, axis=1)].copy()
            if not available_source_table.empty
            else pd.DataFrame()
        )
        available_diagnostics = (
            sportsbook_match_diagnostics[
                sportsbook_match_diagnostics["espn_player"].isin(
                    available_source_hitters["player"]
                    if not available_source_hitters.empty
                    else []
                )
            ].copy()
            if not sportsbook_match_diagnostics.empty
            else pd.DataFrame()
        )
        exact_normalized_count = (
            int((available_diagnostics["match_method"] == "exact normalized name").sum())
            if not available_diagnostics.empty
            else 0
        )
        exact_key_count = (
            int((available_diagnostics["match_method"] == "exact match key").sum())
            if not available_diagnostics.empty
            else 0
        )
        fuzzy_match_count = (
            int((available_diagnostics["match_method"] == "fuzzy").sum())
            if not available_diagnostics.empty
            else 0
        )
        unmatched_espn_free_agents = (
            int((~as_bool_series(available_diagnostics["has_sportsbook_props"])).sum())
            if not available_diagnostics.empty
            else len(available_source_hitters)
        )

        debug_metrics = st.columns(7)
        debug_metrics[0].metric(
            "Sportsbook Projection Rows",
            len(sportsbook_projection_table),
        )
        debug_metrics[1].metric(
            "ESPN FA Hitter Rows",
            len(available_source_hitters),
        )
        debug_metrics[2].metric("Exact Key Matches", exact_key_count)
        debug_metrics[3].metric(
            "Exact Normalized Matches",
            exact_normalized_count,
        )
        debug_metrics[4].metric("Fuzzy Matches", fuzzy_match_count)
        debug_metrics[5].metric(
            "Unmatched ESPN FAs",
            unmatched_espn_free_agents,
        )
        debug_metrics[6].metric(
            "Unmatched Sportsbook Players",
            profile_counts["unmatched_sportsbook_projection_players"],
        )

        exact_key_matched_free_agents = master_pickup_table[
            (master_pickup_table["roster_status"] == "Available")
            & master_pickup_table["sportsbook_match_method"].astype(str).eq(
                "exact match key"
            )
        ].copy()
        if not exact_key_matched_free_agents.empty:
            exact_key_matched_free_agents["matched_projection_player"] = (
                exact_key_matched_free_agents["matched_sportsbook_player"]
            )
            exact_key_matched_free_agents[
                "included_in_ranked_sportsbook_table"
            ] = exact_key_matched_free_agents.apply(
                is_ranked_sportsbook_table_candidate,
                axis=1,
            )
            exact_key_matched_free_agents["exclusion_reason"] = (
                exact_key_matched_free_agents.apply(
                    lambda row: get_ranked_sportsbook_exclusion_reason(
                        row,
                        player_search,
                    ),
                    axis=1,
                )
            )

        st.subheader("Exact-Key Matched ESPN Free Agents")
        st.write(
            "These rows matched ESPN free agents to sportsbook projections by "
            "player_match_key. If a matched player is not in the ranked table, "
            "the exclusion_reason column explains why."
        )
        if exact_key_matched_free_agents.empty:
            st.info("No exact-key matched ESPN free agents found.")
        else:
            matched_columns = [
                column
                for column in [
                    "player",
                    "matched_projection_player",
                    "projected_fantasy_points",
                    "projection_source",
                    "bookmaker_count",
                    "markets_available",
                    "has_sportsbook_props",
                    "has_game_today",
                    "is_available_today",
                    "player_type",
                    "included_in_ranked_sportsbook_table",
                    "exclusion_reason",
                ]
                if column in exact_key_matched_free_agents.columns
            ]
            st.dataframe(
                clean_dataframe_for_streamlit(
                    exact_key_matched_free_agents[matched_columns]
                    .sort_values("projected_fantasy_points", ascending=False)
                    .round(3)
                ),
                width="stretch",
            )

        st.subheader("Expected Player Match Checks")
        expected_debug_rows = []
        for expected_player in ["Tyler Stephenson", "Heriberto Hernandez", "Estury Ruiz"]:
            expected_key = build_player_match_key(expected_player)
            espn_rows = (
                available_source_table[
                    available_source_table["player_match_key"].eq(expected_key)
                ]
                if not available_source_table.empty
                else pd.DataFrame()
            )
            projection_rows = (
                projection_base_table[
                    projection_base_table["player_match_key"].eq(expected_key)
                ]
                if not projection_base_table.empty
                else pd.DataFrame()
            )
            final_rows = (
                master_pickup_table[
                    master_pickup_table["player_match_key"].eq(expected_key)
                    & master_pickup_table["roster_status"].eq("Available")
                ]
                if not master_pickup_table.empty
                else pd.DataFrame()
            )
            final_row = final_rows.iloc[0] if not final_rows.empty else {}
            expected_debug_rows.append(
                {
                    "player": expected_player,
                    "found_in_espn_free_agents": not espn_rows.empty,
                    "found_in_projection_table": not projection_rows.empty,
                    "match_method": final_row.get("sportsbook_match_method", ""),
                    "match_score": final_row.get("sportsbook_match_score", 0),
                    "matched_projection_player": final_row.get(
                        "matched_sportsbook_player",
                        "",
                    ),
                    "projected_fantasy_points": final_row.get(
                        "projected_fantasy_points",
                        "",
                    ),
                    "projection_source": final_row.get("projection_source", ""),
                    "has_sportsbook_props": final_row.get("has_sportsbook_props", ""),
                    "has_game_today": final_row.get("has_game_today", ""),
                    "is_available_today": final_row.get("is_available_today", ""),
                    "included_in_ranked_sportsbook_table": (
                        is_ranked_sportsbook_table_candidate(final_row)
                        if not final_rows.empty
                        else False
                    ),
                    "exclusion_reason": (
                        get_ranked_sportsbook_exclusion_reason(
                            final_row,
                            player_search,
                        )
                        if not final_rows.empty
                        else (
                            "not found in ESPN free agents"
                            if espn_rows.empty
                            else "not found in projection table"
                        )
                    ),
                }
            )
        st.dataframe(
            clean_dataframe_for_streamlit(pd.DataFrame(expected_debug_rows)),
            width="stretch",
        )

        debug_player_search = st.text_input(
            "Search match debug by player",
            value="",
            placeholder="Example: Heriberto Hernandez",
        )
        if debug_player_search:
            normalized_search = normalize_player_name(debug_player_search)
            match_key_search = build_player_match_key(debug_player_search)
            st.caption(
                f"Search normalized name: `{normalized_search}` | "
                f"match key: `{match_key_search}`"
            )

            espn_search_rows = (
                available_source_table[
                    available_source_table["player"].str.contains(
                        debug_player_search,
                        case=False,
                        na=False,
                    )
                    | (available_source_table["normalized_player_name"] == normalized_search)
                    | (available_source_table["player_match_key"] == match_key_search)
                ].copy()
                if not available_source_table.empty
                else pd.DataFrame()
            )
            sportsbook_search_rows = (
                projection_base_table[
                    projection_base_table["player"].str.contains(
                        debug_player_search,
                        case=False,
                        na=False,
                    )
                    | (projection_base_table["normalized_player_name"] == normalized_search)
                    | (projection_base_table["player_match_key"] == match_key_search)
                ].copy()
                if not projection_base_table.empty
                else pd.DataFrame()
            )
            final_match_rows = (
                master_pickup_table[
                    master_pickup_table["player"].str.contains(
                        debug_player_search,
                        case=False,
                        na=False,
                    )
                    | (master_pickup_table["normalized_player_name"] == normalized_search)
                    | (master_pickup_table["player_match_key"] == match_key_search)
                    | master_pickup_table["matched_sportsbook_player"].astype(str).str.contains(
                        debug_player_search,
                        case=False,
                        na=False,
                    )
                ].copy()
                if not master_pickup_table.empty
                else pd.DataFrame()
            )
            diagnostic_search_rows = (
                sportsbook_match_diagnostics[
                    sportsbook_match_diagnostics["espn_player"].astype(str).str.contains(
                        debug_player_search,
                        case=False,
                        na=False,
                    )
                    | sportsbook_match_diagnostics["matched_projection_player"]
                    .astype(str)
                    .str.contains(debug_player_search, case=False, na=False)
                    | (sportsbook_match_diagnostics["normalized_player_name"] == normalized_search)
                    | (sportsbook_match_diagnostics["player_match_key"] == match_key_search)
                ].copy()
                if not sportsbook_match_diagnostics.empty
                else pd.DataFrame()
            )

            st.subheader("ESPN Row")
            if espn_search_rows.empty:
                if not sportsbook_search_rows.empty:
                    st.warning(
                        "This player exists in the sportsbook projection table, "
                        "but was not found in the ESPN free-agent table."
                    )
                else:
                    st.info("No ESPN free-agent row matched that search.")
            else:
                st.dataframe(
                    clean_dataframe_for_streamlit(espn_search_rows),
                    width="stretch",
                )

            st.subheader("Sportsbook Projection Row")
            if sportsbook_search_rows.empty:
                if not espn_search_rows.empty:
                    st.warning(
                        "This player exists in ESPN free agents, but was not "
                        "found in the sportsbook projection table."
                    )
                else:
                    st.info("No sportsbook projection row matched that search.")
            else:
                st.dataframe(
                    clean_dataframe_for_streamlit(
                        sportsbook_search_rows[
                            [
                                column
                                for column in [
                                    "player",
                                    "normalized_player_name",
                                    "player_match_key",
                                    "projected_fantasy_points",
                                    "projection_source",
                                    "bookmaker_count",
                                    "markets_available",
                                    "has_sportsbook_props",
                                ]
                                if column in sportsbook_search_rows.columns
                            ]
                        ].round(3)
                    ),
                    width="stretch",
                )

            st.subheader("Match Result")
            if diagnostic_search_rows.empty and final_match_rows.empty:
                st.info("No final match result found for that search.")
            else:
                if not diagnostic_search_rows.empty:
                    st.dataframe(
                        clean_dataframe_for_streamlit(diagnostic_search_rows),
                        width="stretch",
                    )
                if not final_match_rows.empty:
                    final_match_rows["filter_reason"] = final_match_rows.apply(
                        lambda row: get_available_hitter_filter_reason(
                            row,
                            include_unavailable_players,
                            include_players_without_games,
                            include_stat_fallback_only_hitters,
                        ),
                        axis=1,
                    )
                    final_columns = [
                        column
                        for column in [
                            "player",
                            "espn_player_id",
                            "normalized_player_name",
                            "player_match_key",
                            "matched_sportsbook_player",
                            "sportsbook_match_method",
                            "sportsbook_match_score",
                            "projected_fantasy_points",
                            "projection_source",
                            "bookmaker_count",
                            "markets_available",
                            "has_sportsbook_props",
                            "has_game_today",
                            "is_available_today",
                            "filter_reason",
                        ]
                        if column in final_match_rows.columns
                    ]
                    st.dataframe(
                        clean_dataframe_for_streamlit(
                            final_match_rows[final_columns].round(3)
                        ),
                        width="stretch",
                    )
        elif sportsbook_match_diagnostics.empty:
            st.info("No match diagnostics are available.")

    if has_roster_data and show_roster_protection_review:
        with st.expander("Roster Protection Review"):
            st.write(
                "The optimizer only suggests dropping players marked droppable "
                "or streamable. Protected, Core, IL, and unavailable players are not dropped."
            )
            st.dataframe(
                clean_dataframe_for_streamlit(
                    build_roster_protection_review(
                        roster_group_table if not roster_group_table.empty else roster_dataframe
                    )
                ),
                width="stretch",
            )

    if has_roster_data:
        with st.expander("Droppable Candidate Review"):
            review = build_roster_protection_review(
                roster_group_table if not roster_group_table.empty else roster_dataframe
            )
            st.dataframe(clean_dataframe_for_streamlit(review), width="stretch")

    if has_roster_data or has_available_data:
        with st.expander("Projection Coverage Review"):
            st.write(
                "This separates sportsbook-line projections from fallback or "
                "missing projections so empty recommendation sections are easier to diagnose."
            )
            st.dataframe(
                clean_dataframe_for_streamlit(
                    build_projection_coverage_review(master_pickup_table).round(3)
                ),
                width="stretch",
            )
    if has_roster_data and has_available_data:
        with st.expander("Replacement Value by Position"):
            phase_start = perf_counter()
            show_replacement_value_by_position(
                roster_projection_table,
                reduced_available_projection_table,
                master_pickup_table,
            )
            log_timing(timing_rows, "Replacement value generation", phase_start)
    else:
        with st.expander("Replacement Value by Position"):
            st.info("Load both roster and available-player data to compare replacement value.")

    with st.expander("Final Performance Timing Log"):
        st.write(
            "This includes lower-page work such as replacement value generation."
        )
        timing_rows.append(
            {
                "phase": "Total Top Pickups render",
                "seconds": round(perf_counter() - page_start_time, 3),
            }
        )
        st.dataframe(
            clean_dataframe_for_streamlit(pd.DataFrame(timing_rows)),
            width="stretch",
        )

except MissingOddsAPIKeyError as error:
    st.warning(str(error))
    st.write("Create a `.env` file and add your The Odds API key to load pickups.")
except OddsAPIError as error:
    st.error(str(error))
    show_odds_api_error_debug(error)
