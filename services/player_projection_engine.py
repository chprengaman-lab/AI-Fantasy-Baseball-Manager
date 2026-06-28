"""Unified player projection engine.

This service is the single source of truth for the dashboard's player projection
table. Pages should call this module instead of rebuilding projections
themselves.
"""

import pandas as pd

from config import LEAGUE_SCORING
from services.odds_api import fetch_todays_mlb_events, fetch_todays_mlb_hitter_props
from services.player_stats import (
    PlayerStatsError,
    add_player_stats_to_dataframe,
    fetch_season_to_date_hitting_stats,
    normalize_player_name,
)
from services.projections import build_hitter_prop_projection_table


PLAYER_PROJECTION_COLUMNS = [
    "player",
    "team",
    "eligible_positions",
    "bookmaker",
    "projected_hits",
    "projected_home_runs",
    "projected_rbi",
    "projected_runs",
    "projected_stolen_bases",
    "projected_total_bases",
    "projected_fantasy_points",
    "projection_source",
    "projection_confidence",
    "batting_average",
    "obp",
    "slg",
    "ops",
]


STAT_CONTEXT_COLUMNS = [
    "games_played",
    "plate_appearances",
    "batting_average",
    "obp",
    "slg",
    "ops",
    "hits",
    "doubles",
    "triples",
    "home_runs",
    "walks",
    "rbi",
    "runs",
    "stolen_bases",
    "strikeouts",
    "strikeout_rate",
]


def _empty_projection_table() -> pd.DataFrame:
    """Return an empty projection table with the standard columns."""

    projection_table = pd.DataFrame(columns=PLAYER_PROJECTION_COLUMNS)
    projection_table.attrs["player_stats_loaded"] = False
    projection_table.attrs["player_stats_error"] = None

    return projection_table


def _pick_best_bookmaker_row_per_player(projection_table: pd.DataFrame) -> pd.DataFrame:
    """Keep exactly one row per player.

    Sportsbooks can post different lines for the same player. For now, we keep
    the bookmaker row with the highest projected fantasy point total.
    """

    if projection_table.empty:
        return projection_table

    return (
        projection_table.sort_values(
            by=["player", "projected_fantasy_points"],
            ascending=[True, False],
        )
        .drop_duplicates(subset=["player"], keep="first")
        .reset_index(drop=True)
    )


def _add_blank_context_columns(projection_table: pd.DataFrame) -> pd.DataFrame:
    """Ensure optional context columns exist even when stats are unavailable."""

    for column in STAT_CONTEXT_COLUMNS:
        if column not in projection_table.columns:
            projection_table[column] = pd.NA

    return projection_table


def _to_number(value, default: float = 0.0) -> float:
    """Convert a dataframe value to a float for projection math."""

    numeric_value = pd.to_numeric(value, errors="coerce")

    if pd.isna(numeric_value):
        return default

    return float(numeric_value)


def _has_usable_stat_context(player_row) -> bool:
    """Return True when a row has enough season data for a fallback estimate."""

    stat_columns = [
        "games_played",
        "plate_appearances",
        "hits",
        "home_runs",
        "rbi",
        "runs",
        "stolen_bases",
    ]

    return any(_to_number(player_row.get(column, pd.NA)) > 0 for column in stat_columns)


def _estimate_stat_based_fantasy_points(player_row) -> dict:
    """Estimate a conservative daily projection from season-to-date stats.

    This fallback is intentionally simple and lower confidence. It uses season
    totals, converts them to a per-game estimate when possible, and applies a
    0.85 multiplier because it is not adjusted for today's matchup, ballpark,
    weather, or lineup spot.
    """

    hitting_scoring = LEAGUE_SCORING["hitting"]
    games_played = _to_number(player_row.get("games_played", pd.NA))
    plate_appearances = _to_number(player_row.get("plate_appearances", pd.NA))

    if games_played > 0:
        playing_time_denominator = games_played
    elif plate_appearances > 0:
        # Roughly estimate games from PA when games played is unavailable.
        playing_time_denominator = max(plate_appearances / 4.2, 1)
    else:
        # Last-resort denominator keeps season totals from becoming a one-day
        # projection when the source does not include games or PA.
        playing_time_denominator = 162

    hits = _to_number(player_row.get("hits", pd.NA))
    doubles = _to_number(player_row.get("doubles", pd.NA))
    triples = _to_number(player_row.get("triples", pd.NA))
    home_runs = _to_number(player_row.get("home_runs", pd.NA))
    walks = _to_number(player_row.get("walks", pd.NA))
    rbi = _to_number(player_row.get("rbi", pd.NA))
    runs = _to_number(player_row.get("runs", pd.NA))
    stolen_bases = _to_number(player_row.get("stolen_bases", pd.NA))
    strikeouts = _to_number(player_row.get("strikeouts", pd.NA))
    singles = max(hits - doubles - triples - home_runs, 0)

    season_points = (
        singles * hitting_scoring["Single"]
        + doubles * hitting_scoring["Double"]
        + triples * hitting_scoring["Triple"]
        + home_runs * hitting_scoring["Home Run"]
        + walks * hitting_scoring["Walk"]
        + rbi * hitting_scoring["RBI"]
        + runs * hitting_scoring["Run"]
        + stolen_bases * hitting_scoring["Stolen Base"]
        + strikeouts * hitting_scoring["Strikeout"]
    )
    conservative_multiplier = 0.85

    return {
        "projected_fantasy_points": (
            season_points / playing_time_denominator
        )
        * conservative_multiplier,
        "projected_hits": (hits / playing_time_denominator) * conservative_multiplier,
        "projected_home_runs": (
            home_runs / playing_time_denominator
        )
        * conservative_multiplier,
        "projected_rbi": (rbi / playing_time_denominator) * conservative_multiplier,
        "projected_runs": (runs / playing_time_denominator) * conservative_multiplier,
        "projected_stolen_bases": (
            stolen_bases / playing_time_denominator
        )
        * conservative_multiplier,
        "projected_total_bases": (
            (singles + (doubles * 2) + (triples * 3) + (home_runs * 4))
            / playing_time_denominator
        )
        * conservative_multiplier,
    }


def _get_availability_position_column(
    availability_dataframe: pd.DataFrame,
) -> str | None:
    """Return the uploaded CSV column to use for position eligibility."""

    if "eligible_positions" in availability_dataframe.columns:
        return "eligible_positions"

    if "position" in availability_dataframe.columns:
        return "position"

    return None


def _build_uploaded_player_table(
    availability_dataframe: pd.DataFrame | None,
) -> pd.DataFrame:
    """Create one row per uploaded player for fallback or missing projections."""

    if availability_dataframe is None or availability_dataframe.empty:
        return pd.DataFrame(columns=["player", "eligible_positions"])

    if "player" not in availability_dataframe.columns:
        return pd.DataFrame(columns=["player", "eligible_positions"])

    position_column = _get_availability_position_column(availability_dataframe)
    uploaded_players = availability_dataframe[["player"]].copy()

    if position_column is None:
        uploaded_players["eligible_positions"] = ""
    else:
        uploaded_players["eligible_positions"] = availability_dataframe[
            position_column
        ]

    uploaded_players["normalized_player_name"] = uploaded_players["player"].apply(
        normalize_player_name
    )

    return uploaded_players.drop_duplicates(
        subset=["normalized_player_name"],
        keep="first",
    )


def _enrich_positions_from_availability(
    projection_table: pd.DataFrame,
    availability_dataframe: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add eligible positions from an uploaded availability CSV when possible."""

    if availability_dataframe is None or availability_dataframe.empty:
        projection_table["eligible_positions"] = ""
        return projection_table

    if "player" not in availability_dataframe.columns:
        projection_table["eligible_positions"] = ""
        return projection_table

    position_column = _get_availability_position_column(availability_dataframe)

    if position_column is None:
        projection_table["eligible_positions"] = ""
        return projection_table

    availability_lookup = availability_dataframe[["player", position_column]].copy()
    availability_lookup["normalized_player_name"] = availability_lookup["player"].apply(
        normalize_player_name
    )
    availability_lookup = availability_lookup.rename(
        columns={position_column: "eligible_positions"}
    )
    availability_lookup = availability_lookup[
        ["normalized_player_name", "eligible_positions"]
    ].drop_duplicates(subset=["normalized_player_name"], keep="first")

    projection_table["normalized_player_name"] = projection_table["player"].apply(
        normalize_player_name
    )
    projection_table = projection_table.merge(
        availability_lookup,
        on="normalized_player_name",
        how="left",
        suffixes=("", "_availability"),
    )
    projection_table["eligible_positions"] = projection_table[
        "eligible_positions"
    ].fillna("")
    projection_table = projection_table.drop(columns=["normalized_player_name"])

    return projection_table


def _add_uploaded_players_without_lines(
    projection_table: pd.DataFrame,
    availability_dataframe: pd.DataFrame | None,
) -> pd.DataFrame:
    """Append uploaded players who do not have sportsbook prop rows today."""

    uploaded_players = _build_uploaded_player_table(availability_dataframe)

    if uploaded_players.empty:
        return projection_table

    if projection_table.empty:
        existing_names = set()
    else:
        existing_names = set(
            projection_table["player"].dropna().apply(normalize_player_name)
        )

    missing_uploaded_players = uploaded_players[
        ~uploaded_players["normalized_player_name"].isin(existing_names)
    ].copy()

    if missing_uploaded_players.empty:
        return projection_table

    for column in PLAYER_PROJECTION_COLUMNS:
        if column not in missing_uploaded_players.columns:
            missing_uploaded_players[column] = pd.NA

    missing_uploaded_players["team"] = ""
    missing_uploaded_players["bookmaker"] = ""
    missing_uploaded_players["projected_hits"] = 0.0
    missing_uploaded_players["projected_home_runs"] = 0.0
    missing_uploaded_players["projected_rbi"] = 0.0
    missing_uploaded_players["projected_runs"] = 0.0
    missing_uploaded_players["projected_stolen_bases"] = 0.0
    missing_uploaded_players["projected_total_bases"] = 0.0
    missing_uploaded_players["projected_fantasy_points"] = 0.0
    missing_uploaded_players["projection_source"] = "Missing"
    missing_uploaded_players["projection_confidence"] = "Unknown"

    return pd.concat(
        [projection_table, missing_uploaded_players[PLAYER_PROJECTION_COLUMNS]],
        ignore_index=True,
    )


def _apply_projection_sources(projection_table: pd.DataFrame) -> pd.DataFrame:
    """Label each row as sportsbook, fallback, or missing."""

    if projection_table.empty:
        return projection_table

    projection_table = projection_table.copy()

    for index, player_row in projection_table.iterrows():
        if player_row.get("projection_source") == "Sportsbook lines":
            continue

        if _has_usable_stat_context(player_row):
            fallback_values = _estimate_stat_based_fantasy_points(player_row)

            for column, value in fallback_values.items():
                projection_table.at[index, column] = value

            projection_table.at[index, "projection_source"] = "Stat-based fallback"
            projection_table.at[index, "projection_confidence"] = "Low"
        else:
            projection_table.at[index, "projected_fantasy_points"] = 0.0
            projection_table.at[index, "projection_source"] = "Missing"
            projection_table.at[index, "projection_confidence"] = "Unknown"

    return projection_table


def build_player_projection_table(
    availability_dataframe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the unified player projection table.

    This function combines:
    - Today's sportsbook hitter props from The Odds API
    - Implied probabilities and fantasy scoring from the prop projection service
    - Season-to-date player stats from pybaseball when available
    - Uploaded position eligibility when an availability CSV is provided

    Returns:
        A dataframe with exactly one row per player.
    """

    todays_events = fetch_todays_mlb_events()
    prop_rows = fetch_todays_mlb_hitter_props(todays_events) if todays_events else []
    prop_projection_table = build_hitter_prop_projection_table(prop_rows)

    projection_table = _pick_best_bookmaker_row_per_player(prop_projection_table)

    # Team is part of the long-term data model, but it is not available from the
    # current prop feed yet.
    projection_table["team"] = ""
    projection_table["projection_source"] = "Sportsbook lines"
    projection_table["projection_confidence"] = "High"
    projection_table = _enrich_positions_from_availability(
        projection_table,
        availability_dataframe,
    )
    projection_table = _add_uploaded_players_without_lines(
        projection_table,
        availability_dataframe,
    )

    stats_loaded = False
    stats_error = None

    try:
        player_stats = fetch_season_to_date_hitting_stats()
        projection_table = add_player_stats_to_dataframe(projection_table, player_stats)
        stats_loaded = True
    except (PlayerStatsError, Exception) as error:
        stats_error = str(error)
        projection_table = _add_blank_context_columns(projection_table)

    projection_table = _apply_projection_sources(projection_table)

    # Keep the engine output stable for every page that consumes it.
    for column in PLAYER_PROJECTION_COLUMNS:
        if column not in projection_table.columns:
            projection_table[column] = pd.NA

    projection_table = projection_table[PLAYER_PROJECTION_COLUMNS].sort_values(
        by="projected_fantasy_points",
        ascending=False,
    )

    projection_table.attrs["player_stats_loaded"] = stats_loaded
    projection_table.attrs["player_stats_error"] = stats_error

    return projection_table.reset_index(drop=True)
