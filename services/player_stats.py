"""Service functions for season-to-date MLB hitter statistics.

This module owns pybaseball access and player-name matching. Streamlit pages and
projection services should call these functions instead of importing pybaseball
directly.
"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.name_matching import (
    build_player_match_key,
    clean_player_name_encoding,
    fuzzy_match_player_name,
    normalize_player_name as normalize_match_name,
)


# pybaseball and matplotlib both create cache/config files during import. We
# point those folders inside this project so the app does not try to write into
# a user's home directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PYBASEBALL_CACHE", str(PROJECT_ROOT / ".pybaseball_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))


PLAYER_STATS_COLUMNS = [
    "player",
    "stats_player_name_before_cleaning",
    "stats_player_name_after_cleaning",
    "games_played",
    "at_bats",
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
    "caught_stealing",
    "strikeouts",
    "strikeout_rate",
    "hit_by_pitch",
    "intentional_walks",
    "sacrifices",
    "ground_into_double_play",
]


class PlayerStatsError(Exception):
    """Raised when player statistics cannot be loaded or normalized."""


def normalize_player_name(player_name: str) -> str:
    """Normalize a player name so names from different sources can be matched.

    Sportsbooks and stat providers sometimes use slightly different punctuation,
    accents, or suffixes. This function makes a best-effort comparable version
    of a name.
    """

    return normalize_match_name(player_name)


def _first_available_column(dataframe: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name that exists in a dataframe."""

    for column in candidates:
        if column in dataframe.columns:
            return column

    return None


def _copy_column(
    source_dataframe: pd.DataFrame,
    output_dataframe: pd.DataFrame,
    output_column: str,
    candidates: list[str],
) -> None:
    """Copy the first matching source column into a standard output column."""

    source_column = _first_available_column(source_dataframe, candidates)

    if source_column is None:
        output_dataframe[output_column] = pd.NA
    else:
        output_dataframe[output_column] = source_dataframe[source_column]


def _clean_strikeout_rate(value):
    """Convert strikeout rate values into a decimal when possible."""

    if pd.isna(value):
        return pd.NA

    if isinstance(value, str):
        cleaned_value = value.strip().replace("%", "")

        if not cleaned_value:
            return pd.NA

        numeric_value = float(cleaned_value)
        return numeric_value / 100

    # Some sources already return K% as 0.22. Others return 22.0.
    if value > 1:
        return value / 100

    return value


def fetch_season_to_date_hitting_stats(season: int | None = None) -> pd.DataFrame:
    """Retrieve season-to-date MLB hitter statistics from pybaseball.

    Args:
        season: MLB season year. Defaults to the current calendar year.

    Returns:
        A normalized dataframe with stable column names for the dashboard.
    """

    selected_season = season or datetime.now().year

    raw_stats = None

    try:
        # Import pybaseball inside the function so our cache environment
        # variables are set before pybaseball initializes.
        from pybaseball import batting_stats

        raw_stats = batting_stats(selected_season, qual=0)
    except Exception:
        raw_stats = None

    if raw_stats is None:
        try:
            # FanGraphs can sometimes reject automated requests. Baseball
            # Reference is a useful pybaseball fallback for core hitter stats.
            from pybaseball import batting_stats_bref

            raw_stats = batting_stats_bref(selected_season)
        except Exception:
            raw_stats = None

    if raw_stats is None:
        raise PlayerStatsError(
            f"Unable to load pybaseball hitting stats for {selected_season}."
        ) from None

    if not isinstance(raw_stats, pd.DataFrame):
        raise PlayerStatsError("pybaseball returned an unexpected stats format.")

    return normalize_hitting_stats(raw_stats)


def normalize_hitting_stats(raw_stats: pd.DataFrame) -> pd.DataFrame:
    """Convert pybaseball's dataframe into app-friendly hitting stat columns."""

    stats_dataframe = pd.DataFrame()

    _copy_column(raw_stats, stats_dataframe, "player", ["Name", "name", "Player"])
    stats_dataframe["stats_player_name_before_cleaning"] = stats_dataframe["player"]
    stats_dataframe["player"] = stats_dataframe["player"].apply(
        clean_player_name_encoding
    )
    stats_dataframe["stats_player_name_after_cleaning"] = stats_dataframe["player"]
    _copy_column(raw_stats, stats_dataframe, "games_played", ["G", "Games"])
    _copy_column(raw_stats, stats_dataframe, "at_bats", ["AB", "At Bats"])
    _copy_column(raw_stats, stats_dataframe, "plate_appearances", ["PA"])
    _copy_column(raw_stats, stats_dataframe, "batting_average", ["AVG", "BA"])
    _copy_column(raw_stats, stats_dataframe, "obp", ["OBP"])
    _copy_column(raw_stats, stats_dataframe, "slg", ["SLG"])
    _copy_column(raw_stats, stats_dataframe, "ops", ["OPS"])
    _copy_column(raw_stats, stats_dataframe, "hits", ["H", "Hits"])
    _copy_column(raw_stats, stats_dataframe, "doubles", ["2B", "Doubles"])
    _copy_column(raw_stats, stats_dataframe, "triples", ["3B", "Triples"])
    _copy_column(raw_stats, stats_dataframe, "home_runs", ["HR", "Home Runs"])
    _copy_column(raw_stats, stats_dataframe, "walks", ["BB", "Walks"])
    _copy_column(raw_stats, stats_dataframe, "rbi", ["RBI"])
    _copy_column(raw_stats, stats_dataframe, "runs", ["R", "Runs"])
    _copy_column(raw_stats, stats_dataframe, "stolen_bases", ["SB", "Stolen Bases"])
    _copy_column(raw_stats, stats_dataframe, "caught_stealing", ["CS"])
    _copy_column(raw_stats, stats_dataframe, "strikeouts", ["SO", "K", "Strikeouts"])
    _copy_column(raw_stats, stats_dataframe, "strikeout_rate", ["K%", "SO%", "KRate"])
    _copy_column(raw_stats, stats_dataframe, "hit_by_pitch", ["HBP"])
    _copy_column(raw_stats, stats_dataframe, "intentional_walks", ["IBB"])
    _copy_column(raw_stats, stats_dataframe, "sacrifices", ["SF", "SH"])
    _copy_column(raw_stats, stats_dataframe, "ground_into_double_play", ["GDP", "GIDP"])

    if "player" not in stats_dataframe.columns:
        raise PlayerStatsError("pybaseball stats did not include player names.")

    stats_dataframe["normalized_player_name"] = stats_dataframe["player"].apply(
        normalize_player_name
    )

    # If the source does not include K%, calculate it from strikeouts and plate
    # appearances when both columns are available.
    if stats_dataframe["strikeout_rate"].isna().all():
        strikeout_column = _first_available_column(raw_stats, ["SO", "K", "Strikeouts"])
        plate_appearance_column = _first_available_column(raw_stats, ["PA"])

        if strikeout_column and plate_appearance_column:
            stats_dataframe["strikeout_rate"] = (
                raw_stats[strikeout_column] / raw_stats[plate_appearance_column]
            )

    stats_dataframe["strikeout_rate"] = stats_dataframe["strikeout_rate"].apply(
        _clean_strikeout_rate
    )

    return stats_dataframe.sort_values("player").reset_index(drop=True)


def match_sportsbook_players_to_stats(
    sportsbook_player_names: list[str],
    stats_dataframe: pd.DataFrame,
) -> dict:
    """Match sportsbook player names to pybaseball stat rows when possible.

    Returns:
        A dictionary keyed by the original sportsbook player name. Each value is
        either a matched stats row as a dictionary or None when no match exists.
    """

    stats_by_normalized_name = {
        row["normalized_player_name"]: row.to_dict()
        for _, row in stats_dataframe.iterrows()
    }
    matches = {}

    for player_name in sportsbook_player_names:
        normalized_name = normalize_player_name(player_name)
        matches[player_name] = stats_by_normalized_name.get(normalized_name)

    return matches


def add_player_stats_to_dataframe(
    player_dataframe: pd.DataFrame,
    stats_dataframe: pd.DataFrame,
    player_column: str = "player",
) -> pd.DataFrame:
    """Add pybaseball stats to a dataframe that already has player names.

    Unmatched players stay in the output with blank stat values. This lets the
    dashboard continue gracefully when names do not match perfectly.
    """

    if player_dataframe.empty:
        return player_dataframe.copy()

    output_dataframe = player_dataframe.copy()
    output_dataframe["normalized_player_name"] = output_dataframe[player_column].apply(
        normalize_player_name
    )
    output_dataframe["player_match_key"] = output_dataframe[player_column].apply(
        build_player_match_key
    )

    normalized_stats = stats_dataframe.copy()
    if "normalized_player_name" not in normalized_stats.columns:
        normalized_stats["normalized_player_name"] = normalized_stats["player"].apply(
            normalize_player_name
        )
    normalized_stats["player_match_key"] = normalized_stats["player"].apply(
        build_player_match_key
    )

    stats_by_match_key = (
        normalized_stats.drop_duplicates(subset=["player_match_key"], keep="first")
        .set_index("player_match_key")
    )
    stats_by_normalized_name = (
        normalized_stats.drop_duplicates(subset=["normalized_player_name"], keep="first")
        .set_index("normalized_player_name")
    )
    stats_names = normalized_stats["player"].dropna().astype(str).unique().tolist()
    stats_columns = [column for column in PLAYER_STATS_COLUMNS if column != "player"]
    enriched_rows = []

    for _, player_row in output_dataframe.iterrows():
        enriched_row = player_row.to_dict()
        player_match_key = enriched_row.get("player_match_key", "")
        normalized_name = enriched_row.get("normalized_player_name", "")
        matched_stats_row = None
        match_method = ""
        match_score = 0

        if player_match_key and player_match_key in stats_by_match_key.index:
            matched_stats_row = stats_by_match_key.loc[player_match_key]
            match_method = "exact match key"
            match_score = 100
        elif normalized_name and normalized_name in stats_by_normalized_name.index:
            matched_stats_row = stats_by_normalized_name.loc[normalized_name]
            match_method = "exact normalized name"
            match_score = 100
        else:
            fuzzy_name, fuzzy_score = fuzzy_match_player_name(
                enriched_row.get(player_column, ""),
                stats_names,
                min_score=90,
            )
            if fuzzy_name:
                matched_stats_row = normalized_stats[
                    normalized_stats["player"] == fuzzy_name
                ].iloc[0]
                match_method = "fuzzy"
                match_score = fuzzy_score

        if matched_stats_row is not None:
            enriched_row["matched_stats_player"] = matched_stats_row.get("player", "")
            enriched_row["normalized_stats_name"] = matched_stats_row.get(
                "normalized_player_name",
                "",
            )
            enriched_row["stats_match_method"] = match_method
            enriched_row["stats_match_score"] = match_score
            for column in stats_columns:
                enriched_row[column] = matched_stats_row.get(column, pd.NA)
        else:
            enriched_row["matched_stats_player"] = ""
            enriched_row["normalized_stats_name"] = ""
            enriched_row["stats_match_method"] = "unmatched"
            enriched_row["stats_match_score"] = 0
            for column in stats_columns:
                if column not in enriched_row:
                    enriched_row[column] = pd.NA

        enriched_rows.append(enriched_row)

    enriched_dataframe = pd.DataFrame(enriched_rows)
    enriched_dataframe["stats_match_score"] = pd.to_numeric(
        enriched_dataframe["stats_match_score"],
        errors="coerce",
    ).fillna(0)

    return enriched_dataframe
