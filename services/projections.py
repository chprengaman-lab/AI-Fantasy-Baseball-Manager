"""Projection services for fantasy baseball analytics.

This module turns raw data into projection tables. Streamlit pages can import
these functions so projection math stays in one reusable place.
"""

import pandas as pd

from config import LEAGUE_SCORING
from utils.odds import american_odds_to_implied_probability


# This mapping translates The Odds API market names into baseball stat names.
PROP_MARKET_TO_STAT = {
    "batter_hits": "Hits",
    "batter_home_runs": "Home Run",
    "batter_rbis": "RBI",
    "batter_runs_scored": "Run",
    "batter_stolen_bases": "Stolen Base",
    "batter_total_bases": "Total Bases",
}


# These are the standard columns for prop-based hitter projection tables.
HITTER_PROP_PROJECTION_COLUMNS = [
    "player",
    "bookmaker",
    "projected_fantasy_points",
    "estimated_singles",
    "estimated_doubles",
    "estimated_triples",
    "projected_hits",
    "projected_home_runs",
    "projected_rbi",
    "projected_runs",
    "projected_stolen_bases",
    "projected_total_bases",
]


def estimate_hit_type_breakdown(
    projected_hits: float,
    projected_total_bases: float,
    projected_home_runs: float,
) -> dict:
    """Estimate singles, doubles, and triples from broad prop markets.

    The Odds API gives us hits, total bases, and home runs. Your league scores
    singles, doubles, triples, and home runs separately, so this approximation
    turns the broad markets into the hit types needed for fantasy scoring.
    """

    non_hr_hits = max(projected_hits - projected_home_runs, 0)
    extra_bases_remaining = max(
        projected_total_bases - projected_hits - (projected_home_runs * 3),
        0,
    )
    estimated_triples = min(non_hr_hits * 0.05, extra_bases_remaining / 2)
    estimated_doubles = min(
        non_hr_hits - estimated_triples,
        extra_bases_remaining - (estimated_triples * 2),
    )
    estimated_singles = max(
        non_hr_hits - estimated_doubles - estimated_triples,
        0,
    )

    return {
        "estimated_singles": estimated_singles,
        "estimated_doubles": estimated_doubles,
        "estimated_triples": estimated_triples,
    }


def build_hitter_prop_projection_table(prop_rows: list[dict]) -> pd.DataFrame:
    """Build player-level projections from raw hitter prop rows.

    Each prop row gives us one market line and one Over price. We estimate the
    expected stat with:

    expected_stat = line * over_implied_probability

    Hits and total bases are not scored directly. After we create player-level
    projected hits, total bases, and home runs, we estimate singles, doubles,
    and triples for league scoring.
    """

    if not prop_rows:
        return pd.DataFrame(columns=HITTER_PROP_PROJECTION_COLUMNS)

    props_dataframe = pd.DataFrame(prop_rows)

    # Convert American odds like -120 or +150 into decimal probabilities.
    props_dataframe["over_implied_probability"] = props_dataframe["over_odds"].apply(
        american_odds_to_implied_probability
    )

    # Drop rows that cannot produce an expected stat because they are missing a
    # line or an Over probability.
    props_dataframe = props_dataframe.dropna(
        subset=["line", "over_implied_probability"]
    ).copy()

    if props_dataframe.empty:
        return pd.DataFrame(columns=HITTER_PROP_PROJECTION_COLUMNS)

    # Translate API market names into stat names we can display and score.
    props_dataframe["stat"] = props_dataframe["market"].map(PROP_MARKET_TO_STAT)
    props_dataframe = props_dataframe.dropna(subset=["stat"]).copy()

    # Calculate the temporary expected stat from the prop line and Over
    # probability.
    props_dataframe["expected_stat"] = (
        props_dataframe["line"] * props_dataframe["over_implied_probability"]
    )

    # Sum expected stats by player, bookmaker, and stat. A player can have
    # several markets from the same bookmaker.
    stat_totals = (
        props_dataframe.groupby(["player", "bookmaker", "stat"], as_index=False)[
            "expected_stat"
        ]
        .sum()
    )

    # Pivot stat names into columns so each player/bookmaker has one row.
    projection_table = stat_totals.pivot_table(
        index=["player", "bookmaker"],
        columns="stat",
        values="expected_stat",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    # Rename stat columns into app-facing output column names.
    projection_table = projection_table.rename(
        columns={
            "Hits": "projected_hits",
            "Home Run": "projected_home_runs",
            "RBI": "projected_rbi",
            "Run": "projected_runs",
            "Stolen Base": "projected_stolen_bases",
            "Total Bases": "projected_total_bases",
        }
    )

    # Ensure every expected column exists even if a market is missing today.
    for column in HITTER_PROP_PROJECTION_COLUMNS:
        if column not in projection_table.columns:
            projection_table[column] = 0

    # Estimate singles, doubles, and triples for each player/bookmaker row.
    hit_type_breakdown = projection_table.apply(
        lambda row: estimate_hit_type_breakdown(
            row["projected_hits"],
            row["projected_total_bases"],
            row["projected_home_runs"],
        ),
        axis=1,
        result_type="expand",
    )
    projection_table["estimated_singles"] = hit_type_breakdown["estimated_singles"]
    projection_table["estimated_doubles"] = hit_type_breakdown["estimated_doubles"]
    projection_table["estimated_triples"] = hit_type_breakdown["estimated_triples"]

    # Calculate fantasy points from the categories your league actually scores.
    hitting_scoring = LEAGUE_SCORING["hitting"]
    projection_table["projected_fantasy_points"] = (
        projection_table["estimated_singles"] * hitting_scoring["Single"]
        + projection_table["estimated_doubles"] * hitting_scoring["Double"]
        + projection_table["estimated_triples"] * hitting_scoring["Triple"]
        + projection_table["projected_home_runs"] * hitting_scoring["Home Run"]
        + projection_table["projected_rbi"] * hitting_scoring["RBI"]
        + projection_table["projected_runs"] * hitting_scoring["Run"]
        + projection_table["projected_stolen_bases"] * hitting_scoring["Stolen Base"]
    )

    projection_table = projection_table[HITTER_PROP_PROJECTION_COLUMNS]

    return projection_table.sort_values(
        by="projected_fantasy_points",
        ascending=False,
    )
