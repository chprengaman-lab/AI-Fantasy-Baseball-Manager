"""Projection services for fantasy baseball analytics.

This module turns raw sportsbook prop rows into player-level projection rows.
The key idea is that most hitter props are thresholds, not averages. For
example, Over 0.5 home runs means "P(HR >= 1)", so expected home runs should be
calculated from survival probabilities instead of line x probability.
"""

import pandas as pd

from config import LEAGUE_SCORING
from utils.odds import american_odds_to_implied_probability


OVER_ONLY_DISCOUNT = 0.92
MIN_OVER_ONLY_DISCOUNT = 0.85
MAX_OVER_ONLY_DISCOUNT = 1.00


PROP_MARKET_TO_STAT = {
    "batter_hits": "Hits",
    "batter_home_runs": "Home Run",
    "batter_rbis": "RBI",
    "batter_runs_scored": "Run",
    "batter_stolen_bases": "Stolen Base",
    "batter_total_bases": "Total Bases",
}


HITTER_PROP_PROJECTION_COLUMNS = [
    "player",
    "bookmaker",
    "bookmaker_count",
    "markets_available",
    "vig_removed",
    "over_only_discount_applied",
    "probability_methods",
    "projected_fantasy_points",
    "projected_singles",
    "projected_doubles",
    "projected_triples",
    "estimated_singles",
    "estimated_doubles",
    "estimated_triples",
    "projected_extra_base_hits",
    "estimated_extra_base_hits",
    "projected_hits",
    "projected_home_runs",
    "projected_rbi",
    "projected_runs",
    "projected_stolen_bases",
    "projected_total_bases",
    "projected_walks",
    "projected_strikeouts",
    "projected_hit_by_pitch",
    "projected_intentional_walks",
    "projected_sacrifices",
    "projected_caught_stealing",
    "projected_ground_into_double_play",
    "projected_game_winning_rbi",
    "projected_grand_slams",
    "projected_hit_for_cycle",
    "projection_adjustment_note",
]


MARKET_DIAGNOSTIC_COLUMNS = [
    "player",
    "bookmaker",
    "market",
    "line",
    "over_odds",
    "under_odds",
    "raw_over_implied_probability",
    "raw_under_implied_probability",
    "no_vig_over_probability",
    "final_projection_probability",
    "vig_removed",
    "over_only_discount_applied",
    "probability_method",
    "expected_stat_component",
]


def _to_projection_number(value, default: float = 0.0) -> float:
    """Convert a value to a float for projection math."""

    numeric_value = pd.to_numeric(value, errors="coerce")

    if pd.isna(numeric_value):
        return default

    return float(numeric_value)


def _is_missing(value) -> bool:
    """Return True for empty dataframe values."""

    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _clamp_probability(value) -> float:
    """Keep a probability inside the valid 0 to 1 range."""

    return max(min(_to_projection_number(value), 1.0), 0.0)


def _threshold_from_line(line) -> int:
    """Convert a 0.5-style prop line into a count threshold.

    Over 0.5 means at least 1. Over 1.5 means at least 2, and so on.
    """

    return int(_to_projection_number(line) + 0.5)


def calculate_no_vig_over_probability(over_odds, under_odds) -> dict:
    """Calculate raw and no-vig Over probabilities from American odds."""

    raw_over_probability = american_odds_to_implied_probability(over_odds)
    raw_under_probability = american_odds_to_implied_probability(under_odds)

    if (
        raw_over_probability is not None
        and raw_under_probability is not None
        and (raw_over_probability + raw_under_probability) > 0
    ):
        no_vig_probability = raw_over_probability / (
            raw_over_probability + raw_under_probability
        )
        vig_removed = True
    else:
        no_vig_probability = pd.NA
        vig_removed = False

    return {
        "raw_over_implied_probability": raw_over_probability,
        "raw_under_implied_probability": raw_under_probability,
        "no_vig_over_probability": no_vig_probability,
        "vig_removed": vig_removed,
    }


def calculate_projection_probability(
    over_odds,
    under_odds=None,
    market=None,
    line=None,
) -> dict:
    """Choose the final projection probability for one sportsbook prop row.

    If both Over and Under prices exist, we remove vig directly. If only the
    Over price exists, we apply a smaller configurable discount because one-side
    prices still include sportsbook margin but do not let us calculate no-vig
    exactly.
    """

    no_vig_values = calculate_no_vig_over_probability(over_odds, under_odds)
    raw_over_probability = no_vig_values["raw_over_implied_probability"]
    no_vig_probability = no_vig_values["no_vig_over_probability"]

    if no_vig_values["vig_removed"]:
        final_probability = no_vig_probability
        over_only_discount_applied = False
        probability_method = "no_vig_over_under"
    elif raw_over_probability is not None:
        discount = max(
            min(OVER_ONLY_DISCOUNT, MAX_OVER_ONLY_DISCOUNT),
            MIN_OVER_ONLY_DISCOUNT,
        )
        final_probability = raw_over_probability * discount
        over_only_discount_applied = True
        probability_method = "over_only_discounted"
    else:
        final_probability = pd.NA
        over_only_discount_applied = False
        probability_method = "missing"

    return {
        **no_vig_values,
        "final_projection_probability": (
            pd.NA if _is_missing(final_probability) else _clamp_probability(final_probability)
        ),
        "over_only_discount_applied": over_only_discount_applied,
        "probability_method": probability_method,
    }


def _prepare_prop_dataframe(prop_rows: list[dict]) -> pd.DataFrame:
    """Add probability diagnostics to flattened prop rows."""

    if not prop_rows:
        return pd.DataFrame()

    props_dataframe = pd.DataFrame(prop_rows)

    if props_dataframe.empty:
        return props_dataframe

    probability_columns = props_dataframe.apply(
        lambda row: calculate_projection_probability(
            over_odds=row.get("over_odds"),
            under_odds=row.get("under_odds"),
            market=row.get("market"),
            line=row.get("line"),
        ),
        axis=1,
        result_type="expand",
    )

    props_dataframe = pd.concat([props_dataframe, probability_columns], axis=1)

    return props_dataframe


def _build_survival_probabilities(
    market_rows: pd.DataFrame,
    max_threshold: int | None = None,
) -> dict[int, float]:
    """Build clamped survival probabilities for one player and market.

    Expected counts from threshold props equal the sum of survival
    probabilities. For example, E(HR) = P(HR >= 1) + P(HR >= 2) + P(HR >= 3).
    """

    if market_rows.empty:
        return {}

    threshold_probabilities = {}

    for _, row in market_rows.iterrows():
        probability = row.get("final_projection_probability")

        if _is_missing(probability):
            continue

        threshold = _threshold_from_line(row.get("line"))

        if max_threshold is not None and threshold > max_threshold:
            continue

        threshold_probabilities.setdefault(threshold, []).append(
            _clamp_probability(probability)
        )

    averaged_probabilities = {
        threshold: sum(values) / len(values)
        for threshold, values in threshold_probabilities.items()
        if values
    }
    clamped_probabilities = {}
    previous_probability = 1.0

    for threshold in sorted(averaged_probabilities):
        current_probability = min(averaged_probabilities[threshold], previous_probability)
        clamped_probabilities[threshold] = current_probability
        previous_probability = current_probability

    return clamped_probabilities


def _sum_survival_probabilities(probabilities: dict[int, float]) -> float:
    """Convert threshold survival probabilities into an expected stat count."""

    return sum(probabilities.values())


def _estimate_hits_from_thresholds(market_rows: pd.DataFrame) -> float | pd.NA:
    """Project hits from hit survival probabilities."""

    probabilities = _build_survival_probabilities(market_rows, max_threshold=3)

    if not probabilities:
        return pd.NA

    if 1 not in probabilities and 2 in probabilities:
        probabilities[1] = min(probabilities[2] + 0.35, 0.95)

    probabilities = dict(sorted(probabilities.items()))
    clamped = {}
    previous_probability = 1.0

    for threshold, probability in probabilities.items():
        clamped[threshold] = min(probability, previous_probability)
        previous_probability = clamped[threshold]

    return _sum_survival_probabilities(clamped)


def _estimate_total_bases_from_thresholds(
    market_rows: pd.DataFrame,
    projected_hits,
    projected_home_runs,
) -> tuple[float | pd.NA, str]:
    """Project total bases from survival probabilities and reconcile with HR."""

    probabilities = _build_survival_probabilities(market_rows)

    if not probabilities:
        return pd.NA, ""

    if 1 not in probabilities and not _is_missing(projected_hits):
        probabilities[1] = min(_to_projection_number(projected_hits), 1.0)

    projected_total_bases = _sum_survival_probabilities(probabilities)
    minimum_total_bases = _to_projection_number(projected_hits) + (
        3 * _to_projection_number(projected_home_runs)
    )

    if projected_total_bases + 0.01 < minimum_total_bases:
        return (
            minimum_total_bases,
            "Total bases reconciled upward so hits and home runs are internally consistent.",
        )

    return projected_total_bases, ""


def _estimate_counting_stat_from_thresholds(
    market_rows: pd.DataFrame,
    market: str,
) -> float | pd.NA:
    """Project runs, RBI, and stolen bases from threshold probabilities."""

    probabilities = _build_survival_probabilities(market_rows, max_threshold=3)

    if not probabilities:
        return pd.NA

    if len(probabilities) == 1 and 1 in probabilities:
        multiplier_by_market = {
            "batter_runs_scored": 1.10,
            "batter_rbis": 1.10,
            "batter_stolen_bases": 1.05,
        }
        return probabilities[1] * multiplier_by_market.get(market, 1.0)

    return _sum_survival_probabilities(probabilities)


def _estimate_expected_stat_component(row) -> float | pd.NA:
    """Diagnostic expected-stat component for one prop row before aggregation."""

    probability = row.get("final_projection_probability")

    if _is_missing(probability):
        return pd.NA

    market = row.get("market")
    line = _to_projection_number(row.get("line"))

    if market in {"batter_hits", "batter_home_runs", "batter_total_bases"}:
        return _clamp_probability(probability)

    if line == 0.5 and market == "batter_runs_scored":
        return _clamp_probability(probability) * 1.10

    if line == 0.5 and market == "batter_rbis":
        return _clamp_probability(probability) * 1.10

    if line == 0.5 and market == "batter_stolen_bases":
        return _clamp_probability(probability) * 1.05

    return _clamp_probability(probability)


def estimate_hit_type_breakdown(
    projected_hits: float,
    projected_home_runs: float,
    projected_total_bases: float,
    season_stats=None,
) -> dict:
    """Estimate singles, doubles, and triples with total-base reconciliation.

    Season hit-type distribution is the preferred split. If it is unavailable,
    a neutral non-HR hit split is used. Then total bases are reconciled by
    shifting some singles into doubles/triples while keeping total hits stable.
    """

    projected_hits = _to_projection_number(projected_hits)
    projected_home_runs = min(_to_projection_number(projected_home_runs), projected_hits)
    projected_total_bases = _to_projection_number(projected_total_bases)
    non_hr_hits = max(projected_hits - projected_home_runs, 0.0)

    season_row = season_stats if season_stats is not None else {}
    season_hits = _to_projection_number(
        season_row.get("hits", 0) if hasattr(season_row, "get") else 0
    )
    season_doubles = _to_projection_number(
        season_row.get("doubles", 0) if hasattr(season_row, "get") else 0
    )
    season_triples = _to_projection_number(
        season_row.get("triples", 0) if hasattr(season_row, "get") else 0
    )
    season_home_runs = _to_projection_number(
        season_row.get("home_runs", 0) if hasattr(season_row, "get") else 0
    )
    season_singles = max(
        season_hits - season_doubles - season_triples - season_home_runs,
        0.0,
    )
    season_non_hr_hits = season_singles + season_doubles + season_triples

    if season_non_hr_hits > 0:
        singles_pct_non_hr = season_singles / season_non_hr_hits
        doubles_pct_non_hr = season_doubles / season_non_hr_hits
        triples_pct_non_hr = season_triples / season_non_hr_hits
    else:
        singles_pct_non_hr = 0.70
        doubles_pct_non_hr = 0.27
        triples_pct_non_hr = 0.03

    estimated_singles = non_hr_hits * singles_pct_non_hr
    estimated_doubles = non_hr_hits * doubles_pct_non_hr
    estimated_triples = non_hr_hits * triples_pct_non_hr

    calculated_total_bases = (
        estimated_singles
        + (2 * estimated_doubles)
        + (3 * estimated_triples)
        + (4 * projected_home_runs)
    )
    missing_total_bases = max(projected_total_bases - calculated_total_bases, 0.0)

    if missing_total_bases > 0.01 and non_hr_hits > 0:
        # Upgrade singles into doubles first. One upgraded single adds one total
        # base without changing hit count.
        singles_to_doubles = min(estimated_singles, missing_total_bases)
        estimated_singles -= singles_to_doubles
        estimated_doubles += singles_to_doubles
        missing_total_bases -= singles_to_doubles

        # If more total bases are needed, upgrade doubles into triples. That
        # also adds one total base while keeping total hits stable.
        doubles_to_triples = min(estimated_doubles, missing_total_bases)
        estimated_doubles -= doubles_to_triples
        estimated_triples += doubles_to_triples

    return {
        "projected_singles": estimated_singles,
        "projected_doubles": estimated_doubles,
        "projected_triples": estimated_triples,
        "estimated_singles": estimated_singles,
        "estimated_doubles": estimated_doubles,
        "estimated_triples": estimated_triples,
        "projected_extra_base_hits": (
            estimated_doubles + estimated_triples + projected_home_runs
        ),
        "estimated_extra_base_hits": (
            estimated_doubles + estimated_triples + projected_home_runs
        ),
    }


def calculate_hitter_fantasy_points(row) -> float:
    """Calculate fantasy points from every configured hitting category."""

    scoring = LEAGUE_SCORING["hitting"]

    return (
        _to_projection_number(row.get("projected_singles", row.get("estimated_singles", 0)))
        * scoring["Single"]
        + _to_projection_number(row.get("projected_doubles", row.get("estimated_doubles", 0)))
        * scoring["Double"]
        + _to_projection_number(row.get("projected_triples", row.get("estimated_triples", 0)))
        * scoring["Triple"]
        + _to_projection_number(row.get("projected_home_runs", 0))
        * scoring["Home Run"]
        + _to_projection_number(row.get("projected_walks", 0))
        * scoring["Walk"]
        + _to_projection_number(row.get("projected_runs", 0))
        * scoring["Run"]
        + _to_projection_number(row.get("projected_rbi", 0))
        * scoring["RBI"]
        + _to_projection_number(row.get("projected_stolen_bases", 0))
        * scoring["Stolen Base"]
        + _to_projection_number(row.get("projected_strikeouts", 0))
        * scoring["Strikeout"]
        + _to_projection_number(row.get("projected_extra_base_hits", 0))
        * scoring["Extra Base Hit"]
        + _to_projection_number(row.get("projected_game_winning_rbi", 0))
        * scoring["Game Winning RBI"]
        + _to_projection_number(row.get("projected_intentional_walks", 0))
        * scoring["Intentional Walk"]
        + _to_projection_number(row.get("projected_hit_by_pitch", 0))
        * scoring["Hit By Pitch"]
        + _to_projection_number(row.get("projected_sacrifices", 0))
        * scoring["Sacrifice"]
        + _to_projection_number(row.get("projected_caught_stealing", 0))
        * scoring["Caught Stealing"]
        + _to_projection_number(row.get("projected_ground_into_double_play", 0))
        * scoring["Ground Into Double Play"]
        + _to_projection_number(row.get("projected_hit_for_cycle", 0))
        * scoring["Hit For The Cycle"]
        + _to_projection_number(row.get("projected_grand_slams", 0))
        * scoring["Grand Slam Home Run"]
    )


def _build_player_projection_row(player_name: str, player_rows: pd.DataFrame) -> dict:
    """Build one aggregated sportsbook projection row for one player."""

    row = {
        "player": player_name,
        "bookmaker": "Aggregated",
        "bookmaker_count": player_rows["bookmaker"].nunique(),
        "markets_available": ", ".join(sorted(set(player_rows["market"].dropna()))),
        "vig_removed": bool(player_rows["vig_removed"].all()),
        "over_only_discount_applied": bool(
            player_rows["over_only_discount_applied"].any()
        ),
        "probability_methods": ", ".join(
            sorted(set(player_rows["probability_method"].dropna()))
        ),
        "projection_adjustment_note": "",
    }

    rows_by_market = {
        market: rows.copy()
        for market, rows in player_rows.groupby("market")
    }
    row["projected_home_runs"] = _sum_survival_probabilities(
        _build_survival_probabilities(
            rows_by_market.get("batter_home_runs", pd.DataFrame()),
            max_threshold=3,
        )
    )
    row["projected_hits"] = _estimate_hits_from_thresholds(
        rows_by_market.get("batter_hits", pd.DataFrame())
    )
    row["projected_runs"] = _estimate_counting_stat_from_thresholds(
        rows_by_market.get("batter_runs_scored", pd.DataFrame()),
        "batter_runs_scored",
    )
    row["projected_rbi"] = _estimate_counting_stat_from_thresholds(
        rows_by_market.get("batter_rbis", pd.DataFrame()),
        "batter_rbis",
    )
    row["projected_stolen_bases"] = _estimate_counting_stat_from_thresholds(
        rows_by_market.get("batter_stolen_bases", pd.DataFrame()),
        "batter_stolen_bases",
    )
    projected_total_bases, total_base_note = _estimate_total_bases_from_thresholds(
        rows_by_market.get("batter_total_bases", pd.DataFrame()),
        row["projected_hits"],
        row["projected_home_runs"],
    )
    row["projected_total_bases"] = projected_total_bases
    row["projection_adjustment_note"] = total_base_note

    for column in HITTER_PROP_PROJECTION_COLUMNS:
        if column.startswith("projected_") and column not in row:
            row[column] = pd.NA

    hit_breakdown = estimate_hit_type_breakdown(
        projected_hits=row["projected_hits"],
        projected_home_runs=row["projected_home_runs"],
        projected_total_bases=row["projected_total_bases"],
    )
    row.update(hit_breakdown)

    row["projected_game_winning_rbi"] = _to_projection_number(row["projected_rbi"]) * 0.12
    row["projected_grand_slams"] = _to_projection_number(row["projected_home_runs"]) * 0.0225
    row["projected_hit_for_cycle"] = 0.0
    row["projected_fantasy_points"] = calculate_hitter_fantasy_points(row)

    return row


def build_hitter_prop_projection_table(prop_rows: list[dict]) -> pd.DataFrame:
    """Build player-level projections from raw hitter prop rows."""

    if not prop_rows:
        return pd.DataFrame(columns=HITTER_PROP_PROJECTION_COLUMNS)

    props_dataframe = _prepare_prop_dataframe(prop_rows)

    if props_dataframe.empty:
        return pd.DataFrame(columns=HITTER_PROP_PROJECTION_COLUMNS)

    props_dataframe = props_dataframe.dropna(
        subset=["player", "market", "line", "final_projection_probability"]
    ).copy()
    props_dataframe = props_dataframe[
        props_dataframe["market"].isin(PROP_MARKET_TO_STAT)
    ].copy()

    if props_dataframe.empty:
        return pd.DataFrame(columns=HITTER_PROP_PROJECTION_COLUMNS)

    projection_rows = [
        _build_player_projection_row(player_name, player_rows)
        for player_name, player_rows in props_dataframe.groupby("player")
    ]
    projection_table = pd.DataFrame(projection_rows)

    for column in HITTER_PROP_PROJECTION_COLUMNS:
        if column not in projection_table.columns:
            projection_table[column] = pd.NA

    projection_table = projection_table[HITTER_PROP_PROJECTION_COLUMNS]

    return projection_table.sort_values(
        by="projected_fantasy_points",
        ascending=False,
    )


def build_hitter_prop_market_diagnostics(prop_rows: list[dict]) -> pd.DataFrame:
    """Build market-level diagnostic rows for sportsbook probability math."""

    if not prop_rows:
        return pd.DataFrame(columns=MARKET_DIAGNOSTIC_COLUMNS)

    diagnostics = _prepare_prop_dataframe(prop_rows)

    if diagnostics.empty:
        return pd.DataFrame(columns=MARKET_DIAGNOSTIC_COLUMNS)

    diagnostics["expected_stat_component"] = diagnostics.apply(
        _estimate_expected_stat_component,
        axis=1,
    )

    for column in MARKET_DIAGNOSTIC_COLUMNS:
        if column not in diagnostics.columns:
            diagnostics[column] = pd.NA

    return diagnostics[MARKET_DIAGNOSTIC_COLUMNS].sort_values(
        by=["player", "market", "line", "bookmaker"],
        ascending=True,
    )
