"""Unified player projection engine.

This service is the single source of truth for the dashboard's player projection
table. Pages should call this module instead of rebuilding projections
themselves.
"""

import pandas as pd

from config import LEAGUE_SCORING
from services.odds_api import (
    fetch_todays_mlb_hitter_props,
    get_mlb_teams_playing_today_from_cache,
    get_odds_data_source_status,
)
from services.player_stats import (
    PlayerStatsError,
    add_player_stats_to_dataframe,
    fetch_season_to_date_hitting_stats,
)
from services.projections import (
    build_hitter_prop_market_diagnostics,
    build_hitter_prop_projection_table,
    calculate_hitter_fantasy_points,
    estimate_hit_type_breakdown,
)
from utils.name_matching import build_player_match_key, normalize_player_name
from utils.mlb_teams import normalize_mlb_team
from utils.player_availability import is_player_unavailable


AVAILABILITY_METADATA_COLUMNS = [
    "injury_status",
    "roster_spot",
    "player_type",
    "status",
    "pro_team",
    "player_notes",
    "lineup_status",
    "fantasy_status",
    "espn_player_id",
    "player_match_key",
]


PLAYER_PROJECTION_COLUMNS = [
    "player",
    "normalized_player_name",
    "player_match_key",
    "team",
    "eligible_positions",
    "bookmaker",
    "bookmaker_count",
    "markets_available",
    "missing_markets",
    "fallback_markets_used",
    "fallback_projection_note",
    "secondary_fallback_stats_used",
    "secondary_fallback_note",
    "matched_stats_player",
    "stats_player_name_before_cleaning",
    "stats_player_name_after_cleaning",
    "normalized_stats_name",
    "stats_match_method",
    "stats_match_score",
    "projected_hits",
    "projected_home_runs",
    "projected_rbi",
    "projected_runs",
    "projected_stolen_bases",
    "projected_total_bases",
    "projected_singles",
    "projected_doubles",
    "projected_triples",
    "estimated_singles",
    "estimated_doubles",
    "estimated_triples",
    "projected_extra_base_hits",
    "estimated_extra_base_hits",
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
    "projected_fantasy_points",
    "projection_source",
    "projection_confidence",
    "has_sportsbook_props",
    "is_available_today",
    "availability_note",
    "has_game_today",
    "game_today_note",
    "injury_status",
    "status",
    "pro_team",
    "player_notes",
    "lineup_status",
    "fantasy_status",
    "espn_player_id",
    "vig_removed",
    "over_only_discount_applied",
    "probability_methods",
    "projection_adjustment_note",
    "games_played",
    "at_bats",
    "plate_appearances",
    "hits",
    "walks",
    "strikeouts",
    "hit_by_pitch",
    "intentional_walks",
    "sacrifices",
    "stolen_bases",
    "caught_stealing",
    "ground_into_double_play",
    "batting_average",
    "obp",
    "slg",
    "ops",
]
STAT_CONTEXT_COLUMNS = [
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
    "strikeouts",
    "strikeout_rate",
    "hit_by_pitch",
    "intentional_walks",
    "sacrifices",
    "caught_stealing",
    "ground_into_double_play",
]
TEXT_PROJECTION_COLUMNS = [
    "player",
    "normalized_player_name",
    "player_match_key",
    "team",
    "eligible_positions",
    "bookmaker",
    "markets_available",
    "missing_markets",
    "fallback_markets_used",
    "fallback_projection_note",
    "secondary_fallback_stats_used",
    "secondary_fallback_note",
    "matched_stats_player",
    "stats_player_name_before_cleaning",
    "stats_player_name_after_cleaning",
    "normalized_stats_name",
    "stats_match_method",
    "projection_source",
    "projection_confidence",
    "availability_note",
    "game_today_note",
    "injury_status",
    "status",
    "pro_team",
    "player_notes",
    "lineup_status",
    "fantasy_status",
    "espn_player_id",
    "probability_methods",
    "projection_adjustment_note",
]
NUMERIC_PROJECTION_COLUMNS = [
    "bookmaker_count",
    "stats_match_score",
    "projected_hits",
    "projected_home_runs",
    "projected_rbi",
    "projected_runs",
    "projected_stolen_bases",
    "projected_total_bases",
    "projected_singles",
    "projected_doubles",
    "projected_triples",
    "estimated_singles",
    "estimated_doubles",
    "estimated_triples",
    "projected_extra_base_hits",
    "estimated_extra_base_hits",
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
    "projected_fantasy_points",
    "games_played",
    "at_bats",
    "plate_appearances",
    "hits",
    "walks",
    "strikeouts",
    "hit_by_pitch",
    "intentional_walks",
    "sacrifices",
    "stolen_bases",
    "caught_stealing",
    "ground_into_double_play",
    "batting_average",
    "obp",
    "slg",
    "ops",
]


def ensure_projection_column_types(projection_table: pd.DataFrame) -> pd.DataFrame:
    """Keep projection tracking text and numeric columns in stable dtypes."""

    projection_table = projection_table.copy()

    for column in TEXT_PROJECTION_COLUMNS:
        if column not in projection_table.columns:
            projection_table[column] = ""
        projection_table[column] = projection_table[column].astype("object")
        projection_table[column] = projection_table[column].where(
            projection_table[column].notna(),
            "",
        )

    for column in NUMERIC_PROJECTION_COLUMNS:
        if column not in projection_table.columns:
            projection_table[column] = 0.0
        projection_table[column] = pd.to_numeric(
            projection_table[column],
            errors="coerce",
        )

    zero_fill_columns = [
        column
        for column in NUMERIC_PROJECTION_COLUMNS
        if column
        not in {
            "batting_average",
            "obp",
            "slg",
            "ops",
            "games_played",
            "at_bats",
            "plate_appearances",
        }
    ]
    projection_table[zero_fill_columns] = projection_table[zero_fill_columns].fillna(
        0.0
    )

    return projection_table
MARKET_TO_PROJECTION_COLUMN = {
    "batter_hits": "projected_hits",
    "batter_home_runs": "projected_home_runs",
    "batter_rbis": "projected_rbi",
    "batter_runs_scored": "projected_runs",
    "batter_stolen_bases": "projected_stolen_bases",
    "batter_total_bases": "projected_total_bases",
}


def _empty_projection_table() -> pd.DataFrame:
    """Return an empty projection table with the standard columns."""

    projection_table = pd.DataFrame(columns=PLAYER_PROJECTION_COLUMNS)
    projection_table.attrs["player_stats_loaded"] = False
    projection_table.attrs["player_stats_error"] = None

    return projection_table


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


def _to_bool(value, default: bool = False) -> bool:
    """Convert optional dataframe values to bool without tripping on pd.NA."""

    if pd.isna(value):
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1"}

    return bool(value)


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


def _get_playing_time_denominator(player_row) -> float:
    """Estimate games played for neutral season-rate projections."""

    games_played = _to_number(player_row.get("games_played", pd.NA))
    at_bats = _to_number(player_row.get("at_bats", pd.NA))
    plate_appearances = _to_number(player_row.get("plate_appearances", pd.NA))

    if games_played > 0:
        return games_played

    if plate_appearances > 0:
        return max(plate_appearances / 4.2, 1)

    if at_bats > 0:
        return max(at_bats / 3.8, 1)

    return 0.0


def _has_usable_secondary_stat_context(player_row) -> bool:
    """Return True when season stats can fill non-prop scoring categories."""

    if _get_playing_time_denominator(player_row) <= 0:
        return False

    secondary_columns = [
        "walks",
        "strikeouts",
        "hit_by_pitch",
        "intentional_walks",
        "sacrifices",
        "caught_stealing",
        "ground_into_double_play",
        "stolen_bases",
    ]
    return any(_to_number(player_row.get(column, pd.NA)) > 0 for column in secondary_columns)


def _estimate_secondary_scoring_fallbacks(player_row) -> dict:
    """Estimate secondary scoring categories from neutral season rates.

    These categories generally do not have hitter prop markets in our current
    feed, so sportsbook-projected players still need season-rate estimates for
    walks, strikeouts, HBP, IBB, sacrifices, caught stealing, and GIDP.
    """

    playing_time_denominator = _get_playing_time_denominator(player_row)

    if playing_time_denominator <= 0:
        return {
            "projected_walks": 0.0,
            "projected_strikeouts": 0.0,
            "projected_hit_by_pitch": 0.0,
            "projected_intentional_walks": 0.0,
            "projected_sacrifices": 0.0,
            "projected_caught_stealing": 0.0,
            "projected_ground_into_double_play": 0.0,
        }

    def per_game(stat_column: str) -> float:
        return _to_number(player_row.get(stat_column, pd.NA)) / playing_time_denominator

    stolen_bases = _to_number(player_row.get("stolen_bases", pd.NA))
    caught_stealing = _to_number(player_row.get("caught_stealing", pd.NA))
    projected_stolen_bases = _to_number(player_row.get("projected_stolen_bases", 0.0))
    if projected_stolen_bases <= 0:
        projected_stolen_bases = per_game("stolen_bases")
    projected_caught_stealing = 0.0

    if stolen_bases + caught_stealing > 0:
        projected_caught_stealing = projected_stolen_bases * (
            caught_stealing / (stolen_bases + caught_stealing)
        )

    return {
        "projected_walks": per_game("walks"),
        "projected_strikeouts": per_game("strikeouts"),
        "projected_hit_by_pitch": per_game("hit_by_pitch"),
        "projected_intentional_walks": per_game("intentional_walks"),
        "projected_sacrifices": per_game("sacrifices"),
        "projected_caught_stealing": projected_caught_stealing,
        "projected_ground_into_double_play": per_game("ground_into_double_play"),
    }


def _estimate_stat_based_fantasy_points(player_row) -> dict:
    """Estimate a neutral daily projection from season-to-date stats.

    This fallback is intentionally simple and lower confidence. It uses season
    per-game rates as neutral baselines. It does not apply a vig-style discount
    because season stats are not sportsbook prices.
    """

    playing_time_denominator = _get_playing_time_denominator(player_row)
    matchup_adjustment_factor = 1.00

    if playing_time_denominator <= 0:
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
    hit_by_pitch = _to_number(player_row.get("hit_by_pitch", pd.NA))
    intentional_walks = _to_number(player_row.get("intentional_walks", pd.NA))
    sacrifices = _to_number(player_row.get("sacrifices", pd.NA))
    caught_stealing = _to_number(player_row.get("caught_stealing", pd.NA))
    ground_into_double_play = _to_number(
        player_row.get("ground_into_double_play", pd.NA)
    )
    singles = max(hits - doubles - triples - home_runs, 0)
    total_bases = singles + (doubles * 2) + (triples * 3) + (home_runs * 4)

    def per_game(value):
        """Convert a season total into a neutral per-game estimate."""

        return (value / playing_time_denominator) * matchup_adjustment_factor

    projected_stolen_bases = per_game(stolen_bases)
    projected_caught_stealing = 0.0

    if stolen_bases + caught_stealing > 0:
        projected_caught_stealing = projected_stolen_bases * (
            caught_stealing / (stolen_bases + caught_stealing)
        )

    fallback_values = {
        "projected_hits": per_game(hits),
        "projected_home_runs": per_game(home_runs),
        "projected_rbi": per_game(rbi),
        "projected_runs": per_game(runs),
        "projected_stolen_bases": projected_stolen_bases,
        "projected_total_bases": per_game(total_bases),
        "projected_walks": per_game(walks),
        "projected_strikeouts": per_game(strikeouts),
        "projected_hit_by_pitch": per_game(hit_by_pitch),
        "projected_intentional_walks": per_game(intentional_walks),
        "projected_sacrifices": per_game(sacrifices),
        "projected_caught_stealing": projected_caught_stealing,
        "projected_ground_into_double_play": per_game(ground_into_double_play),
    }
    fallback_values.update(_estimate_secondary_scoring_fallbacks(player_row))
    fallback_values.update(
        estimate_hit_type_breakdown(
            fallback_values["projected_hits"],
            fallback_values["projected_home_runs"],
            fallback_values["projected_total_bases"],
            player_row,
        )
    )

    # Game-winning RBI is a placeholder until we model game run distribution.
    fallback_values["projected_game_winning_rbi"] = (
        fallback_values["projected_rbi"] * 0.12
    )
    # Grand slams are approximated as 2.25% of home runs.
    fallback_values["projected_grand_slams"] = (
        fallback_values["projected_home_runs"] * 0.0225
    )
    # Cycles are too rare to project meaningfully at this stage.
    fallback_values["projected_hit_for_cycle"] = 0.0
    fallback_values["projected_fantasy_points"] = calculate_hitter_fantasy_points(
        fallback_values
    )

    return fallback_values


def _estimate_stat_based_projection_values(player_row) -> dict:
    """Estimate daily stat categories from season-to-date per-game rates.

    Sportsbook props are still the primary projection source. This fallback only
    fills markets that are missing from today's odds board. These are neutral
    per-game baselines, not vig-adjusted sportsbook probabilities.
    """

    fallback_values = _estimate_stat_based_fantasy_points(player_row)

    return {
        column: fallback_values.get(column, 0.0)
        for column in MARKET_TO_PROJECTION_COLUMN.values()
    }


def _estimate_additional_scoring_fallbacks(player_row) -> dict:
    """Return neutral fallbacks for scoring categories without prop markets."""

    fallback_values = _estimate_stat_based_fantasy_points(player_row)
    additional_columns = [
        "projected_walks",
        "projected_strikeouts",
        "projected_hit_by_pitch",
        "projected_intentional_walks",
        "projected_sacrifices",
        "projected_caught_stealing",
        "projected_ground_into_double_play",
    ]

    return {
        column: fallback_values.get(column, 0.0)
        for column in additional_columns
    }


def _recalculate_hit_breakdown_and_points(projection_table: pd.DataFrame) -> pd.DataFrame:
    """Recalculate hit-type estimates and fantasy points after fallbacks fill gaps."""

    projection_table = projection_table.copy()
    numeric_columns = [
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
    ]

    for column in numeric_columns:
        projection_table[column] = pd.to_numeric(
            projection_table[column],
            errors="coerce",
        ).fillna(0.0)

    minimum_total_bases = projection_table["projected_hits"] + (
        3 * projection_table["projected_home_runs"]
    )
    needs_total_base_reconciliation = (
        projection_table["projected_total_bases"] + 0.01
        < minimum_total_bases
    )
    projection_table.loc[
        needs_total_base_reconciliation,
        "projected_total_bases",
    ] = minimum_total_bases[needs_total_base_reconciliation]

    if "projection_adjustment_note" not in projection_table.columns:
        projection_table["projection_adjustment_note"] = ""

    projection_table.loc[
        needs_total_base_reconciliation,
        "projection_adjustment_note",
    ] = (
        projection_table.loc[
            needs_total_base_reconciliation,
            "projection_adjustment_note",
        ]
        .fillna("")
        .astype(str)
        + " Total bases reconciled upward after fallback fill."
    ).str.strip()

    hit_breakdown = projection_table.apply(
        lambda row: estimate_hit_type_breakdown(
            row["projected_hits"],
            row["projected_home_runs"],
            row["projected_total_bases"],
            row,
        ),
        axis=1,
        result_type="expand",
    )

    for column in hit_breakdown.columns:
        projection_table[column] = hit_breakdown[column]

    # TODO: Game-winning RBI can eventually use projected RBI divided by total
    # game RBI opportunities or expected total runs in the game.
    projection_table["projected_game_winning_rbi"] = (
        projection_table["projected_rbi"] * 0.12
    )
    # Grand slams are approximated as 2.25% of home runs until base-state odds
    # are modeled.
    projection_table["projected_grand_slams"] = (
        projection_table["projected_home_runs"] * 0.0225
    )
    # Cycles are ignored because they are too rare to project meaningfully.
    projection_table["projected_hit_for_cycle"] = 0.0
    projection_table["projected_fantasy_points"] = projection_table.apply(
        calculate_hitter_fantasy_points,
        axis=1,
    )

    return projection_table


def _parse_available_markets(value) -> set[str]:
    """Turn the markets_available display string into a set of API market names."""

    if not isinstance(value, str) or not value.strip():
        return set()

    return {market.strip() for market in value.split(",") if market.strip()}


def _apply_missing_market_fallbacks(
    projection_table: pd.DataFrame,
    include_unavailable_players: bool = False,
) -> pd.DataFrame:
    """Fill missing sportsbook markets with conservative stat-based estimates."""

    if projection_table.empty:
        return projection_table

    projection_table = ensure_projection_column_types(projection_table)

    for index, player_row in projection_table.iterrows():
        available_markets = _parse_available_markets(
            player_row.get("markets_available", "")
        )
        missing_markets = [
            market
            for market in MARKET_TO_PROJECTION_COLUMN
            if market not in available_markets
        ]
        fallback_markets_used = []
        fallback_note = ""
        additional_fallback_values = {}
        secondary_fallback_values = {}
        secondary_fallback_stats_used = []
        secondary_fallback_note = ""

        # Prop rows can also have a market present but an empty value after
        # cleaning. Treat that as missing for fallback purposes.
        for market, projection_column in MARKET_TO_PROJECTION_COLUMN.items():
            if pd.isna(player_row.get(projection_column, pd.NA)):
                if market not in missing_markets:
                    missing_markets.append(market)

        can_use_fallback = _has_usable_stat_context(player_row)
        can_use_secondary_fallback = _has_usable_secondary_stat_context(player_row)

        if (
            not include_unavailable_players
            and not _to_bool(player_row.get("is_available_today", True), True)
        ):
            can_use_fallback = False
            can_use_secondary_fallback = False
            if missing_markets:
                if not _to_bool(player_row.get("has_game_today", True), True):
                    fallback_note = (
                        "Projection suppressed because player's MLB team does not play today."
                    )
                else:
                    fallback_note = (
                        "Projection suppressed because player is unavailable."
                    )

        if can_use_fallback:
            fallback_values = _estimate_stat_based_projection_values(player_row)

            for market in missing_markets:
                projection_column = MARKET_TO_PROJECTION_COLUMN[market]
                fallback_value = fallback_values.get(projection_column, 0.0)

                if fallback_value > 0:
                    projection_table.at[index, projection_column] = fallback_value
                    fallback_markets_used.append(market)

            if fallback_markets_used:
                fallback_note = (
                    "Missing sportsbook markets filled from neutral season "
                    "per-game stats. No arbitrary 0.85 discount was applied."
                )

        if can_use_secondary_fallback:
            updated_player_row = projection_table.loc[index].copy()
            secondary_fallback_values = _estimate_secondary_scoring_fallbacks(
                updated_player_row
            )
            secondary_stat_columns = {
                "projected_walks": "BB",
                "projected_strikeouts": "K",
                "projected_hit_by_pitch": "HBP",
                "projected_intentional_walks": "IBB",
                "projected_sacrifices": "SAC",
                "projected_caught_stealing": "CS",
                "projected_ground_into_double_play": "GIDP",
            }

            for projection_column, source_label in secondary_stat_columns.items():
                fallback_value = secondary_fallback_values.get(projection_column, 0.0)
                projection_table.at[index, projection_column] = fallback_value
                if fallback_value > 0:
                    secondary_fallback_stats_used.append(source_label)

            if secondary_fallback_stats_used:
                secondary_fallback_note = (
                    "Secondary scoring categories filled from neutral season "
                    "per-game rates even though primary projection may use sportsbook props."
                )
        elif not str(player_row.get("matched_stats_player", "")).strip():
            secondary_fallback_note = "No season stats matched."
        elif _get_playing_time_denominator(player_row) <= 0:
            secondary_fallback_note = (
                "Season stats matched, but games played, plate appearances, "
                "and at-bats were unavailable."
            )

        projection_table.at[index, "missing_markets"] = ", ".join(missing_markets)
        projection_table.at[index, "fallback_markets_used"] = ", ".join(
            fallback_markets_used
        )
        projection_table.at[index, "fallback_projection_note"] = fallback_note
        projection_table.at[index, "secondary_fallback_stats_used"] = ", ".join(
            secondary_fallback_stats_used
        )
        projection_table.at[index, "secondary_fallback_note"] = secondary_fallback_note

    projection_table = _recalculate_hit_breakdown_and_points(projection_table)
    projection_table = ensure_projection_column_types(projection_table)

    return projection_table


def _get_availability_position_column(
    availability_dataframe: pd.DataFrame,
) -> str | None:
    """Return the uploaded CSV column to use for position eligibility."""

    if "eligible_positions" in availability_dataframe.columns:
        return "eligible_positions"

    if "position" in availability_dataframe.columns:
        return "position"

    return None


def _player_has_sportsbook_markets(player_row) -> bool:
    """Return True when the row has today's sportsbook prop markets."""

    markets_available = player_row.get("markets_available", "")

    return isinstance(markets_available, str) and bool(markets_available.strip())


def _get_player_team_code(player_row) -> str:
    """Read ESPN/CSV team metadata and normalize it to an MLB team code."""

    for column in ["pro_team", "team"]:
        team_code = normalize_mlb_team(player_row.get(column, ""))

        if team_code:
            return team_code

    return ""


def _player_has_game_today(player_row, teams_playing_today: set[str]) -> bool:
    """Return True when the player should be eligible for today's games."""

    if _player_has_sportsbook_markets(player_row):
        return True

    team_code = _get_player_team_code(player_row)

    if not team_code:
        return False

    return team_code in teams_playing_today


def _game_today_note_from_row(player_row, teams_playing_today: set[str]) -> str:
    """Return a beginner-friendly explanation of game-today status."""

    if _player_has_sportsbook_markets(player_row):
        return "Game today inferred from sportsbook prop market."

    team_code = _get_player_team_code(player_row)

    if not team_code:
        return "No game today: MLB team could not be identified."

    if team_code in teams_playing_today:
        return "MLB team plays today."

    return "No game today: player's MLB team does not play today."


def _availability_note_from_row(player_row) -> str:
    """Build a plain-English availability note from ESPN/CSV context."""

    if is_player_unavailable(player_row):
        for label, column in [
            ("injury status", "injury_status"),
            ("status", "status"),
            ("player notes", "player_notes"),
            ("lineup status", "lineup_status"),
            ("fantasy status", "fantasy_status"),
            ("roster spot", "roster_spot"),
        ]:
            value = player_row.get(column, "")

            if value:
                return f"Unavailable: {label} is {value}"

        return "Unavailable"

    return ""


def _apply_availability_flags(
    projection_table: pd.DataFrame,
    teams_playing_today: set[str] | None = None,
) -> pd.DataFrame:
    """Add injury and game-today flags used by daily projections."""

    if projection_table.empty:
        projection_table["is_available_today"] = True
        projection_table["availability_note"] = ""
        projection_table["has_game_today"] = True
        projection_table["game_today_note"] = ""
        return projection_table

    projection_table = projection_table.copy()
    teams_playing_today = teams_playing_today or set()
    notes = projection_table.apply(_availability_note_from_row, axis=1)
    has_game_today = projection_table.apply(
        lambda row: _player_has_game_today(row, teams_playing_today),
        axis=1,
    )
    game_today_notes = projection_table.apply(
        lambda row: _game_today_note_from_row(row, teams_playing_today),
        axis=1,
    )
    projection_table["availability_note"] = notes
    projection_table["has_game_today"] = has_game_today
    projection_table["game_today_note"] = game_today_notes
    projection_table["is_available_today"] = ~projection_table.apply(
        is_player_unavailable,
        axis=1,
    ) & projection_table["has_game_today"]

    return projection_table


def _build_uploaded_player_table(
    availability_dataframe: pd.DataFrame | None,
) -> pd.DataFrame:
    """Create one row per uploaded player for fallback or missing projections."""

    if availability_dataframe is None or availability_dataframe.empty:
        return pd.DataFrame(columns=["player", "eligible_positions"])

    if "player" not in availability_dataframe.columns:
        return pd.DataFrame(columns=["player", "eligible_positions"])

    position_column = _get_availability_position_column(availability_dataframe)
    metadata_columns = [
        column
        for column in ["player"] + AVAILABILITY_METADATA_COLUMNS
        if column in availability_dataframe.columns
    ]
    uploaded_players = availability_dataframe[metadata_columns].copy()

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


def _enrich_availability_metadata(
    projection_table: pd.DataFrame,
    availability_dataframe: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach ESPN/CSV availability context to projection rows."""

    if availability_dataframe is None or availability_dataframe.empty:
        for column in AVAILABILITY_METADATA_COLUMNS:
            if column not in projection_table.columns:
                projection_table[column] = ""
        return projection_table

    if "player" not in availability_dataframe.columns:
        return projection_table

    metadata_columns = [
        column
        for column in ["player"] + AVAILABILITY_METADATA_COLUMNS
        if column in availability_dataframe.columns
    ]

    if len(metadata_columns) == 1:
        return projection_table

    metadata_lookup = availability_dataframe[metadata_columns].copy()
    metadata_lookup["normalized_player_name"] = metadata_lookup["player"].apply(
        normalize_player_name
    )
    metadata_lookup = metadata_lookup.drop(columns=["player"]).drop_duplicates(
        subset=["normalized_player_name"],
        keep="first",
    )
    projection_table["normalized_player_name"] = projection_table["player"].apply(
        normalize_player_name
    )
    projection_table = projection_table.merge(
        metadata_lookup,
        on="normalized_player_name",
        how="left",
        suffixes=("", "_availability"),
    )
    projection_table = projection_table.drop(columns=["normalized_player_name"])

    for column in AVAILABILITY_METADATA_COLUMNS:
        if column not in projection_table.columns:
            projection_table[column] = ""
        else:
            projection_table[column] = projection_table[column].fillna("")

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


def _apply_projection_sources(
    projection_table: pd.DataFrame,
    include_unavailable_players: bool = False,
) -> pd.DataFrame:
    """Label each row as sportsbook, sportsbook+fallback, fallback, or missing."""

    if projection_table.empty:
        return projection_table

    projection_table = projection_table.copy()

    for index, player_row in projection_table.iterrows():
        fallback_used = isinstance(
            player_row.get("fallback_markets_used", ""),
            str,
        ) and bool(player_row.get("fallback_markets_used", "").strip())
        has_sportsbook_markets = isinstance(
            player_row.get("markets_available", ""),
            str,
        ) and bool(player_row.get("markets_available", "").strip())

        if has_sportsbook_markets:
            missing_market_count = len(
                _parse_available_markets(player_row.get("missing_markets", ""))
            )
            if fallback_used:
                projection_table.at[index, "projection_source"] = (
                    "Sportsbook + stat fallback"
                )
                projection_table.at[index, "projection_confidence"] = (
                    "Low" if missing_market_count >= 3 else "Medium"
                )
            else:
                projection_table.at[index, "projection_source"] = "Sportsbook lines"
                if missing_market_count >= 3:
                    confidence = "Low"
                elif _to_bool(player_row.get("vig_removed", False)):
                    confidence = "High"
                else:
                    confidence = "Medium"

                projection_table.at[index, "projection_confidence"] = confidence
            continue

        if (
            not include_unavailable_players
            and not _to_bool(player_row.get("is_available_today", True), True)
        ):
            projection_table.at[index, "projected_fantasy_points"] = 0.0
            if not _to_bool(player_row.get("has_game_today", True), True):
                projection_table.at[index, "projection_source"] = "No game today"
                projection_table.at[index, "fallback_projection_note"] = (
                    "Projection suppressed because player's MLB team does not play today."
                )
            else:
                projection_table.at[index, "projection_source"] = "Unavailable"
                projection_table.at[index, "fallback_projection_note"] = (
                    "Projection suppressed because player is unavailable."
                )
            projection_table.at[index, "projection_confidence"] = "Unavailable"
            projection_table.at[index, "availability_note"] = (
                f"{player_row.get('availability_note', '')}; "
                "fallback projection suppressed"
            ).strip("; ")
            continue

        if fallback_used or _has_usable_stat_context(player_row):
            if not fallback_used:
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
    force_refresh_odds: bool = False,
    include_unavailable_players: bool = False,
    include_players_without_games: bool = False,
    prop_rows: list[dict] | None = None,
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

    if prop_rows is None:
        prop_rows = fetch_todays_mlb_hitter_props(
            force_refresh=force_refresh_odds,
        )

    prop_projection_table = build_hitter_prop_projection_table(prop_rows)

    projection_table = prop_projection_table.copy()

    # Team is part of the long-term data model, but it is not available from the
    # current prop feed yet.
    projection_table["team"] = ""
    projection_table["projection_source"] = "Sportsbook lines"
    projection_table["projection_confidence"] = "High"
    projection_table = _enrich_positions_from_availability(
        projection_table,
        availability_dataframe,
    )
    projection_table = _enrich_availability_metadata(
        projection_table,
        availability_dataframe,
    )
    projection_table = _add_uploaded_players_without_lines(
        projection_table,
        availability_dataframe,
    )
    teams_playing_today = get_mlb_teams_playing_today_from_cache()
    projection_table = _apply_availability_flags(
        projection_table,
        teams_playing_today=teams_playing_today,
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
        projection_table["matched_stats_player"] = ""
        projection_table["normalized_stats_name"] = ""
        projection_table["stats_match_method"] = "stats load failed"
        projection_table["stats_match_score"] = 0

    projection_table = _apply_missing_market_fallbacks(
        projection_table,
        include_unavailable_players=include_unavailable_players,
    )
    projection_table = _apply_projection_sources(
        projection_table,
        include_unavailable_players=include_unavailable_players,
    )
    projection_table["has_sportsbook_props"] = projection_table.apply(
        lambda row: (
            "Sportsbook" in str(row.get("projection_source", ""))
            or _player_has_sportsbook_markets(row)
            or pd.to_numeric(row.get("bookmaker_count", 0), errors="coerce") > 0
        ),
        axis=1,
    )

    no_game_mask = ~projection_table["has_game_today"]
    projection_table.loc[no_game_mask, "projected_fantasy_points"] = 0.0
    projection_table.loc[no_game_mask, "projection_source"] = "No game today"
    projection_table.loc[no_game_mask, "projection_confidence"] = "Unavailable"
    projection_table.loc[
        no_game_mask,
        "fallback_projection_note",
    ] = "Projection suppressed because player's MLB team does not play today."

    if not include_unavailable_players:
        unavailable_mask = (
            ~projection_table["is_available_today"] & projection_table["has_game_today"]
        )
        projection_table.loc[
            unavailable_mask,
            "projected_fantasy_points",
        ] = 0.0
        projection_table.loc[
            unavailable_mask,
            "projection_source",
        ] = "Unavailable"
        projection_table.loc[
            unavailable_mask,
            "projection_confidence",
        ] = "Unavailable"
        projection_table.loc[
            unavailable_mask,
            "fallback_projection_note",
        ] = "Projection suppressed because player is unavailable."

    # Name keys let ESPN, CSV, and sportsbook player names join reliably. We
    # compute them from the final display name so blank upstream keys cannot
    # break Top Pickups matching.
    projection_table["normalized_player_name"] = projection_table["player"].apply(
        normalize_player_name
    )
    projection_table["player_match_key"] = projection_table["player"].apply(
        build_player_match_key
    )

    # Keep the engine output stable for every page that consumes it.
    for column in PLAYER_PROJECTION_COLUMNS:
        if column not in projection_table.columns:
            projection_table[column] = pd.NA

    projection_table = ensure_projection_column_types(projection_table)
    projection_table = projection_table[PLAYER_PROJECTION_COLUMNS].sort_values(
        by="projected_fantasy_points",
        ascending=False,
    )

    output_table = projection_table.reset_index(drop=True)
    output_table.attrs["player_stats_loaded"] = stats_loaded
    output_table.attrs["player_stats_error"] = stats_error
    stats_matched = (
        output_table["matched_stats_player"].fillna("").astype(str).str.strip() != ""
        if "matched_stats_player" in output_table.columns
        else pd.Series([False] * len(output_table))
    )
    sportsbook_projected = (
        output_table["projection_source"].fillna("").astype(str).str.contains(
            "Sportsbook",
            case=False,
        )
        if "projection_source" in output_table.columns
        else pd.Series([False] * len(output_table))
    )
    stat_fallback_only = (
        output_table["projection_source"].fillna("").astype(str).eq(
            "Stat-based fallback"
        )
        if "projection_source" in output_table.columns
        else pd.Series([False] * len(output_table))
    )
    output_table.attrs["season_stats_match_counts"] = {
        "total_projection_rows": int(len(output_table)),
        "projection_rows_with_season_stats_matched": int(stats_matched.sum()),
        "projection_rows_missing_season_stats": int((~stats_matched).sum()),
        "sportsbook_projected_rows_missing_season_stats": int(
            (sportsbook_projected & ~stats_matched).sum()
        ),
        "stat_fallback_only_rows_missing_season_stats": int(
            (stat_fallback_only & ~stats_matched).sum()
        ),
    }
    output_table.attrs["odds_data_source"] = get_odds_data_source_status()
    output_table.attrs["market_diagnostics_row_count"] = len(
        build_hitter_prop_market_diagnostics(prop_rows)
    )

    return output_table
