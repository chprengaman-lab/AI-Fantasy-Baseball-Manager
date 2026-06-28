"""Streamlit page for prop-based top pickup rankings.

This page ranks hitters using the same prop-based projection logic as the
Hitter Prop Projections page.
"""

import re

import pandas as pd
import streamlit as st

from config import LEAGUE_RULES
from services.odds_api import (
    MissingOddsAPIKeyError,
    OddsAPIError,
)
from services.lineup_optimizer import (
    calculate_replacement_value_by_position,
    compare_lineups,
    evaluate_multi_add_hitter_scenarios,
    evaluate_single_hitter_pickups,
    get_drop_protection_status,
    get_roster_flexibility_summary,
    optimize_hitter_lineup,
)
from services.player_projection_engine import build_player_projection_table
from services.player_stats import normalize_player_name


TOP_PICKUPS_COLUMNS = [
    "player",
    "team",
    "eligible_positions",
    "roster_status",
    "bookmaker",
    "tier",
    "projected_fantasy_points",
    "projection_source",
    "projection_confidence",
    "projected_hits",
    "projected_home_runs",
    "projected_rbi",
    "projected_runs",
    "projected_stolen_bases",
    "projected_total_bases",
    "batting_average",
    "obp",
    "slg",
    "ops",
]


def get_pickup_tier(projected_fantasy_points: float) -> str:
    """Return a simple pickup tier from projected fantasy points."""

    if projected_fantasy_points >= 8:
        return "Elite pickup"

    if projected_fantasy_points >= 6:
        return "Strong pickup"

    if projected_fantasy_points >= 4:
        return "Watchlist"

    return "Low priority"


def split_positions(eligible_positions: str) -> list[str]:
    """Split a position string like '1B,3B' or '2B/SS' into position tokens."""

    if not isinstance(eligible_positions, str) or not eligible_positions.strip():
        return []

    positions = re.split(r"[,/]", eligible_positions)
    return [position.strip().upper() for position in positions if position.strip()]


def player_has_position(eligible_positions: str, selected_position: str) -> bool:
    """Return True when a player is eligible at the selected position."""

    if selected_position == "All Positions":
        return True

    return selected_position in split_positions(eligible_positions)


def get_normalized_player_names(players_dataframe: pd.DataFrame) -> set[str]:
    """Return normalized player names from an uploaded CSV dataframe."""

    return set(players_dataframe["player"].dropna().apply(normalize_player_name))


def get_position_column(players_dataframe: pd.DataFrame) -> str | None:
    """Choose the position column from an uploaded CSV when one exists."""

    if "eligible_positions" in players_dataframe.columns:
        return "eligible_positions"

    if "position" in players_dataframe.columns:
        return "position"

    return None


def build_position_enrichment_dataframe(
    available_players_dataframe: pd.DataFrame | None,
    roster_dataframe: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Build a small dataframe for enriching eligible positions.

    Available-player positions are preferred first. Roster positions can fill in
    blanks when the same player appears there or when only a roster CSV exists.
    """

    position_dataframes = []

    for players_dataframe in [available_players_dataframe, roster_dataframe]:
        if players_dataframe is None or "player" not in players_dataframe.columns:
            continue

        position_column = get_position_column(players_dataframe)

        if position_column is None:
            continue

        position_dataframe = players_dataframe[["player", position_column]].copy()
        position_dataframe = position_dataframe.rename(
            columns={position_column: "eligible_positions"}
        )
        position_dataframe["normalized_player_name"] = position_dataframe[
            "player"
        ].apply(normalize_player_name)
        position_dataframes.append(position_dataframe)

    if not position_dataframes:
        return available_players_dataframe

    combined_dataframe = pd.concat(position_dataframes, ignore_index=True)

    return combined_dataframe.drop_duplicates(
        subset=["normalized_player_name"],
        keep="first",
    )


def add_roster_metadata(
    projection_table: pd.DataFrame,
    roster_dataframe: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach roster CSV fields needed by lineup and pickup optimizers."""

    if roster_dataframe is None or "player" not in roster_dataframe.columns:
        return projection_table

    metadata_columns = [
        column
        for column in [
            "player",
            "roster_spot",
            "droppable",
            "undroppable",
            "long_term_value",
            "keeper_status",
            "notes",
        ]
        if column in roster_dataframe.columns
    ]

    if len(metadata_columns) == 1:
        return projection_table

    roster_metadata = roster_dataframe[metadata_columns].copy()
    roster_metadata["normalized_player_name"] = roster_metadata["player"].apply(
        normalize_player_name
    )
    roster_metadata = roster_metadata.drop(columns=["player"]).drop_duplicates(
        subset=["normalized_player_name"],
        keep="first",
    )

    return projection_table.merge(
        roster_metadata,
        on="normalized_player_name",
        how="left",
    )


def get_roster_status(normalized_player_name: str, available_names: set, roster_names: set) -> str:
    """Return whether a player is available, rostered, or unknown."""

    if normalized_player_name in roster_names:
        return "On My Roster"

    if normalized_player_name in available_names:
        return "Available"

    return "Unknown"


def build_available_players_template() -> pd.DataFrame:
    """Create an example available-players CSV template."""

    return pd.DataFrame(
        [
            {
                "player": "Example Catcher",
                "eligible_positions": "C,DH",
                "player_type": "Hitter",
            },
            {
                "player": "Example Left Fielder",
                "eligible_positions": "LF,DH",
                "player_type": "Hitter",
            },
            {
                "player": "Example Third Baseman",
                "eligible_positions": "3B,1B",
                "player_type": "Hitter",
            },
        ]
    )


def build_roster_template() -> pd.DataFrame:
    """Create an example current-roster CSV template."""

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
                "notes": "IL players are protected from drop recommendations",
            },
        ]
    )


def build_roster_protection_review(roster_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create the table that explains which roster players can be dropped."""

    review_columns = [
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

    if roster_dataframe is None or roster_dataframe.empty:
        return pd.DataFrame(columns=review_columns)

    review_dataframe = roster_dataframe.copy()

    # Uploaded CSVs are allowed to omit optional columns. Add blank versions so
    # the review table always has the same beginner-friendly shape.
    for column in review_columns:
        if column not in review_dataframe.columns:
            review_dataframe[column] = ""

    protection_details = review_dataframe.apply(
        get_drop_protection_status,
        axis=1,
        result_type="expand",
    )
    review_dataframe["inferred_drop_status"] = protection_details[0]
    review_dataframe["drop_reason"] = protection_details[1]

    return review_dataframe[review_columns]


def get_projection_coverage_status(projection_source: str) -> str:
    """Translate projection source into a plain coverage status."""

    if projection_source == "Sportsbook lines":
        return "Line Projection"

    if projection_source == "Stat-based fallback":
        return "Fallback Projection"

    return "Missing Projection"


def build_projection_coverage_review(projection_table: pd.DataFrame) -> pd.DataFrame:
    """Show whether uploaded players have line, fallback, or missing projections."""

    coverage_columns = [
        "player",
        "roster_status",
        "eligible_positions",
        "projected_fantasy_points",
        "projection_source",
        "projection_confidence",
        "coverage_status",
    ]

    if projection_table.empty:
        return pd.DataFrame(columns=coverage_columns)

    review_dataframe = projection_table.copy()
    review_dataframe["coverage_status"] = review_dataframe["projection_source"].apply(
        get_projection_coverage_status
    )

    return review_dataframe[coverage_columns].sort_values(
        by=["coverage_status", "player"],
        ascending=[True, True],
    )


def show_weekly_sp_start_tracker() -> dict:
    """Display weekly SP start context without changing optimizer behavior."""

    weekly_sp_start_cap = LEAGUE_RULES["weekly_sp_start_cap"]

    st.header("Weekly SP Start Tracker")
    st.write(
        "This is context only. It helps you decide whether open active roster "
        "spots should be used for hitter streaming or saved for future SP starts."
    )

    input_columns = st.columns(2)
    starts_used = input_columns[0].number_input(
        "SP starts already used this week",
        min_value=0,
        value=0,
        step=1,
    )
    starts_planned = input_columns[1].number_input(
        "SP starts currently planned for rest of week",
        min_value=0,
        value=0,
        step=1,
    )
    total_projected_starts = starts_used + starts_planned
    starts_remaining = weekly_sp_start_cap - total_projected_starts

    metric_columns = st.columns(3)
    metric_columns[0].metric("SP Starts Used", starts_used)
    metric_columns[1].metric("SP Starts Planned", starts_planned)
    metric_columns[2].metric("SP Starts Remaining", starts_remaining)

    if total_projected_starts > weekly_sp_start_cap:
        st.warning(
            "Warning: your planned SP starts exceed the weekly cap. Some starts "
            "may not count."
        )
    elif starts_remaining == 0:
        st.info(
            "You have filled your weekly SP start capacity, so extra SPs may be "
            "less valuable for the rest of this week."
        )
    else:
        st.info(
            "You still have SP starts available this week, so be careful "
            "dropping or avoiding SP depth if you need future starts."
        )

    return {
        "starts_used": starts_used,
        "starts_planned": starts_planned,
        "weekly_sp_start_cap": weekly_sp_start_cap,
        "total_projected_starts": total_projected_starts,
        "starts_remaining": starts_remaining,
    }


def show_roster_flexibility_summary(
    roster_dataframe: pd.DataFrame,
    sp_start_context: dict | None = None,
) -> None:
    """Display active roster space, IL count, and pitcher count context."""

    roster_summary = get_roster_flexibility_summary(roster_dataframe)

    st.header("Roster Flexibility Summary")
    st.write(
        "This section explains whether you can add a hitter without dropping "
        "anyone. In this league, carrying fewer than 5 SP can create extra "
        "flexibility for hitter streaming."
    )

    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "Active Roster Spots Used",
        f"{roster_summary['active_roster_count']} / "
        f"{roster_summary['active_roster_limit']}",
    )
    metric_columns[1].metric(
        "Open Active Spots",
        roster_summary["open_active_spots"],
    )
    metric_columns[2].metric(
        "SP Count",
        f"{roster_summary['sp_count']} / {roster_summary['sp_limit']}",
    )
    metric_columns[3].metric(
        "RP Count",
        f"{roster_summary['rp_count']} / {roster_summary['rp_limit']}",
    )
    metric_columns[4].metric(
        "IL Count",
        f"{roster_summary['il_count']} / {roster_summary['il_limit']}",
    )

    flexibility_note = roster_summary["flexibility_note"]

    if sp_start_context is not None:
        starts_remaining = sp_start_context["starts_remaining"]

        if starts_remaining == 0 and roster_summary["open_active_spots"] > 0:
            flexibility_note += (
                " Your weekly SP start capacity is already filled, so hitter "
                "streaming is especially attractive for open active roster spots."
            )
        elif starts_remaining > 0:
            flexibility_note += (
                " Hitter streaming is still useful, but balance it against "
                "future SP needs because you still have SP starts remaining."
            )

    st.info(flexibility_note)


def show_replacement_value_by_position(
    roster_dataframe: pd.DataFrame,
    available_players_dataframe: pd.DataFrame,
    projections_dataframe: pd.DataFrame,
) -> None:
    """Display roster-versus-waiver value for each exact hitter position."""

    replacement_value_table = calculate_replacement_value_by_position(
        roster_dataframe,
        available_players_dataframe,
        projections_dataframe,
    )

    st.header("Replacement Value by Position")
    st.write(
        "This helps identify positions where your current roster has a real "
        "advantage versus positions that can be streamed from the waiver wire."
    )
    st.dataframe(replacement_value_table.round(3), width="stretch")


def get_best_positive_move(
    pickup_recommendations: pd.DataFrame,
    multi_add_scenarios: pd.DataFrame | None = None,
):
    """Return the best positive-gain move, or None if one does not exist.

    No-drop moves preserve season-long roster value. The app ranks by
    risk_adjusted_score, then uses this tie-break order when scores are within
    0.5 points: Multi Add, Add Only, Add/Drop.
    """

    move_tables = []

    if pickup_recommendations is not None and not pickup_recommendations.empty:
        move_tables.append(pickup_recommendations)

    if multi_add_scenarios is not None and not multi_add_scenarios.empty:
        move_tables.append(multi_add_scenarios)

    if not move_tables:
        return None

    all_moves = pd.concat(move_tables, ignore_index=True, sort=False)
    positive_moves = all_moves[
        (all_moves["projected_gain"] > 0)
        & (all_moves["risk_adjusted_score"] > 0)
    ].copy()

    if positive_moves.empty:
        return None

    positive_moves = positive_moves.sort_values(
        by="risk_adjusted_score",
        ascending=False,
    )
    best_score = positive_moves.iloc[0]["risk_adjusted_score"]
    close_moves = positive_moves[
        best_score - positive_moves["risk_adjusted_score"] <= 0.5
    ].copy()
    move_type_priority = {
        "Multi Add": 0,
        "Add Only": 1,
        "Add/Drop": 2,
    }
    close_moves["move_type_priority"] = close_moves["move_type"].map(
        move_type_priority
    ).fillna(99)

    return close_moves.sort_values(
        by=["move_type_priority", "risk_adjusted_score"],
        ascending=[True, False],
    ).iloc[0]


def show_recommendation_table(
    recommendations: pd.DataFrame,
    columns: list[str],
    empty_message: str,
) -> None:
    """Display a recommendation table or a friendly empty-state message."""

    visible_columns = [
        column for column in columns if column in recommendations.columns
    ]

    if recommendations.empty:
        st.info(empty_message)
        return

    st.dataframe(recommendations[visible_columns].round(3), width="stretch")


def show_split_pickup_recommendations(
    pickup_recommendations: pd.DataFrame,
    selected_min_gain: float,
    show_no_gain_moves: bool,
) -> None:
    """Show add-only moves separately from add/drop moves."""

    st.write(
        "Add-only moves are usually safer because you do not lose a rostered "
        "player. Add/drop moves may improve today's lineup but can create "
        "season-long risk."
    )

    filtered_recommendations = pickup_recommendations[
        pickup_recommendations["projected_gain"] >= selected_min_gain
    ].copy()

    if not show_no_gain_moves:
        filtered_recommendations = filtered_recommendations[
            filtered_recommendations["projected_gain"] > 0
        ]

    add_only_moves = filtered_recommendations[
        filtered_recommendations["move_type"] == "Add Only"
    ].sort_values(by="risk_adjusted_score", ascending=False)
    add_drop_moves = filtered_recommendations[
        filtered_recommendations["move_type"] == "Add/Drop"
    ].sort_values(by="risk_adjusted_score", ascending=False)

    st.subheader("Best Add-Only Moves")
    show_recommendation_table(
        add_only_moves,
        [
            "add_player",
            "add_eligible_positions",
            "projected_gain",
            "risk_adjusted_score",
            "risk_adjusted_label",
            "new_starting_total",
            "recommendation",
            "add_projection_source",
            "add_projection_confidence",
        ],
        "No positive add-only moves found.",
    )

    st.subheader("Best Add/Drop Moves")
    show_recommendation_table(
        add_drop_moves,
        [
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
        ],
        "No positive add/drop moves found.",
    )


def show_multi_add_scenarios(
    multi_add_scenarios: pd.DataFrame,
    show_no_gain_moves: bool,
) -> None:
    """Display the best no-drop multi-add combinations."""

    st.header("Best Multi-Add Scenarios")
    st.write(
        "Multi-add scenarios are only evaluated when you have more than one "
        "open active roster spot. These moves do not require dropping a "
        "rostered player."
    )

    visible_scenarios = multi_add_scenarios.copy()

    if not show_no_gain_moves:
        visible_scenarios = visible_scenarios[visible_scenarios["projected_gain"] > 0]

    show_recommendation_table(
        visible_scenarios.sort_values(by="risk_adjusted_score", ascending=False),
        [
            "add_players",
            "add_eligible_positions",
            "number_of_adds",
            "projected_gain",
            "risk_adjusted_score",
            "risk_adjusted_label",
            "recommendation",
        ],
        "No positive multi-add scenarios found.",
    )


def get_confidence_text(best_move) -> str:
    """Explain projection confidence for the action-plan summary."""

    if best_move.get("move_type") == "Multi Add":
        return (
            "Confidence: Mixed because this multi-add recommendation may combine "
            "players from different projection sources. Lower-confidence added "
            "players are already penalized in the risk-adjusted score."
        )

    projection_source = best_move.get("add_projection_source", "")
    projection_confidence = best_move.get("add_projection_confidence", "")

    if projection_confidence == "High":
        return (
            "Confidence: High because this recommendation is based on "
            "sportsbook line projections."
        )

    if projection_confidence == "Low":
        return (
            "Confidence: Low because this recommendation uses fallback "
            "stat-based projections."
        )

    if projection_confidence == "Unknown":
        return (
            "Confidence: Unknown because the added player does not have a "
            "usable projection source."
        )

    return f"Confidence: {projection_confidence or 'Unknown'} from {projection_source or 'unknown projection source'}."


def show_todays_action_plan(best_move) -> None:
    """Show the simple top-level pickup recommendation for today."""

    st.header("Today's Action Plan")

    if best_move is None or best_move.get("risk_adjusted_score", 0) <= 0:
        st.warning(
            "No recommended pickup move today. Your best move may be to hold "
            "your roster."
        )
        return

    move_type = best_move["move_type"]
    projected_gain = best_move["projected_gain"]
    risk_adjusted_score = best_move["risk_adjusted_score"]
    risk_adjusted_label = best_move["risk_adjusted_label"]
    drop_risk = best_move.get("drop_risk", "")
    uses_open_roster_spot = move_type in {"Add Only", "Multi Add"}
    requires_drop = move_type == "Add/Drop"

    if move_type == "Multi Add":
        best_move_text = f"Best move today: Add {best_move['add_players']}."
        why_text = (
            f"Why: This is a Multi Add move, so it uses open roster spots and "
            f"does not require dropping anyone. It adds +{projected_gain:.1f} "
            f"raw projected points with a risk-adjusted score of "
            f"{risk_adjusted_score:.2f}. No-drop moves are preferred when "
            f"scores are close because they preserve season-long roster value."
        )
    elif move_type == "Add Only":
        best_move_text = (
            f"Best move today: Add {best_move['add_player']} without dropping anyone."
        )
        why_text = (
            f"Why: This is an Add Only move, so it uses an open roster spot and "
            f"does not require dropping a player. It adds +{projected_gain:.1f} "
            f"raw projected points with a risk-adjusted score of "
            f"{risk_adjusted_score:.2f}. No-drop moves are preferred when "
            f"scores are close because they preserve season-long roster value."
        )
    else:
        best_move_text = (
            f"Best move today: Add {best_move['add_player']} and drop "
            f"{best_move['drop_player']}."
        )
        why_text = (
            f"Why: This is an Add/Drop move, so it requires dropping a player. "
            f"It adds +{projected_gain:.1f} raw projected points with a "
            f"risk-adjusted score of {risk_adjusted_score:.2f}."
        )

        if drop_risk:
            why_text += f" Drop risk is {drop_risk}."

    detail_text = (
        f"{best_move_text}\n\n"
        f"Move type: {move_type}. Open roster spot used: "
        f"{'Yes' if uses_open_roster_spot else 'No'}. Requires drop: "
        f"{'Yes' if requires_drop else 'No'}. Risk-adjusted label: "
        f"{risk_adjusted_label}.\n\n"
        f"{why_text}\n\n"
        f"{get_confidence_text(best_move)}"
    )

    # Strong and good labels are positive enough to show as success unless the
    # move has meaningful drop risk. Medium/high drop risk gets a warning so the
    # season-long cost is visible before the user acts.
    if requires_drop and drop_risk in {"Medium", "High"}:
        st.warning(detail_text)
    elif risk_adjusted_label in {"Excellent", "Good"}:
        st.success(detail_text)
    elif risk_adjusted_label == "Thin edge":
        st.info(detail_text)
    else:
        st.warning(detail_text)


def show_best_move_summary(best_move) -> None:
    """Display a plain-English explanation of the best pickup move."""

    if best_move is None:
        st.info(
            "No positive-gain pickup move found today based on the current projections."
        )
        return

    projected_gain = best_move["projected_gain"]
    risk_adjusted_score = best_move["risk_adjusted_score"]

    if best_move["move_type"] == "Multi Add":
        st.success(
            f"Recommended move: Add {best_move['add_players']} without dropping "
            f"anyone because you have multiple open active roster spots. This "
            f"improves your projected starting hitter total by "
            f"+{projected_gain:.1f} raw points today, with a risk-adjusted "
            f"score of {risk_adjusted_score:.2f}."
        )
        st.write(
            "Why this move? The optimizer found a combination of hitters that "
            "improves your exact starting slots without giving up a rostered "
            "player. Only starting hitters score, so the combination is judged "
            "by lineup impact instead of bench points."
        )
    elif best_move["move_type"] == "Add Only":
        add_player = best_move["add_player"]
        add_positions = best_move["add_eligible_positions"]
        st.success(
            f"Recommended move: Add {add_player} ({add_positions}) without "
            f"dropping anyone because you have an open active roster spot. This "
            f"improves your projected starting hitter total by "
            f"+{projected_gain:.1f} raw points today, with a risk-adjusted "
            f"score of {risk_adjusted_score:.2f}."
        )
        st.write(
            f"Why this move? {add_player} is projected to improve one of your "
            f"exact hitter starting slots. Only starting hitters score, so the "
            f"optimizer focuses on the lineup impact instead of bench points. "
            f"LF, CF, RF, and DH are treated as exact positions. The raw "
            f"projected gain is today's point gain; the risk-adjusted score "
            f"also accounts for projection confidence."
        )
    else:
        add_player = best_move["add_player"]
        add_positions = best_move["add_eligible_positions"]
        drop_player = best_move["drop_player"]
        drop_positions = best_move["drop_eligible_positions"]

        st.success(
            f"Recommended move: Add {add_player} ({add_positions}) and drop "
            f"{drop_player} ({drop_positions}). This improves your projected "
            f"starting hitter total by +{projected_gain:.1f} raw points today, "
            f"with a risk-adjusted score of {risk_adjusted_score:.2f}."
        )
        st.write(
            f"Why this move? {add_player} fits an exact starting-slot need better "
            f"than the current roster construction after dropping {drop_player}. "
            f"Only starting hitters score, so bench-only value does not drive the "
            f"recommendation. LF, CF, RF, and DH are treated as exact positions. "
            f"The raw projected gain is today's point gain; the risk-adjusted "
            f"score also accounts for roster risk and projection confidence."
        )

        drop_risk = best_move.get("drop_risk", "")

        if drop_risk == "High":
            st.warning(
                "Warning: this move drops a high long-term value player. Only "
                "make this move if you are comfortable with the season-long risk. "
                "The risk-adjusted score is lower because dropping that player "
                "has season-long cost."
            )
        elif drop_risk == "Medium":
            st.warning(
                "Note: this move drops a player with some long-term value. "
                "Review before making the move. The risk-adjusted score is "
                "lower because dropping that player has season-long cost."
            )


def apply_pickup_move(
    roster_projection_table: pd.DataFrame,
    available_projection_table: pd.DataFrame,
    best_move,
) -> pd.DataFrame:
    """Return a new roster dataframe after applying the recommended move."""

    if best_move is None:
        return roster_projection_table.copy()

    if best_move["move_type"] == "Multi Add":
        add_player_names = [
            player_name.strip()
            for player_name in best_move["add_players"].split(",")
            if player_name.strip()
        ]
        add_player_rows = available_projection_table[
            available_projection_table["player"].isin(add_player_names)
        ].copy()

        if add_player_rows.empty:
            return roster_projection_table.copy()

        return pd.concat([roster_projection_table, add_player_rows], ignore_index=True)

    add_player = best_move["add_player"]
    add_player_row = available_projection_table[
        available_projection_table["player"] == add_player
    ].head(1)

    if add_player_row.empty:
        return roster_projection_table.copy()

    if best_move["move_type"] == "Add/Drop":
        drop_player = best_move["drop_player"]
        new_roster_table = roster_projection_table[
            roster_projection_table["player"] != drop_player
        ].copy()
    else:
        new_roster_table = roster_projection_table.copy()

    return pd.concat([new_roster_table, add_player_row], ignore_index=True)


def show_lineup_impact(
    roster_projection_table: pd.DataFrame,
    available_projection_table: pd.DataFrame,
    best_move,
) -> None:
    """Show how the best pickup changes the optimized starting lineup."""

    if best_move is None:
        return

    current_lineup_df, _, _ = optimize_hitter_lineup(roster_projection_table)
    new_roster_table = apply_pickup_move(
        roster_projection_table,
        available_projection_table,
        best_move,
    )
    new_lineup_df, _, _ = optimize_hitter_lineup(new_roster_table)
    changed_slots_df = compare_lineups(current_lineup_df, new_lineup_df)
    current_total = current_lineup_df["projected_fantasy_points"].sum()
    new_total = new_lineup_df["projected_fantasy_points"].sum()
    projected_gain = new_total - current_total

    st.subheader("Lineup Impact")
    st.write(
        "This compares your optimized hitter lineup before and after the "
        "recommended move."
    )

    metric_columns = st.columns(3)
    metric_columns[0].metric("Current Starting Total", round(current_total, 2))
    metric_columns[1].metric("New Starting Total", round(new_total, 2))
    metric_columns[2].metric("Projected Gain", round(projected_gain, 2))

    if changed_slots_df.empty:
        st.info("The optimized starting lineup slots do not change for this move.")
    else:
        st.dataframe(changed_slots_df.round(3), width="stretch")


st.title("Top Pickups")
st.info(
    "These rankings are based on sportsbook hitter props. They do not yet know "
    "which players are available in your fantasy league."
)
st.caption(
    "Upload a CSV of your waiver wire or available players list to filter these "
    "rankings to players you can actually add. The CSV must include a `player` "
    "column."
)
st.caption(
    "Upload a CSV with player and eligible_positions columns to filter "
    "recommendations by roster need."
)
st.caption(
    "You can also upload your current roster CSV to compare recommendations "
    "against players you already have."
)
st.caption(
    "If your active roster has fewer than 17 players, the app can recommend "
    "add-only hitter pickups without forcing a drop."
)
st.caption(
    "Fallback projections are lower confidence because they are based on "
    "season stats and do not fully account for today's matchup, ballpark, "
    "weather, or lineup spot."
)
st.caption(
    "Download a template, fill it out with your league data, then upload it "
    "back into the app."
)
st.markdown(
    """
CSV column guide:

- `player` = player name
- `eligible_positions` = exact ESPN positions like `C`, `1B`, `2B`, `3B`, `SS`, `LF`, `CF`, `RF`, `DH`, `SP`, `RP`
- `roster_spot` = `Starter`, `Bench`, `IL`, `SP`, `RP`, etc.
- `player_type` = `Hitter`, `SP`, or `RP`
- `droppable` = `TRUE` if the app may suggest dropping this player
- `undroppable` = `TRUE` if the app should never suggest dropping this player
- `long_term_value` = optional season-long value label like `High`, `Medium`, or `Low`
- `keeper_status` = optional roster strategy label like `Core`, `Hold`, `Streamer`, or `Drop Candidate`
- `notes` = optional context for your own review
"""
)
st.warning(
    "LF, CF, RF, and DH are treated as exact positions. Do not enter OF unless "
    "your league actually uses OF."
)
st.warning(
    "If you leave droppable and undroppable blank, the app will use conservative "
    "default logic."
)


# Build templates with pandas so the generated CSVs always match the supported
# upload columns.
available_players_template = build_available_players_template()
roster_template = build_roster_template()

st.download_button(
    "Download available players CSV template",
    data=available_players_template.to_csv(index=False),
    file_name="available_players_template.csv",
    mime="text/csv",
)
st.download_button(
    "Download roster CSV template",
    data=roster_template.to_csv(index=False),
    file_name="roster_template.csv",
    mime="text/csv",
)


uploaded_available_players = st.file_uploader(
    "Upload available players CSV",
    type=["csv"],
)
uploaded_roster = st.file_uploader(
    "Upload your current roster CSV",
    type=["csv"],
)


@st.cache_data(ttl=60 * 60)
def load_cached_player_projection_table(availability_dataframe=None):
    """Build and cache the unified projection table for one hour."""

    return build_player_projection_table(availability_dataframe)


try:
    available_players_dataframe = None
    roster_dataframe = None
    has_valid_available_players_csv = False
    has_valid_roster_csv = False
    show_roster_protection_review = False
    sp_start_context = None

    if uploaded_available_players is not None:
        available_players_dataframe = pd.read_csv(uploaded_available_players)

        if "player" not in available_players_dataframe.columns:
            st.warning(
                "The uploaded CSV must include a `player` column. Showing all "
                "ranked players instead."
            )
            available_players_dataframe = None
        else:
            has_valid_available_players_csv = True

    if uploaded_roster is not None:
        roster_dataframe = pd.read_csv(uploaded_roster)

        if "player" not in roster_dataframe.columns:
            st.warning(
                "The roster CSV must include a `player` column. Roster matching "
                "will be skipped."
            )
            roster_dataframe = None
        else:
            has_valid_roster_csv = True
            show_roster_protection_review = st.sidebar.checkbox(
                "Show roster protection review",
                value=True,
            )

    if has_valid_roster_csv:
        sp_start_context = show_weekly_sp_start_tracker()
        show_roster_flexibility_summary(roster_dataframe, sp_start_context)

    if has_valid_roster_csv and show_roster_protection_review:
        st.header("Roster Protection Review")
        st.write(
            "The optimizer will only suggest dropping players marked droppable. "
            "If you leave droppable and undroppable blank, the app protects the "
            "player by default."
        )
        st.dataframe(
            build_roster_protection_review(roster_dataframe),
            width="stretch",
        )

    position_enrichment_dataframe = build_position_enrichment_dataframe(
        available_players_dataframe,
        roster_dataframe,
    )

    pickup_table = load_cached_player_projection_table(position_enrichment_dataframe)

    if pickup_table.empty:
        st.info("No player projections are available right now.")
    else:
        pickup_table["normalized_player_name"] = pickup_table["player"].apply(
            normalize_player_name
        )
        available_player_names = (
            get_normalized_player_names(available_players_dataframe)
            if has_valid_available_players_csv
            else set()
        )
        roster_player_names = (
            get_normalized_player_names(roster_dataframe)
            if has_valid_roster_csv
            else set()
        )
        pickup_table["roster_status"] = pickup_table["normalized_player_name"].apply(
            lambda player_name: get_roster_status(
                player_name,
                available_player_names,
                roster_player_names,
            )
        )
        pickup_table = add_roster_metadata(pickup_table, roster_dataframe)
        master_pickup_table = pickup_table.copy()

        if has_valid_roster_csv or has_valid_available_players_csv:
            st.header("Projection Coverage Review")
            st.write(
                "This separates players with sportsbook-line projections from "
                "players using lower-confidence stat-based fallback projections "
                "or no usable projection."
            )
            st.dataframe(
                build_projection_coverage_review(master_pickup_table).round(3),
                width="stretch",
            )

        if has_valid_roster_csv and has_valid_available_players_csv:
            show_replacement_value_by_position(
                roster_dataframe,
                available_players_dataframe,
                master_pickup_table,
            )

        # If a CSV is uploaded, use it as an available-player filter.
        if has_valid_available_players_csv:
            pickup_table = pickup_table[
                pickup_table["normalized_player_name"].isin(available_player_names)
            ]

            st.success("Filtering rankings to players found in your uploaded CSV.")

        if pickup_table.empty:
            st.info("No ranked players matched the current uploaded CSV.")
            st.stop()

        if has_valid_roster_csv:
            st.header("Best Current Hitter Lineup")

            # Only optimize players from the uploaded current roster. Available
            # players are not used for this lineup feature.
            roster_projection_table = master_pickup_table[
                master_pickup_table["roster_status"] == "On My Roster"
            ].copy()

            if roster_projection_table.empty:
                st.info(
                    "No projected hitters matched your current roster CSV, so "
                    "a lineup could not be optimized."
                )
            else:
                lineup_df, bench_df, warning_message = optimize_hitter_lineup(
                    roster_projection_table
                )
                projected_starting_total = lineup_df[
                    "projected_fantasy_points"
                ].sum()

                if warning_message:
                    st.warning(warning_message)

                st.metric(
                    "Projected Starting Hitter Total",
                    round(projected_starting_total, 2),
                )
                st.subheader("Optimized Starting Lineup")
                lineup_display_columns = [
                    column
                    for column in [
                        "roster_slot",
                        "player",
                        "eligible_positions",
                        "projected_fantasy_points",
                        "projection_source",
                        "projection_confidence",
                    ]
                    if column in lineup_df.columns
                ]
                st.dataframe(
                    lineup_df[lineup_display_columns].round(3),
                    width="stretch",
                )

                st.subheader("Projected Bench Hitters")
                bench_display_columns = [
                    column
                    for column in [
                        "player",
                        "eligible_positions",
                        "projected_fantasy_points",
                        "projection_source",
                        "projection_confidence",
                    ]
                    if column in bench_df.columns
                ]
                st.dataframe(
                    bench_df[bench_display_columns].round(3),
                    width="stretch",
                )

        if has_valid_roster_csv and has_valid_available_players_csv:
            st.header("Best One-Move Pickup Recommendations")

            roster_projection_table = master_pickup_table[
                master_pickup_table["roster_status"] == "On My Roster"
            ].copy()
            available_projection_table = master_pickup_table[
                master_pickup_table["roster_status"] == "Available"
            ].copy()

            pickup_recommendations = evaluate_single_hitter_pickups(
                roster_projection_table,
                available_projection_table,
            )
            roster_summary = get_roster_flexibility_summary(roster_projection_table)
            multi_add_scenarios = pd.DataFrame()

            if roster_summary["open_active_spots"] > 1:
                multi_add_scenarios = evaluate_multi_add_hitter_scenarios(
                    roster_projection_table,
                    available_projection_table,
                )

            selected_min_gain = st.sidebar.slider(
                "Minimum Projected Gain",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
            )
            show_no_gain_moves = st.sidebar.checkbox(
                "Show no-gain moves",
                value=False,
            )

            if pickup_recommendations.empty and multi_add_scenarios.empty:
                st.info(
                    "No one-move pickup recommendations could be created. Check "
                    "that your roster CSV has droppable bench hitters or open "
                    "active roster spots."
                )
            else:
                best_move = get_best_positive_move(
                    pickup_recommendations,
                    multi_add_scenarios,
                )
                show_todays_action_plan(best_move)
                show_best_move_summary(best_move)
                show_lineup_impact(
                    roster_projection_table,
                    available_projection_table,
                    best_move,
                )
                show_split_pickup_recommendations(
                    pickup_recommendations,
                    selected_min_gain,
                    show_no_gain_moves,
                )

                if roster_summary["open_active_spots"] > 1:
                    show_multi_add_scenarios(
                        multi_add_scenarios,
                        show_no_gain_moves,
                    )

        # Add the tier label after projection points are calculated.
        pickup_table["tier"] = pickup_table["projected_fantasy_points"].apply(
            get_pickup_tier
        )

        bookmakers = sorted(pickup_table["bookmaker"].dropna().unique())
        selected_bookmaker = st.sidebar.selectbox(
            "Bookmaker",
            ["All Bookmakers"] + bookmakers,
        )

        max_points = float(pickup_table["projected_fantasy_points"].max())
        selected_min_points = st.sidebar.slider(
            "Minimum Projected Fantasy Points",
            min_value=0.0,
            max_value=max_points,
            value=0.0,
            step=0.5,
        )

        player_search = st.sidebar.text_input("Player Search")
        hide_rostered_players = st.sidebar.checkbox(
            "Hide players already on my roster",
            value=True,
        )

        position_options = sorted(
            {
                position
                for value in pickup_table["eligible_positions"].dropna()
                for position in split_positions(value)
            }
        )
        selected_position = st.sidebar.selectbox(
            "Position",
            ["All Positions"] + position_options,
        )

        # Apply each filter only when the user gives us a filter value.
        if selected_bookmaker != "All Bookmakers":
            pickup_table = pickup_table[pickup_table["bookmaker"] == selected_bookmaker]

        pickup_table = pickup_table[
            pickup_table["projected_fantasy_points"] >= selected_min_points
        ]

        if hide_rostered_players and has_valid_roster_csv:
            pickup_table = pickup_table[pickup_table["roster_status"] != "On My Roster"]

        if player_search:
            pickup_table = pickup_table[
                pickup_table["player"].str.contains(
                    player_search,
                    case=False,
                    na=False,
                )
            ]

        if selected_position != "All Positions":
            pickup_table = pickup_table[
                pickup_table["eligible_positions"].apply(
                    lambda value: player_has_position(value, selected_position)
                )
            ]

        # Highest projected fantasy point totals appear first.
        pickup_table = pickup_table.sort_values(
            by="projected_fantasy_points",
            ascending=False,
        ).drop(columns=["normalized_player_name"], errors="ignore")

        st.dataframe(pickup_table[TOP_PICKUPS_COLUMNS].round(3), width="stretch")
except MissingOddsAPIKeyError as error:
    st.warning(str(error))
    st.write("Create a `.env` file and add your The Odds API key to load pickups.")
except OddsAPIError as error:
    st.error(str(error))
