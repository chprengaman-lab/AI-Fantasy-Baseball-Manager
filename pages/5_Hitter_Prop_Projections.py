"""Streamlit page for hitter fantasy point projections from raw props.

This page uses The Odds API player prop lines as a temporary projection source.
It does not build a full projection model yet.
"""

import streamlit as st

from config import HITTER_PROP_MARKETS
from services.odds_api import (
    MissingOddsAPIKeyError,
    OddsAPIError,
    fetch_todays_mlb_hitter_props,
    load_cached_hitter_props,
    odds_cache_exists_for_today,
)
from services.player_projection_engine import build_player_projection_table
from services.projections import build_hitter_prop_market_diagnostics
from utils.streamlit_dataframe import clean_dataframe_for_streamlit
from utils.streamlit_debug import show_odds_api_error_debug


@st.cache_data(ttl=60 * 60)
def load_cached_player_projection_table(force_refresh_odds: bool = False):
    """Build and cache the unified projection table for one hour."""

    return build_player_projection_table(force_refresh_odds=force_refresh_odds)


st.title("Hitter Prop Projections")
st.info(
    "These projections are estimated from raw sportsbook prop lines. They are "
    "not a full fantasy projection model yet."
)
st.caption(
    "Sportsbook props already reflect many daily context factors, including "
    "ballpark, weather, opponent, and lineup expectations. Season stats below "
    "are context only and do not create a separate context score."
)
st.caption(
    "Singles, doubles, and triples are estimated from projected hits, total "
    "bases, and home runs because those hit types are not available as direct "
    "prop markets."
)
st.caption(
    "Fallback projections are lower confidence because they are based on season "
    "stats and do not fully account for today's matchup, ballpark, weather, or "
    "lineup spot."
)
st.sidebar.warning("Live refresh may consume API credits.")
refresh_live_odds_data = st.sidebar.button("Refresh live odds data")
today_cache_exists = odds_cache_exists_for_today()

if not today_cache_exists:
    st.warning("No odds have been pulled for today yet.")

pull_todays_live_odds = False

if not today_cache_exists:
    pull_todays_live_odds = st.button("Pull today's live odds")

should_pull_live_odds = refresh_live_odds_data or pull_todays_live_odds
include_unavailable_players = st.sidebar.checkbox(
    "Include injured/unavailable players",
    value=False,
)
include_players_without_games = st.sidebar.checkbox(
    "Include players without games today",
    value=False,
)

if should_pull_live_odds:
    load_cached_player_projection_table.clear()


def get_projection_empty_reason(
    cache_metadata: dict,
    raw_prop_count: int,
    aggregated_projection_count: int,
    filtered_projection_count: int,
    should_pull_live_odds: bool,
) -> str:
    """Explain why projection rows are missing or hidden."""

    cache_path = cache_metadata.get("cache_path", "")
    data_source = cache_metadata.get("data_source", "")

    if raw_prop_count == 0:
        if not cache_path and not should_pull_live_odds:
            return "no cache found"

        if data_source == "Sample data":
            return "sample data used"

        if cache_path:
            return "cache loaded but empty"

        if should_pull_live_odds:
            return "API quota/error or no events/markets available"

        return "unknown no-prop-row reason"

    if aggregated_projection_count == 0:
        return "market filters excluded all rows or selected markets did not match projection markets"

    if filtered_projection_count == 0:
        return "sidebar filters excluded all projection rows"

    return ""


try:
    raw_prop_rows = fetch_todays_mlb_hitter_props(
        force_refresh=should_pull_live_odds,
    )
    projection_table = build_player_projection_table(
        force_refresh_odds=False,
        include_unavailable_players=include_unavailable_players,
        include_players_without_games=include_players_without_games,
        prop_rows=raw_prop_rows,
    )
    odds_data_source = projection_table.attrs.get("odds_data_source", {})
    cached_hitter_props = load_cached_hitter_props()
    market_diagnostics = build_hitter_prop_market_diagnostics(
        raw_prop_rows
    )
    cache_metadata = cached_hitter_props.get("metadata", {})
    hitter_props_source = odds_data_source.get("hitter_props") or cache_metadata.get(
        "data_source",
        "",
    )
    last_refreshed = cache_metadata.get(
        "last_refreshed",
        odds_data_source.get("hitter_props_last_refreshed", ""),
    )

    if projection_table.empty:
        raw_prop_count = len(raw_prop_rows)
        empty_reason = get_projection_empty_reason(
            cache_metadata,
            raw_prop_count,
            len(projection_table),
            len(projection_table),
            should_pull_live_odds,
        )
        st.info(f"No player projections are available right now. Reason: {empty_reason}.")

        if not today_cache_exists and not should_pull_live_odds:
            st.info(
                "Use the button above when you are ready to spend API credits "
                "for today's hitter prop odds."
            )
    else:
        aggregated_projection_count = len(projection_table)
        raw_prop_count = len(raw_prop_rows)
        event_count = len(
            {row.get("event_id") for row in raw_prop_rows if row.get("event_id")}
        )
        cache_file_used = cache_metadata.get("cache_path", "No cache file used")

        if hitter_props_source:
            st.caption(f"Data source: {hitter_props_source}")

        if last_refreshed:
            st.caption(f"Last refreshed: {last_refreshed}")

        if not projection_table.attrs.get("player_stats_loaded"):
            st.warning(
                "Season player stats could not be loaded, so stat context columns "
                "may be blank. Prop projections are still available."
            )

        st.caption(
            "Prop-based stats use no-vig over probabilities when both Over and "
            "Under odds are available. The expected-stat math is still an "
            "approximation because over/under lines are thresholds, not full "
            "outcome distributions."
        )

        with st.expander("Odds Projection Debug Counts", expanded=False):
            metric_columns = st.columns(4)
            metric_columns[0].metric("Events Loaded", event_count)
            metric_columns[1].metric("Raw Prop Rows Loaded", raw_prop_count)
            metric_columns[2].metric(
                "Aggregated Player Projections",
                aggregated_projection_count,
            )
            metric_columns[3].metric(
                "Market Diagnostic Rows",
                len(market_diagnostics),
            )
            st.write(f"Cache file used: `{cache_file_used}`")
            st.write(f"Data source: `{hitter_props_source or 'Unknown'}`")
            st.write(f"Markets requested: `{', '.join(HITTER_PROP_MARKETS)}`")

        if not include_unavailable_players:
            projection_table = projection_table[
                projection_table["is_available_today"].fillna(True)
            ]

        if not include_players_without_games:
            projection_table = projection_table[
                projection_table["has_game_today"].fillna(True)
            ]

        confidence_options = sorted(
            projection_table["projection_confidence"].dropna().unique()
        )
        selected_confidence = st.sidebar.selectbox(
            "Projection Confidence",
            ["All Confidence Levels"] + confidence_options,
        )

        max_bookmaker_count = int(
            projection_table["bookmaker_count"].fillna(0).max()
        )
        selected_min_bookmaker_count = st.sidebar.slider(
            "Minimum Bookmaker Count",
            min_value=0,
            max_value=max(max_bookmaker_count, 1),
            value=0,
            step=1,
        )

        max_points = float(projection_table["projected_fantasy_points"].max())
        selected_min_points = st.sidebar.slider(
            "Minimum Projected Fantasy Points",
            min_value=0.0,
            max_value=max_points,
            value=0.0,
            step=0.5,
        )

        selected_min_ops = st.sidebar.slider(
            "Minimum OPS",
            min_value=0.0,
            max_value=1.500,
            value=0.0,
            step=0.025,
        )

        if selected_confidence != "All Confidence Levels":
            projection_table = projection_table[
                projection_table["projection_confidence"] == selected_confidence
            ]

        projection_table = projection_table[
            projection_table["bookmaker_count"].fillna(0)
            >= selected_min_bookmaker_count
        ]
        projection_table = projection_table[
            projection_table["projected_fantasy_points"] >= selected_min_points
        ]
        projection_table = projection_table[
            projection_table["ops"].fillna(0) >= selected_min_ops
        ]
        filtered_projection_count = len(projection_table)

        with st.expander("Projection Filter Debug", expanded=False):
            st.write(f"Rows before filters: `{aggregated_projection_count}`")
            st.write(f"Rows after filters: `{filtered_projection_count}`")
            st.write(f"Selected confidence: `{selected_confidence}`")
            st.write(f"Minimum bookmaker count: `{selected_min_bookmaker_count}`")
            st.write(f"Minimum projected points: `{selected_min_points}`")
            st.write(f"Minimum OPS: `{selected_min_ops}`")

            empty_reason = get_projection_empty_reason(
                cache_metadata,
                raw_prop_count,
                aggregated_projection_count,
                filtered_projection_count,
                should_pull_live_odds,
            )

            if empty_reason:
                st.warning(empty_reason)

        projection_table = projection_table.sort_values(
            by="projected_fantasy_points",
            ascending=False,
        )

        display_columns = [
            column
            for column in [
                "player",
                "projected_fantasy_points",
                "has_game_today",
                "game_today_note",
                "is_available_today",
                "availability_note",
                "projected_hits",
                "projected_singles",
                "projected_doubles",
                "projected_triples",
                "projected_home_runs",
                "projected_total_bases",
                "projected_extra_base_hits",
                "projected_runs",
                "projected_rbi",
                "projected_walks",
                "projected_strikeouts",
                "projected_stolen_bases",
                "projected_caught_stealing",
                "projected_ground_into_double_play",
                "projected_game_winning_rbi",
                "projected_grand_slams",
                "missing_markets",
                "fallback_markets_used",
                "projection_source",
                "projection_confidence",
                "secondary_fallback_stats_used",
                "secondary_fallback_note",
                "bookmaker_count",
                "markets_available",
            ]
            if column in projection_table.columns
        ]

        st.dataframe(
            clean_dataframe_for_streamlit(projection_table[display_columns].round(3)),
            width="stretch",
        )

        with st.expander("Secondary Stat Fallback Diagnostic"):
            st.write(
                "Sportsbook props drive the primary daily projection when "
                "available. Walks, strikeouts, HBP, IBB, sacrifices, caught "
                "stealing, and GIDP are filled from neutral season rates when "
                "season stats exist because those categories usually do not "
                "have hitter prop markets."
            )
            season_match_counts = projection_table.attrs.get(
                "season_stats_match_counts",
                {},
            )
            if not projection_table.attrs.get("player_stats_loaded", False):
                st.warning(
                    "Season stats did not load, so secondary stat fallback can "
                    "only use zeros until pybaseball stats are available."
                )
                if projection_table.attrs.get("player_stats_error"):
                    st.caption(
                        f"Stats load error: {projection_table.attrs.get('player_stats_error')}"
                    )
            if season_match_counts:
                metric_columns = st.columns(5)
                metric_columns[0].metric(
                    "Projection Rows",
                    season_match_counts.get("total_projection_rows", 0),
                )
                metric_columns[1].metric(
                    "Season Stats Matched",
                    season_match_counts.get(
                        "projection_rows_with_season_stats_matched",
                        0,
                    ),
                )
                metric_columns[2].metric(
                    "Missing Season Stats",
                    season_match_counts.get("projection_rows_missing_season_stats", 0),
                )
                metric_columns[3].metric(
                    "Sportsbook Missing Stats",
                    season_match_counts.get(
                        "sportsbook_projected_rows_missing_season_stats",
                        0,
                    ),
                )
                metric_columns[4].metric(
                    "Fallback Missing Stats",
                    season_match_counts.get(
                        "stat_fallback_only_rows_missing_season_stats",
                        0,
                    ),
                )
            secondary_columns = [
                column
                for column in [
                    "player",
                    "normalized_player_name",
                    "player_match_key",
                    "matched_stats_player",
                    "stats_player_name_before_cleaning",
                    "stats_player_name_after_cleaning",
                    "normalized_stats_name",
                    "stats_match_method",
                    "stats_match_score",
                    "games_played",
                    "at_bats",
                    "plate_appearances",
                    "walks",
                    "strikeouts",
                    "hit_by_pitch",
                    "intentional_walks",
                    "sacrifices",
                    "stolen_bases",
                    "caught_stealing",
                    "ground_into_double_play",
                    "projected_walks",
                    "projected_strikeouts",
                    "projected_hit_by_pitch",
                    "projected_intentional_walks",
                    "projected_sacrifices",
                    "projected_caught_stealing",
                    "projected_ground_into_double_play",
                    "secondary_fallback_stats_used",
                    "secondary_fallback_note",
                    "projected_fantasy_points",
                    "projection_source",
                ]
                if column in projection_table.columns
            ]
            expected_players = [
                "Heriberto Hernandez",
                "Tyler Stephenson",
                "Estury Ruiz",
                "Shea Langeliers",
            ]
            expected_mask = projection_table["player"].isin(expected_players)
            secondary_debug_table = projection_table[
                projection_table["projection_source"]
                .fillna("")
                .astype(str)
                .str.contains("Sportsbook|fallback", case=False, na=False)
                | expected_mask
            ].copy()
            if secondary_debug_table.empty:
                st.info("No projection rows are available for secondary stat diagnostics.")
            else:
                st.dataframe(
                    clean_dataframe_for_streamlit(
                        secondary_debug_table[secondary_columns]
                        .sort_values("projected_fantasy_points", ascending=False)
                        .round(3)
                    ),
                    width="stretch",
                )

        with st.expander("Projection Method Notes"):
            st.write(
                "No-vig probability removes the sportsbook margin when both "
                "Over and Under prices are available. For example, if raw Over "
                "and Under probabilities add to more than 100%, the app divides "
                "the Over probability by the combined total."
            )
            st.write(
                "When only Over odds are available, the app applies a smaller "
                "configurable over-only discount of 0.92. This is not the same "
                "as a flat vig discount; it is only used when the Under side is "
                "missing."
            )
            st.write(
                "Home runs, hits, total bases, runs, RBI, and stolen bases use "
                "survival probabilities. Over 0.5 means P(stat >= 1), Over 1.5 "
                "means P(stat >= 2), and expected count is the sum of available "
                "threshold probabilities."
            )
            st.write(
                "Missing markets use neutral season per-game stat fallbacks "
                "when player stats are available. The app no longer applies an "
                "arbitrary 0.85 discount to season stats because they are not "
                "sportsbook odds."
            )
            st.write(
                "Rare event assumptions are still approximate: game-winning RBI "
                "uses 12% of projected RBI, grand slams use 2.25% of projected "
                "home runs, and cycles are projected as zero for now."
            )

        with st.expander("Market-Level Projection Diagnostics"):
            st.write(
                "These rows show the conservative prop-threshold estimate for "
                "each player, market, and bookmaker before player-level "
                "aggregation."
            )

            if market_diagnostics is None or market_diagnostics.empty:
                st.info("No market-level diagnostic rows are available.")
            else:
                st.dataframe(
                    clean_dataframe_for_streamlit(market_diagnostics.round(4)),
                    width="stretch",
                )
except MissingOddsAPIKeyError as error:
    st.warning(str(error))
    st.write("Create a `.env` file and add your The Odds API key to load projections.")
except OddsAPIError as error:
    st.error(str(error))
    show_odds_api_error_debug(error)
