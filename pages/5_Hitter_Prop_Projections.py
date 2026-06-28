"""Streamlit page for hitter fantasy point projections from raw props.

This page uses The Odds API player prop lines as a temporary projection source.
It does not build a full projection model yet.
"""

import streamlit as st

from services.odds_api import (
    MissingOddsAPIKeyError,
    OddsAPIError,
    load_cached_hitter_props,
    odds_cache_exists_for_today,
)
from services.player_projection_engine import build_player_projection_table
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

if should_pull_live_odds:
    load_cached_player_projection_table.clear()


try:
    projection_table = load_cached_player_projection_table(should_pull_live_odds)
    odds_data_source = projection_table.attrs.get("odds_data_source", {})
    cache_metadata = load_cached_hitter_props().get("metadata", {})
    hitter_props_source = odds_data_source.get("hitter_props") or cache_metadata.get(
        "data_source",
        "",
    )
    last_refreshed = cache_metadata.get(
        "last_refreshed",
        odds_data_source.get("hitter_props_last_refreshed", ""),
    )

    if projection_table.empty:
        if not today_cache_exists and not should_pull_live_odds:
            st.info(
                "Use the button above when you are ready to spend API credits "
                "for today's hitter prop odds."
            )

        st.info("No player projections are available right now.")
    else:
        if hitter_props_source:
            st.caption(f"Data source: {hitter_props_source}")

        if last_refreshed:
            st.caption(f"Last refreshed: {last_refreshed}")

        if not projection_table.attrs.get("player_stats_loaded"):
            st.warning(
                "Season player stats could not be loaded, so stat context columns "
                "may be blank. Prop projections are still available."
            )

        bookmakers = sorted(projection_table["bookmaker"].dropna().unique())
        selected_bookmaker = st.sidebar.selectbox(
            "Bookmaker",
            ["All Bookmakers"] + bookmakers,
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

        if selected_bookmaker != "All Bookmakers":
            projection_table = projection_table[
                projection_table["bookmaker"] == selected_bookmaker
            ]

        projection_table = projection_table[
            projection_table["projected_fantasy_points"] >= selected_min_points
        ]
        projection_table = projection_table[
            projection_table["ops"].fillna(0) >= selected_min_ops
        ]
        projection_table = projection_table.sort_values(
            by="projected_fantasy_points",
            ascending=False,
        )

        st.dataframe(projection_table.round(3), width="stretch")
except MissingOddsAPIKeyError as error:
    st.warning(str(error))
    st.write("Create a `.env` file and add your The Odds API key to load projections.")
except OddsAPIError as error:
    st.error(str(error))
    show_odds_api_error_debug(error)
