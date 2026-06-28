"""Streamlit page for hitter fantasy point projections from raw props.

This page uses The Odds API player prop lines as a temporary projection source.
It does not build a full projection model yet.
"""

import streamlit as st

from services.odds_api import (
    MissingOddsAPIKeyError,
    OddsAPIError,
)
from services.player_projection_engine import build_player_projection_table


@st.cache_data(ttl=60 * 60)
def load_cached_player_projection_table():
    """Build and cache the unified projection table for one hour."""

    return build_player_projection_table()


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


try:
    projection_table = load_cached_player_projection_table()

    if projection_table.empty:
        st.info("No player projections are available right now.")
    else:
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
