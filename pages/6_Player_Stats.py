"""Streamlit page for season-to-date MLB hitter statistics.

This page loads hitter stats from pybaseball and displays them in a searchable
table. The data is cached so pybaseball is not called on every page refresh.
"""

import pandas as pd
import streamlit as st

from services.player_stats import PlayerStatsError, fetch_season_to_date_hitting_stats


@st.cache_data(ttl=60 * 60)
def load_cached_hitting_stats() -> pd.DataFrame:
    """Load hitter stats and cache them for one hour.

    Streamlit reruns this page often. Caching keeps the app from calling
    pybaseball every time a user changes a filter or refreshes the browser.
    """

    return fetch_season_to_date_hitting_stats()


st.title("Player Stats")
st.info(
    "Season-to-date hitter stats are loaded from pybaseball. These stats are "
    "separate from sportsbook props and can be used for future projection improvements."
)


# The refresh button clears the cached result, forcing the next load to call
# pybaseball again.
if st.sidebar.button("Refresh Stats"):
    load_cached_hitting_stats.clear()
    st.sidebar.success("Stats cache cleared. Reloading latest stats.")


try:
    stats_dataframe = load_cached_hitting_stats()

    player_search = st.sidebar.text_input("Player Search")

    if player_search:
        stats_dataframe = stats_dataframe[
            stats_dataframe["player"].str.contains(
                player_search,
                case=False,
                na=False,
            )
        ]

    display_columns = [
        "player",
        "batting_average",
        "obp",
        "slg",
        "ops",
        "home_runs",
        "rbi",
        "runs",
        "stolen_bases",
        "strikeout_rate",
    ]

    display_dataframe = stats_dataframe[display_columns].copy()

    # Format rate stats so the table is easier to scan.
    for column in ["batting_average", "obp", "slg", "ops"]:
        display_dataframe[column] = display_dataframe[column].map(
            lambda value: f"{value:.3f}" if pd.notna(value) else ""
        )

    display_dataframe["strikeout_rate"] = display_dataframe["strikeout_rate"].map(
        lambda value: f"{value:.1%}" if pd.notna(value) else ""
    )

    st.dataframe(display_dataframe, width="stretch")
except PlayerStatsError as error:
    st.error(str(error))
