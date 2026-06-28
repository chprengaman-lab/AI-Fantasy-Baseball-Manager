"""Streamlit page for raw MLB hitter props.

This page displays raw prop lines from The Odds API. It does not calculate
fantasy projections or recommendations yet.
"""

import pandas as pd
import streamlit as st

from services.odds_api import (
    MissingOddsAPIKeyError,
    OddsAPIError,
    fetch_todays_mlb_events,
    fetch_todays_mlb_hitter_props,
)
from utils.odds import american_odds_to_implied_probability


st.title("Raw Player Props")
st.info(
    "These are raw hitter prop lines from The Odds API for today's MLB games. "
    "No fantasy projections are calculated from these props yet."
)


try:
    # Reuse today's MLB events as the source list for event-specific prop calls.
    todays_events = fetch_todays_mlb_events()

    if not todays_events:
        st.info("No MLB games found for today, so there are no props to display.")
    else:
        prop_rows = fetch_todays_mlb_hitter_props(todays_events)

        if prop_rows:
            props_dataframe = pd.DataFrame(prop_rows)

            # Convert raw American odds into implied probabilities. These are
            # decimal values first, such as 0.40 for 40%.
            props_dataframe["over_implied_probability"] = props_dataframe[
                "over_odds"
            ].apply(american_odds_to_implied_probability)
            props_dataframe["under_implied_probability"] = props_dataframe[
                "under_odds"
            ].apply(american_odds_to_implied_probability)

            # Add simple filters so the raw table is easier to inspect.
            bookmakers = sorted(props_dataframe["bookmaker"].dropna().unique())
            markets = sorted(props_dataframe["market"].dropna().unique())

            selected_bookmaker = st.sidebar.selectbox(
                "Bookmaker",
                ["All Bookmakers"] + bookmakers,
            )
            selected_market = st.sidebar.selectbox(
                "Market",
                ["All Markets"] + markets,
            )

            # Apply filters only when the user chooses a specific value.
            if selected_bookmaker != "All Bookmakers":
                props_dataframe = props_dataframe[
                    props_dataframe["bookmaker"] == selected_bookmaker
                ]

            if selected_market != "All Markets":
                props_dataframe = props_dataframe[
                    props_dataframe["market"] == selected_market
                ]

            # Sort the table so similar markets are grouped together, then
            # players are alphabetized within each market.
            props_dataframe = props_dataframe.sort_values(
                by=["market", "player"],
                ascending=True,
            )

            # Format probabilities as percentages for display. We keep this as
            # a display-only dataframe so future math can still use decimals.
            display_dataframe = props_dataframe.copy()
            display_dataframe["over_implied_probability"] = display_dataframe[
                "over_implied_probability"
            ].map(lambda value: f"{value:.1%}" if pd.notna(value) else "")
            display_dataframe["under_implied_probability"] = display_dataframe[
                "under_implied_probability"
            ].map(lambda value: f"{value:.1%}" if pd.notna(value) else "")

            st.dataframe(display_dataframe, width="stretch")
        else:
            st.info("No hitter prop markets were returned for today's MLB games.")
except MissingOddsAPIKeyError as error:
    st.warning(str(error))
    st.write("Create a `.env` file and add your The Odds API key to load props.")
except OddsAPIError as error:
    st.error(str(error))
