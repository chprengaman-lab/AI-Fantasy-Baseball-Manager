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
    get_odds_data_source_status,
    load_cached_hitter_props,
    odds_cache_exists_for_today,
)
from utils.odds import american_odds_to_implied_probability
from utils.streamlit_debug import show_odds_api_error_debug


st.title("Raw Player Props")
st.info(
    "These are raw hitter prop lines from The Odds API for today's MLB games. "
    "No fantasy projections are calculated from these props yet."
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


try:
    if should_pull_live_odds:
        # Live pulls can cost credits because hitter props are requested one
        # event at a time. Only do this after an explicit button click.
        todays_events = fetch_todays_mlb_events(force_refresh=True)

        if not todays_events:
            st.info("No MLB games found for today, so there are no props to display.")
            prop_rows = []
        else:
            prop_rows = fetch_todays_mlb_hitter_props(
                todays_events,
                force_refresh=True,
            )
    else:
        # This reads today's cache if it exists, or latest_hitter_props.json if
        # today's file is missing. It does not call The Odds API.
        prop_rows = fetch_todays_mlb_hitter_props(force_refresh=False)

    cached_payload = load_cached_hitter_props()
    cache_metadata = cached_payload.get("metadata", {})
    odds_status = get_odds_data_source_status()
    prop_source = odds_status.get("hitter_props") or cache_metadata.get(
        "data_source",
        "",
    )
    last_refreshed = cache_metadata.get(
        "last_refreshed",
        odds_status.get("hitter_props_last_refreshed", ""),
    )

    if prop_source:
        st.caption(f"Data source: {prop_source}")

    if last_refreshed:
        st.caption(f"Last refreshed: {last_refreshed}")

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
        if not today_cache_exists and not should_pull_live_odds:
            st.info(
                "Use the button above when you are ready to spend API credits "
                "for today's hitter prop odds."
            )
        else:
            st.info("No hitter prop markets were returned for today's MLB games.")
except MissingOddsAPIKeyError as error:
    st.warning(str(error))
    st.write("Create a `.env` file and add your The Odds API key to load props.")
except OddsAPIError as error:
    st.error(str(error))
    show_odds_api_error_debug(error)
