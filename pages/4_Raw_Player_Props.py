"""Streamlit page for raw MLB hitter props.

This page displays raw prop lines from The Odds API. It does not calculate
fantasy projections or recommendations yet.
"""

import pandas as pd
import streamlit as st

from config import HITTER_PROP_MARKETS
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
from utils.streamlit_dataframe import clean_dataframe_for_streamlit
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


def get_no_prop_rows_reason(
    cache_metadata: dict,
    prop_rows: list[dict],
    should_pull_live_odds: bool,
) -> str:
    """Explain why no prop rows are available without exposing secrets."""

    cache_path = cache_metadata.get("cache_path", "")
    data_source = cache_metadata.get("data_source", "")

    if not cache_path and not should_pull_live_odds:
        return "no cache found"

    if data_source == "Sample data":
        return "sample data used"

    if cache_path and not prop_rows:
        return "cache loaded but empty"

    if should_pull_live_odds and not prop_rows:
        return "API quota/error or no events/markets available"

    return "unknown"


try:
    todays_events = []

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
    cached_rows = cached_payload.get("hitter_props", [])
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

    event_count = (
        len(todays_events)
        if should_pull_live_odds
        else len({row.get("event_id") for row in prop_rows if row.get("event_id")})
    )
    cache_file_used = cache_metadata.get("cache_path", "No cache file used")

    with st.expander("Odds Data Debug Counts", expanded=False):
        metric_columns = st.columns(4)
        metric_columns[0].metric("Events Loaded", event_count)
        metric_columns[1].metric("Raw Prop Rows Loaded", len(prop_rows))
        metric_columns[2].metric("Cached Rows Found", len(cached_rows))
        metric_columns[3].metric("Configured Markets", len(HITTER_PROP_MARKETS))
        st.write(f"Cache file used: `{cache_file_used}`")
        st.write(f"Data source: `{prop_source or 'Unknown'}`")
        st.write(f"Markets requested: `{', '.join(HITTER_PROP_MARKETS)}`")

    raw_props = prop_rows or []
    raw_props_df = clean_dataframe_for_streamlit(raw_props)
    raw_prop_count_before_filters = len(raw_props_df)
    filtered_df = raw_props_df.copy()

    if not filtered_df.empty:
        # Convert raw American odds into implied probabilities. These are
        # decimal values first, such as 0.40 for 40%.
        filtered_df["over_implied_probability"] = filtered_df[
            "over_odds"
        ].apply(american_odds_to_implied_probability)
        filtered_df["under_implied_probability"] = filtered_df[
            "under_odds"
        ].apply(american_odds_to_implied_probability)

    # Add simple filters so the raw table is easier to inspect. These widgets
    # are shown even when no rows exist so the page shape stays predictable.
    bookmakers = (
        sorted(filtered_df["bookmaker"].dropna().astype(str).unique())
        if "bookmaker" in filtered_df.columns
        else []
    )
    markets = (
        sorted(filtered_df["market"].dropna().astype(str).unique())
        if "market" in filtered_df.columns
        else []
    )

    selected_bookmaker = st.sidebar.selectbox(
        "Bookmaker",
        ["All Bookmakers"] + bookmakers,
    )
    selected_market = st.sidebar.selectbox(
        "Market",
        ["All Markets"] + markets,
    )
    player_search = st.sidebar.text_input("Player Search")

    # Apply filters only when the user chooses a specific value.
    if selected_bookmaker != "All Bookmakers" and "bookmaker" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["bookmaker"].astype(str) == selected_bookmaker]

    if selected_market != "All Markets" and "market" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["market"].astype(str) == selected_market]

    if player_search and "player" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["player"].astype(str).str.contains(
                player_search,
                case=False,
                na=False,
            )
        ]

    filtered_prop_count = len(filtered_df)

    with st.expander("Raw Props Filter Debug", expanded=False):
        st.write(f"Rows before filters: `{raw_prop_count_before_filters}`")
        st.write(f"Rows after filters: `{filtered_prop_count}`")
        st.write(f"Selected bookmaker: `{selected_bookmaker}`")
        st.write(f"Selected market: `{selected_market}`")
        st.write(f"Player search: `{player_search or 'None'}`")

        if raw_prop_count_before_filters > 0 and filtered_prop_count == 0:
            st.warning("market filters excluded all rows")

    st.header("Raw Prop Lines")

    if filtered_df.empty:
        no_rows_reason = get_no_prop_rows_reason(
            cache_metadata,
            raw_props,
            should_pull_live_odds,
        )

        if raw_prop_count_before_filters > 0:
            st.info("No raw prop rows match the current filters.")
        else:
            st.info(
                "No hitter prop markets were returned for today's MLB games. "
                f"Reason: {no_rows_reason}."
            )
    else:
        # Sort the table so similar markets are grouped together, then
        # players are alphabetized within each market.
        filtered_df = filtered_df.sort_values(
            by=["market", "player"],
            ascending=True,
        )
        display_df = filtered_df.head(500).copy()

        # Format probabilities as percentages for display. We keep this as
        # a display-only dataframe so future math can still use decimals.
        display_df["over_implied_probability"] = display_df[
            "over_implied_probability"
        ].map(lambda value: f"{value:.1%}" if pd.notna(value) else "")
        display_df["under_implied_probability"] = display_df[
            "under_implied_probability"
        ].map(lambda value: f"{value:.1%}" if pd.notna(value) else "")

        st.write(f"Rows displayed: {len(display_df)}")
        st.caption(
            f"raw_props type: `{type(raw_props).__name__}`, "
            f"raw_props length: `{len(raw_props)}`, "
            f"display_df type: `{type(display_df).__name__}`, "
            f"display_df length: `{len(display_df)}`"
        )
        st.caption(
            f"Showing first {len(display_df)} of "
            f"{len(filtered_df)} matching raw prop rows."
        )
        st.dataframe(
            clean_dataframe_for_streamlit(display_df.head(500)),
            width="stretch",
        )
except MissingOddsAPIKeyError as error:
    st.warning(str(error))
    st.write("Create a `.env` file and add your The Odds API key to load props.")
except OddsAPIError as error:
    st.error(str(error))
    show_odds_api_error_debug(error)
