"""Main Streamlit entry point for the fantasy baseball analytics platform.

This file should stay focused on application-level layout: page configuration,
navigation context, and high-level messaging. Detailed business logic belongs in
the services, models, and utils folders.
"""

import streamlit as st

from config import APP_NAME
from services.odds_api import (
    MissingOddsAPIKeyError,
    OddsAPIError,
    fetch_todays_mlb_events,
    format_mlb_events_for_display,
    get_odds_api_key_debug_info,
    get_odds_data_source_status,
)
from utils.streamlit_dataframe import clean_dataframe_for_streamlit
from utils.streamlit_debug import show_odds_api_error_debug


# Configure the browser tab and default layout before rendering visible content.
st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
)


# Render the landing page for the multi-page Streamlit application.
st.title(APP_NAME)
st.write(
    "Use the pages in the sidebar to navigate between projections, pickup "
    "rankings, and settings."
)


# Keep the sidebar intentionally simple here. Streamlit automatically lists the
# files in the pages/ directory as navigable pages.
st.sidebar.header("Navigation")
st.sidebar.write("Select a page above to continue.")


# This debug panel confirms whether Streamlit can see your local Odds API key
# without ever printing the full secret.
with st.sidebar.expander("Odds API Key Debug"):
    odds_key_debug = get_odds_api_key_debug_info()
    st.write(f"Loaded: `{odds_key_debug['odds_api_key_loaded']}`")
    st.write(f"Key length: `{odds_key_debug['key_length']}`")
    st.write(
        "Key preview: "
        f"`{odds_key_debug['first_4']}...{odds_key_debug['last_4']}`"
        if odds_key_debug["odds_api_key_loaded"]
        else "Key preview: `not loaded`"
    )
    st.caption(f".env found: {odds_key_debug['env_file_exists']}")

st.sidebar.warning("Live refresh may consume API credits.")
refresh_live_odds_data = st.sidebar.button("Refresh live odds data")


# This section is intentionally limited to today's MLB game schedule. We are not
# fetching odds markets or player props yet.
st.header("Today's MLB Games")
st.caption("Game schedule from The Odds API. Player props are not fetched yet.")


try:
    # Fetch raw event data through the service layer, then format it for display.
    mlb_events = fetch_todays_mlb_events(force_refresh=refresh_live_odds_data)
    display_events = format_mlb_events_for_display(mlb_events)
    odds_source = get_odds_data_source_status().get("events", "")

    if odds_source:
        st.caption(f"Data source: {odds_source}")

    if display_events:
        st.dataframe(clean_dataframe_for_streamlit(display_events), width="stretch")
    else:
        st.info("No MLB games found for today.")
except MissingOddsAPIKeyError as error:
    # Missing keys are expected during setup, so we show a helpful warning
    # instead of crashing the app.
    st.warning(str(error))
    st.write("Create a `.env` file and add your The Odds API key to load games.")
except OddsAPIError as error:
    # API errors can happen because of network issues, invalid keys, rate limits,
    # or unexpected responses.
    st.error(str(error))
    show_odds_api_error_debug(error)
