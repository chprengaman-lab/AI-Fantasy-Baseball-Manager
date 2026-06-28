"""Small Streamlit helpers for safe debugging output."""

import streamlit as st


def show_odds_api_error_debug(error) -> None:
    """Display safe Odds API debugging details without exposing the API key."""

    debug_info = getattr(error, "debug_info", {})

    if not debug_info:
        return

    with st.expander("The Odds API Debug Details"):
        st.write(f"HTTP status code: `{debug_info.get('http_status_code', '')}`")
        st.write(f"Error category: `{debug_info.get('error_category', '')}`")
        st.write(f"Requested URL: `{debug_info.get('requested_url', '')}`")
        st.write("Query parameters:")
        st.json(debug_info.get("query_parameters", {}))
        st.write("Response body:")
        st.text(debug_info.get("response_body_text", ""))
