"""Experimental Streamlit page for testing ESPN Fantasy Baseball access."""

import pandas as pd
import streamlit as st

from services.espn_fantasy import (
    ESPNFantasyError,
    extract_roster_for_team,
    extract_teams_from_league_data,
    fetch_espn_league_data,
)
from utils.streamlit_dataframe import clean_dataframe_for_streamlit


def show_espn_error_debug(debug_info: dict) -> None:
    """Display safe debug details for failed ESPN responses."""

    if not debug_info:
        return

    response_kind = debug_info.get("response_kind", "Unknown")

    st.subheader("ESPN Response Debug")
    st.write(f"HTTP status code: `{debug_info.get('status_code', '')}`")
    st.write(f"Response Content-Type: `{debug_info.get('content_type', '')}`")
    st.write(f"Detected response type: `{response_kind}`")
    st.write(f"Original requested API URL: `{debug_info.get('original_requested_url', '')}`")
    st.write(f"Final response URL: `{debug_info.get('final_response_url', '')}`")
    st.write(
        "espn_s2 URL-decoded before send: "
        f"`{debug_info.get('espn_s2_decoded_before_send', False)}`"
    )

    if response_kind == "HTML" or debug_info.get("body_appears_html"):
        st.warning(
            "The ESPN response appears to be HTML. This usually indicates "
            "authentication failed or the wrong endpoint was used."
        )

    st.write("Request headers sent:")
    st.json(debug_info.get("request_headers", {}))

    with st.expander("Raw ESPN Response"):
        st.text(debug_info.get("body_preview", ""))


def clean_cookie_input(cookie_value: str) -> str:
    """Match the service cleanup for display-only validation checks."""

    if not cookie_value:
        return ""

    return cookie_value.strip().strip("\"'")


st.title("ESPN Integration Test")
st.warning(
    "ESPN Fantasy endpoints are unofficial. This test page is experimental."
)
st.write(
    "Use this page to test whether your league, team, and roster data can be "
    "retrieved from ESPN. Cookies are only used for the current Streamlit "
    "session and are not saved to `.env` or any project file."
)


# These inputs are intentionally page-local. We pass their values directly to
# the ESPN request and do not write them to disk.
input_columns = st.columns(3)
league_id = input_columns[0].text_input("ESPN league_id")
season = input_columns[1].number_input(
    "Season year",
    min_value=2000,
    max_value=2100,
    value=2026,
    step=1,
)
team_id = input_columns[2].number_input(
    "Team ID",
    min_value=1,
    value=1,
    step=1,
)

espn_s2 = st.text_input(
    "espn_s2 cookie",
    type="password",
    help="Private ESPN leagues usually require this cookie.",
)
swid = st.text_input(
    "SWID cookie",
    type="password",
    help="Private ESPN leagues usually require this cookie.",
)
decode_espn_s2 = st.checkbox(
    "URL-decode espn_s2 before sending",
    value=False,
)
cleaned_swid = clean_cookie_input(swid)
cleaned_espn_s2 = clean_cookie_input(espn_s2)

if cleaned_swid and not (cleaned_swid.startswith("{") and cleaned_swid.endswith("}")):
    st.warning('SWID should usually start with "{" and end with "}".')

if cleaned_espn_s2 and len(cleaned_espn_s2) < 50:
    st.warning("espn_s2 should be a long string. This value looks unusually short.")

st.info(
    "If this is a private league and cookies are missing or expired, ESPN may "
    "return a permission error."
)


if st.button("Test ESPN Connection"):
    try:
        raw_json = fetch_espn_league_data(
            league_id=league_id,
            season=int(season),
            espn_s2=cleaned_espn_s2 or None,
            swid=cleaned_swid or None,
            decode_espn_s2=decode_espn_s2,
        )
        league_name = raw_json.get("settings", {}).get("name", "")
        teams = extract_teams_from_league_data(raw_json)
        roster = extract_roster_for_team(raw_json, int(team_id))

        st.success("Connection succeeded. ESPN returned league JSON.")

        if league_name:
            st.metric("League Name", league_name)
        else:
            st.info("League name was not found in the returned JSON.")

        st.subheader("Teams")

        if teams:
            st.dataframe(
                clean_dataframe_for_streamlit(pd.DataFrame(teams)),
                width="stretch",
            )
        else:
            st.info("No teams table could be extracted from the ESPN response.")

        st.subheader("Selected Team Roster")

        if roster:
            st.dataframe(
                clean_dataframe_for_streamlit(pd.DataFrame(roster)),
                width="stretch",
            )
        else:
            st.info(
                "No roster could be extracted for that team_id. Check the team "
                "ID in the teams table."
            )

        with st.expander("Raw ESPN JSON Preview"):
            st.json(raw_json)
    except ESPNFantasyError as error:
        st.error(str(error))
        show_espn_error_debug(error.debug_info)
    except Exception as error:
        st.error(
            "Unexpected ESPN integration error. The response shape may have "
            "changed."
        )
        with st.expander("Debug Error"):
            st.write(repr(error))
else:
    st.info("Enter your league details, then click `Test ESPN Connection`.")
