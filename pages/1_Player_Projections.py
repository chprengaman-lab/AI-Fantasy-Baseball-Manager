"""Streamlit page for player projections.

This page shows sample hitting projections and uses the fantasy scoring service
to calculate projected fantasy points. The sample data is temporary; real
projections and sportsbook data will be added later.
"""

import pandas as pd
import streamlit as st

from services.fantasy_scoring import calculate_fantasy_points


st.title("Player Projections")


# Explain the current scope clearly for anyone viewing the page.
st.info(
    "These are sample projections only. They are not real sportsbook data, "
    "not betting advice, and not connected to live APIs yet."
)


# Each dictionary below represents one player's sample projected hitting line.
# The expected_* fields match the names supported by the fantasy scoring engine.
sample_player_projections = [
    {
        "Player": "Aaron Judge",
        "Team": "NYY",
        "expected_singles": 0.55,
        "expected_doubles": 0.22,
        "expected_triples": 0.01,
        "expected_home_runs": 0.32,
        "expected_walks": 0.48,
        "expected_runs": 0.78,
        "expected_rbi": 0.86,
        "expected_stolen_bases": 0.03,
        "expected_strikeouts": 1.35,
    },
    {
        "Player": "Shohei Ohtani",
        "Team": "LAD",
        "expected_singles": 0.62,
        "expected_doubles": 0.25,
        "expected_triples": 0.03,
        "expected_home_runs": 0.30,
        "expected_walks": 0.44,
        "expected_runs": 0.84,
        "expected_rbi": 0.78,
        "expected_stolen_bases": 0.18,
        "expected_strikeouts": 1.12,
    },
    {
        "Player": "Mookie Betts",
        "Team": "LAD",
        "expected_singles": 0.72,
        "expected_doubles": 0.28,
        "expected_triples": 0.02,
        "expected_home_runs": 0.19,
        "expected_walks": 0.40,
        "expected_runs": 0.88,
        "expected_rbi": 0.63,
        "expected_stolen_bases": 0.12,
        "expected_strikeouts": 0.72,
    },
    {
        "Player": "Juan Soto",
        "Team": "NYY",
        "expected_singles": 0.68,
        "expected_doubles": 0.24,
        "expected_triples": 0.01,
        "expected_home_runs": 0.24,
        "expected_walks": 0.70,
        "expected_runs": 0.82,
        "expected_rbi": 0.72,
        "expected_stolen_bases": 0.05,
        "expected_strikeouts": 0.88,
    },
    {
        "Player": "Ronald Acuna Jr.",
        "Team": "ATL",
        "expected_singles": 0.74,
        "expected_doubles": 0.22,
        "expected_triples": 0.03,
        "expected_home_runs": 0.22,
        "expected_walks": 0.38,
        "expected_runs": 0.86,
        "expected_rbi": 0.65,
        "expected_stolen_bases": 0.35,
        "expected_strikeouts": 0.96,
    },
    {
        "Player": "Jose Ramirez",
        "Team": "CLE",
        "expected_singles": 0.70,
        "expected_doubles": 0.26,
        "expected_triples": 0.02,
        "expected_home_runs": 0.20,
        "expected_walks": 0.32,
        "expected_runs": 0.72,
        "expected_rbi": 0.76,
        "expected_stolen_bases": 0.18,
        "expected_strikeouts": 0.58,
    },
    {
        "Player": "Bobby Witt Jr.",
        "Team": "KC",
        "expected_singles": 0.76,
        "expected_doubles": 0.27,
        "expected_triples": 0.04,
        "expected_home_runs": 0.21,
        "expected_walks": 0.28,
        "expected_runs": 0.80,
        "expected_rbi": 0.70,
        "expected_stolen_bases": 0.30,
        "expected_strikeouts": 0.92,
    },
    {
        "Player": "Yordan Alvarez",
        "Team": "HOU",
        "expected_singles": 0.64,
        "expected_doubles": 0.25,
        "expected_triples": 0.01,
        "expected_home_runs": 0.28,
        "expected_walks": 0.46,
        "expected_runs": 0.70,
        "expected_rbi": 0.82,
        "expected_stolen_bases": 0.01,
        "expected_strikeouts": 0.94,
    },
]


# Convert the list of player dictionaries into a pandas DataFrame so we can
# calculate columns, sort rows, filter teams, and display a table.
projections_dataframe = pd.DataFrame(sample_player_projections)


# Calculate projected fantasy points for every player. axis=1 means pandas
# sends one row at a time into our scoring function.
projections_dataframe["Projected Fantasy Points"] = projections_dataframe.apply(
    calculate_fantasy_points,
    axis=1,
)


# Sort the best projected fantasy point totals to the top of the table.
projections_dataframe = projections_dataframe.sort_values(
    by="Projected Fantasy Points",
    ascending=False,
)


# Build a team filter from the teams in the sample data.
available_teams = sorted(projections_dataframe["Team"].unique())
selected_team = st.sidebar.selectbox("Team", ["All Teams"] + available_teams)


# Apply the team filter only when the user chooses a specific team.
if selected_team != "All Teams":
    projections_dataframe = projections_dataframe[
        projections_dataframe["Team"] == selected_team
    ]


# Round the numeric columns so the sample table is easier to read.
display_dataframe = projections_dataframe.round(2)


# Show the projection table in the Streamlit page.
st.dataframe(display_dataframe, width="stretch")
