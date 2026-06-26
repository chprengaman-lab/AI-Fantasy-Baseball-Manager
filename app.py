"""Fantasy Baseball Daily Pickup Dashboard.

This file is the entry point for our Streamlit app. An entry point is the
file we run to start the application.
"""

# pandas helps us create and work with table-shaped data.
import pandas as pd

# streamlit gives us simple Python functions for building a web dashboard.
import streamlit as st


# Configure the browser tab and the default page layout before drawing the app.
st.set_page_config(
    page_title="Fantasy Baseball Daily Pickup Dashboard",
    layout="wide",
)


# Display the main title at the top of the dashboard.
st.title("Fantasy Baseball Daily Pickup Dashboard")


# The sidebar is useful for filters and settings. For now, it only contains
# placeholder controls so the project has a clear place to grow later.
st.sidebar.header("Dashboard Controls")
st.sidebar.write("Filters and settings will go here.")


# This placeholder data lets us design the dashboard before connecting to APIs.
# Each dictionary below represents one row in the table.
pickup_data = [
    {
        "Player": "Example Player 1",
        "Team": "NYY",
        "Market": "Hits",
        "Line": 1.5,
        "Odds": "+120",
        "Implied Probability": "45.5%",
        "Projected Fantasy Points": 8.2,
    },
    {
        "Player": "Example Player 2",
        "Team": "LAD",
        "Market": "Total Bases",
        "Line": 1.5,
        "Odds": "-105",
        "Implied Probability": "51.2%",
        "Projected Fantasy Points": 7.6,
    },
    {
        "Player": "Example Player 3",
        "Team": "ATL",
        "Market": "RBIs",
        "Line": 0.5,
        "Odds": "+150",
        "Implied Probability": "40.0%",
        "Projected Fantasy Points": 6.9,
    },
]


# Convert the list of dictionaries into a pandas DataFrame, which Streamlit can
# display as an interactive table.
pickup_dataframe = pd.DataFrame(pickup_data)


# Add a short section label above the table so users know what they are viewing.
st.subheader("Daily Pickup Candidates")


# Render the placeholder dataframe in the dashboard.
st.dataframe(pickup_dataframe, use_container_width=True)


# Add a note to make it clear that this skeleton intentionally has no API calls.
st.info("Placeholder data only. API connections will be added in a future step.")
