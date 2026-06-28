"""Central configuration for the fantasy baseball analytics platform.

Configuration values live here so the rest of the codebase does not scatter
hard-coded names, environment variable keys, or defaults across many files.
"""

# The app name is used by Streamlit page configuration and visible headings.
APP_NAME = "Fantasy Baseball Daily Pickup Dashboard"


# Environment variable names are defined here so service modules can load
# secrets consistently from a local .env file.
ODDS_API_KEY_ENV = "ODDS_API_KEY"
FANTASY_DATA_API_KEY_ENV = "FANTASY_DATA_API_KEY"


# The Odds API configuration. The sport key comes from The Odds API's sports
# list and represents Major League Baseball.
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
MLB_SPORT_KEY = "baseball_mlb"
ODDS_API_REGION = "us"
ODDS_API_ODDS_FORMAT = "american"
HITTER_PROP_MARKETS = [
    "batter_hits",
    "batter_home_runs",
    "batter_rbis",
    "batter_runs_scored",
    "batter_total_bases",
    "batter_stolen_bases",
]


# We use this timezone to define "today" for the dashboard before converting
# the date window to UTC for the API request.
APP_TIMEZONE = "America/New_York"


# This dictionary stores the fantasy scoring rules for this specific league.
# Keeping it in config.py makes the scoring rules easy to find and avoids
# hardcoding point values inside the scoring engine.
LEAGUE_SCORING = {
    "hitting": {
        "Single": 1,
        "Double": 2,
        "Triple": 3,
        "Home Run": 4,
        "Walk": 1,
        "Run": 1,
        "RBI": 2,
        "Stolen Base": 4,
        "Strikeout": -0.5,
        "Extra Base Hit": 1,
        "Game Winning RBI": 2,
        "Intentional Walk": 1.5,
        "Hit By Pitch": 1,
        "Sacrifice": 0.5,
        "Caught Stealing": -1,
        "Ground Into Double Play": -1,
        "Hit For The Cycle": 20,
        "Grand Slam Home Run": 8
    },
    "pitching": {},
}


# League roster and lineup rules for this ESPN points league. These settings
# will be used by the future lineup optimizer.
LEAGUE_RULES = {
    "hitter_starting_slots": ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"],
    "active_roster_spots": 17,
    "il_spots": 4,
    "max_sp": 5,
    "max_rp": 2,
    "weekly_sp_start_cap": 5,
    "rp_max_counting_innings": 2,
    "no_add_drop_limit": True,
}
