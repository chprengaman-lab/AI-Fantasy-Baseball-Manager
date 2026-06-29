# Fantasy Baseball Daily Pickup Dashboard

This project is a production-oriented Streamlit platform for fantasy baseball analytics.

The current version can load today's MLB game schedule from The Odds API. It does not fetch sportsbook odds, player props, or live projection data yet.

## Project Structure

```text
fantasy-baseball-dashboard/
  app.py
  config.py
  requirements.txt
  README.md
  .env.example
  data/
    .gitkeep
  services/
    odds_api.py
    lineup_optimizer.py
    player_projection_engine.py
    player_stats.py
    projections.py
    fantasy_scoring.py
  models/
    player.py
  utils/
    odds.py
  pages/
    1_Player_Projections.py
    2_Top_Pickups.py
    3_Settings.py
    4_Raw_Player_Props.py
    5_Hitter_Prop_Projections.py
    6_Player_Stats.py
```

## Why Each File Exists

- `app.py`: Main Streamlit entry point. It owns top-level page setup, navigation messaging, and today's MLB games display.
- `config.py`: Central place for shared app settings, API constants, environment variable names, and league scoring rules.
- `requirements.txt`: Python dependency list generated from the current environment.
- `README.md`: Project overview, structure, and run instructions.
- `.env.example`: Safe template showing future environment variables without storing real secrets.
- `data/`: Future home for local data files, exports, or cached datasets.
- `data/.gitkeep`: Keeps the empty `data/` directory in version control until real data files exist.
- `services/odds_api.py`: Service boundary for The Odds API. It loads the API key, fetches today's MLB events, and formats games for display.
- `services/lineup_optimizer.py`: Position eligibility helpers for the future hitter lineup optimizer.
- `services/player_projection_engine.py`: Unified projection engine that combines props, implied probabilities, fantasy scoring, and season stats into one player table.
- `services/player_stats.py`: Service boundary for pybaseball hitter stats and player-name matching.
- `services/projections.py`: Shared projection logic for prop-based fantasy point estimates.
- `services/fantasy_scoring.py`: Future boundary for league scoring calculations.
- `models/player.py`: Future shared player data model.
- `utils/odds.py`: Future odds math helpers, such as implied probability conversion.
- `pages/1_Player_Projections.py`: Streamlit page placeholder for player projection views.
- `pages/2_Top_Pickups.py`: Streamlit page that ranks prop-based hitter pickup candidates.
- `pages/3_Settings.py`: Streamlit page placeholder for settings and configuration.
- `pages/4_Raw_Player_Props.py`: Streamlit page for raw MLB hitter prop lines from The Odds API.
- `pages/5_Hitter_Prop_Projections.py`: Streamlit page that estimates hitter fantasy points from raw prop lines.
- `pages/6_Player_Stats.py`: Streamlit page for searchable season-to-date hitter stats from pybaseball.

## How to Run the Project

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Create a local environment file:

```bash
cp .env.example .env
```

4. Add your The Odds API key to `.env`:

```text
ODDS_API_KEY=your_api_key_here
```

You can get an API key from The Odds API. The app currently uses it only for today's MLB event schedule.

5. Start the Streamlit app:

```bash
streamlit run app.py
```

6. Open the local URL Streamlit prints in your terminal, usually:

```text
http://localhost:8501
```

## The Odds API Scope

The app currently calls The Odds API for the MLB sport key:

```text
baseball_mlb
```

The main app fetches today's games. The Raw Player Props page fetches these hitter prop markets for today's games:

- `batter_hits`
- `batter_home_runs`
- `batter_rbis`
- `batter_runs_scored`
- `batter_total_bases`
- `batter_stolen_bases`

If `ODDS_API_KEY` is missing or the API request fails, the Streamlit app shows a friendly warning or error instead of crashing.

## League Rules

This dashboard is configured for an ESPN fantasy baseball points league with daily lineups.

Daily hitter starting slots:

- `C`
- `1B`
- `2B`
- `3B`
- `SS`
- `LF`
- `CF`
- `RF`
- `DH`

Roster settings:

- 19 active roster spots: 9 starting hitters, 4 bench hitters, 4 SP, and 2 RP
- 4 IL spots
- Max 4 SP on the active roster
- Max 2 RP on the active roster
- No add/drop limit
- SP starts are capped at 5 per week
- RP appearances are unlimited, but RP points should not count if the RP throws more than 2 innings

Important position eligibility rules:

- Positions are exact.
- `LF`, `CF`, and `RF` are separate positions.
- Outfielders are not treated as interchangeable.
- `DH` requires explicit `DH` eligibility.
- For now, position eligibility comes from uploaded CSV files. Eventually, this should use ESPN eligibility data.

## Current Status

- Production-oriented folder structure is in place.
- Streamlit multi-page placeholders are in place.
- Service, model, utility, and configuration boundaries are in place.
- Today's MLB game schedule can be loaded from The Odds API.
- Raw MLB hitter props can be loaded from The Odds API.
- Hitter fantasy point estimates can be calculated from raw prop lines.
- Season-to-date hitter stats can be loaded from pybaseball with cached results.
