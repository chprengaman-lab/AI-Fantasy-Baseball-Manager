"""Service functions for The Odds API.

This module owns the details of talking to The Odds API. Streamlit pages should
call these functions instead of building URLs or handling request details
directly.
"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

from config import (
    APP_TIMEZONE,
    HITTER_PROP_MARKETS,
    MLB_SPORT_KEY,
    ODDS_API_BASE_URL,
    ODDS_API_KEY_ENV,
    ODDS_API_ODDS_FORMAT,
    ODDS_API_REGION,
)


class MissingOddsAPIKeyError(Exception):
    """Raised when the ODDS_API_KEY value is missing from the local environment."""


class OddsAPIError(Exception):
    """Raised when The Odds API request fails or returns invalid data."""


# This list documents the service functions and errors other files are expected
# to import from this module.
__all__ = [
    "MissingOddsAPIKeyError",
    "OddsAPIError",
    "fetch_todays_mlb_events",
    "fetch_event_hitter_props",
    "flatten_event_hitter_props",
    "fetch_todays_mlb_hitter_props",
    "format_mlb_events_for_display",
]


def _raise_for_api_error(response: requests.Response) -> None:
    """Raise a friendly error when The Odds API returns a bad status code.

    The API can fail for several normal reasons, such as an invalid key or a
    rate limit. This helper translates those HTTP statuses into messages a user
    can understand.
    """

    if response.status_code < 400:
        return

    if response.status_code in {401, 403}:
        raise OddsAPIError(
            "The Odds API key is missing, invalid, or not authorized for this request."
        )

    if response.status_code == 429:
        raise OddsAPIError(
            "The Odds API rate limit has been reached. Try again later."
        )

    raise OddsAPIError(
        f"The Odds API returned an error with status code {response.status_code}."
    )


def load_odds_api_key() -> str:
    """Load The Odds API key from the local .env file.

    load_dotenv() reads a file named .env in the project root and adds those
    values to os.environ. If the key is still missing, we raise a clear error
    that the Streamlit app can turn into a friendly message.
    """

    load_dotenv()
    api_key = os.getenv(ODDS_API_KEY_ENV)

    if not api_key:
        raise MissingOddsAPIKeyError(
            f"Missing {ODDS_API_KEY_ENV}. Add it to your .env file."
        )

    return api_key


def get_today_app_date():
    """Return today's date in the dashboard timezone."""

    app_timezone = ZoneInfo(APP_TIMEZONE)
    return datetime.now(app_timezone).date()


def is_event_today(event: dict) -> bool:
    """Return True when an API event starts today in the dashboard timezone."""

    commence_time = event.get("commence_time")

    if not commence_time:
        return False

    try:
        # The API returns times like "2026-06-26T22:41:00Z". Python's
        # fromisoformat understands "+00:00", so we replace the trailing "Z".
        event_start_utc = datetime.fromisoformat(
            commence_time.replace("Z", "+00:00")
        )
    except ValueError:
        return False

    app_timezone = ZoneInfo(APP_TIMEZONE)
    event_start_local = event_start_utc.astimezone(app_timezone)

    return event_start_local.date() == get_today_app_date()


def fetch_todays_mlb_events() -> list[dict]:
    """Fetch today's MLB games from The Odds API.

    This only fetches the event schedule. It does not fetch odds, markets,
    sportsbook lines, or player props.
    """

    api_key = load_odds_api_key()

    url = f"{ODDS_API_BASE_URL}/sports/{MLB_SPORT_KEY}/events"
    params = {"apiKey": api_key}

    try:
        response = requests.get(url, params=params, timeout=20)
    except requests.RequestException as error:
        raise OddsAPIError(
            "Unable to reach The Odds API. Check your internet connection and try again."
        ) from None

    _raise_for_api_error(response)

    try:
        events = response.json()
    except ValueError as error:
        raise OddsAPIError("The Odds API returned invalid JSON.") from None

    if not isinstance(events, list):
        raise OddsAPIError("The Odds API returned an unexpected response format.")

    # The events endpoint returns upcoming games. We filter to today locally so
    # the app displays only today's MLB schedule.
    return [event for event in events if is_event_today(event)]


def fetch_event_hitter_props(event_id: str) -> dict:
    """Fetch hitter props for one MLB event.

    The Odds API returns props one event at a time. This function only requests
    hitter markets and does not request pitcher props, game lines, or team odds.
    """

    api_key = load_odds_api_key()
    url = f"{ODDS_API_BASE_URL}/sports/{MLB_SPORT_KEY}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": ODDS_API_REGION,
        "markets": ",".join(HITTER_PROP_MARKETS),
        "oddsFormat": ODDS_API_ODDS_FORMAT,
    }

    try:
        response = requests.get(url, params=params, timeout=20)
    except requests.RequestException:
        raise OddsAPIError(
            "Unable to reach The Odds API. Check your internet connection and try again."
        ) from None

    _raise_for_api_error(response)

    try:
        event_props = response.json()
    except ValueError:
        raise OddsAPIError("The Odds API returned invalid JSON.") from None

    if not isinstance(event_props, dict):
        raise OddsAPIError("The Odds API returned an unexpected prop response format.")

    return event_props


def flatten_event_hitter_props(event_props: dict) -> list[dict]:
    """Flatten one event's hitter prop response into table-friendly rows.

    API responses are nested like this:
    event -> bookmakers -> markets -> outcomes.

    A dataframe works better with flat rows, so this function pairs Over and
    Under outcomes together for each player, bookmaker, market, and line.
    """

    event_id = event_props.get("id", "")
    flattened_rows = []

    for bookmaker in event_props.get("bookmakers", []):
        bookmaker_name = bookmaker.get("title", bookmaker.get("key", ""))

        for market in bookmaker.get("markets", []):
            market_key = market.get("key", "")

            # This dictionary temporarily groups Over and Under prices together.
            grouped_outcomes = {}

            for outcome in market.get("outcomes", []):
                player_name = outcome.get("description", "")
                line = outcome.get("point")
                outcome_name = outcome.get("name", "")
                odds_price = outcome.get("price")

                # If a market is incomplete, skip only that outcome instead of
                # failing the entire page.
                if not player_name or line is None or not outcome_name:
                    continue

                group_key = (event_id, player_name, bookmaker_name, market_key, line)

                if group_key not in grouped_outcomes:
                    grouped_outcomes[group_key] = {
                        "event_id": event_id,
                        "player": player_name,
                        "bookmaker": bookmaker_name,
                        "market": market_key,
                        "line": line,
                        "over_odds": None,
                        "under_odds": None,
                    }

                if outcome_name.lower() == "over":
                    grouped_outcomes[group_key]["over_odds"] = odds_price
                elif outcome_name.lower() == "under":
                    grouped_outcomes[group_key]["under_odds"] = odds_price

            flattened_rows.extend(grouped_outcomes.values())

    return flattened_rows


def fetch_todays_mlb_hitter_props(events: list[dict] | None = None) -> list[dict]:
    """Fetch and flatten hitter props for today's MLB games.

    If events are provided, we reuse them. If not, we fetch today's events first.
    This keeps the function useful for both Streamlit pages and future services.
    """

    todays_events = events if events is not None else fetch_todays_mlb_events()
    all_prop_rows = []

    for event in todays_events:
        event_id = event.get("id")

        # Some defensive programming: if an event has no id, skip it because the
        # event odds endpoint cannot be called without one.
        if not event_id:
            continue

        event_props = fetch_event_hitter_props(event_id)
        all_prop_rows.extend(flatten_event_hitter_props(event_props))

    return all_prop_rows


def format_mlb_events_for_display(events: list[dict]) -> list[dict]:
    """Convert raw API events into simple rows for Streamlit display."""

    formatted_events = []

    for event in events:
        formatted_events.append(
            {
                "Game ID": event.get("id", ""),
                "Away Team": event.get("away_team", ""),
                "Home Team": event.get("home_team", ""),
                "Start Time": event.get("commence_time", ""),
            }
        )

    return formatted_events
