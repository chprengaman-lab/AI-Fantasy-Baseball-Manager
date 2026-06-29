"""Service functions for The Odds API.

This module owns the details of talking to The Odds API. Streamlit pages should
call these functions instead of building URLs or handling request details
directly.
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
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
from utils.mlb_teams import build_teams_playing_today


class MissingOddsAPIKeyError(Exception):
    """Raised when the ODDS_API_KEY value is missing from the local environment."""


class OddsAPIError(Exception):
    """Raised when The Odds API request fails or returns invalid data."""

    def __init__(self, message: str, debug_info: dict | None = None):
        """Store a user-friendly message plus safe request/response details."""

        super().__init__(message)
        self.debug_info = debug_info or {}


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE_PATH = PROJECT_ROOT / ".env"
ODDS_CACHE_DIR = PROJECT_ROOT / "data" / "cache"
ODDS_DAILY_CACHE_DIR = ODDS_CACHE_DIR / "odds"
SAMPLE_EVENTS_PATH = PROJECT_ROOT / "data" / "sample_mlb_events.json"
SAMPLE_HITTER_PROPS_PATH = PROJECT_ROOT / "data" / "sample_hitter_props.json"
DATA_SOURCE_LIVE = "Live Odds API"
DATA_SOURCE_CACHED = "Cached Odds API"
DATA_SOURCE_TODAY_CACHE = "Today's cached odds"
DATA_SOURCE_LATEST_CACHE = "Latest cached odds"
DATA_SOURCE_SAMPLE = "Sample data"
LAST_ODDS_DATA_SOURCE = {
    "events": "",
    "hitter_props": "",
}
LAST_ODDS_METADATA = {
    "hitter_props_last_refreshed": "",
}


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
    "get_today_cache_path",
    "get_odds_api_key_debug_info",
    "get_odds_data_source_status",
    "odds_cache_exists_for_today",
    "load_cached_hitter_props",
    "save_hitter_props_cache",
    "cleanup_old_odds_cache",
    "get_mlb_teams_playing_today_from_cache",
]


def _redact_query_params(params: dict) -> dict:
    """Return query params with the API key hidden."""

    redacted_params = {}

    for key, value in params.items():
        if key.lower() == "apikey":
            redacted_params[key] = "[REDACTED]"
        else:
            redacted_params[key] = value

    return redacted_params


def _build_redacted_requested_url(url: str, params: dict) -> str:
    """Build the requested URL with apiKey redacted for safe debugging."""

    redacted_params = _redact_query_params(params)

    if not redacted_params:
        return url

    return f"{url}?{urlencode(redacted_params)}"


def _redact_api_key_from_text(text: str, params: dict) -> str:
    """Remove the actual API key from any text before showing it in the app."""

    api_key = str(params.get("apiKey", ""))

    if not api_key:
        return text

    return text.replace(api_key, "[REDACTED]")


def _classify_api_error(response: requests.Response) -> str:
    """Translate an HTTP error into a practical troubleshooting category."""

    response_text = (response.text or "").lower()

    if "out_of_usage_credits" in response_text:
        return "quota exceeded"

    if response.status_code == 401:
        return "invalid key"

    if response.status_code == 403:
        return "unauthorized market/region/bookmaker"

    if response.status_code in {402, 429}:
        return "quota exceeded"

    if response.status_code in {400, 422}:
        if any(word in response_text for word in ["market", "region", "bookmaker"]):
            return "unauthorized market/region/bookmaker"

        return "invalid request"

    return "api error"


def _build_api_error_debug_info(
    response: requests.Response,
    url: str,
    params: dict,
) -> dict:
    """Collect safe debugging details from a failed Odds API response."""

    return {
        "http_status_code": response.status_code,
        "error_category": _classify_api_error(response),
        "response_body_text": _redact_api_key_from_text(response.text, params),
        "requested_url": _build_redacted_requested_url(url, params),
        "query_parameters": _redact_query_params(params),
    }


def _today_cache_date() -> str:
    """Return today's app-local date as a cache key."""

    return get_today_app_date().isoformat()


def _safe_cache_part(value) -> str:
    """Create a short filesystem-safe cache component."""

    raw_value = str(value or "none")
    digest = hashlib.sha256(raw_value.encode("utf-8")).hexdigest()[:12]
    readable_value = "".join(
        character if character.isalnum() else "_"
        for character in raw_value
    ).strip("_")

    return f"{readable_value[:40]}_{digest}"


def _build_cache_path(
    endpoint_type: str,
    event_id: str | None = None,
    markets: list[str] | None = None,
    cache_date: str | None = None,
) -> Path:
    """Build a cache path without including secrets like the API key."""

    selected_date = cache_date or _today_cache_date()
    cache_parts = [selected_date, endpoint_type]

    if event_id:
        cache_parts.append(_safe_cache_part(event_id))

    if markets:
        cache_parts.append(_safe_cache_part(",".join(sorted(markets))))

    return ODDS_CACHE_DIR / f"{'__'.join(cache_parts)}.json"


def _read_json_file(path: Path):
    """Read JSON from disk and return None if it cannot be loaded."""

    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as cache_file:
            return json.load(cache_file)
    except (OSError, json.JSONDecodeError):
        return None


def _read_cache_file(path: Path) -> tuple[object | None, str | None]:
    """Read a cache file and return payload plus its stored source label."""

    cached_value = _read_json_file(path)

    if cached_value is None:
        return None, None

    if isinstance(cached_value, dict) and "payload" in cached_value:
        metadata = cached_value.get("_cache_metadata", {})
        return cached_value.get("payload"), metadata.get("data_source")

    # Older cache files may contain the raw response payload only.
    return cached_value, DATA_SOURCE_CACHED


def _write_cache_file(
    path: Path,
    payload,
    data_source: str = DATA_SOURCE_LIVE,
) -> None:
    """Write API payloads to local cache files without storing the API key."""

    ODDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "_cache_metadata": {
            "data_source": data_source,
            "created_at": datetime.now(ZoneInfo(APP_TIMEZONE)).isoformat(),
        },
        "payload": payload,
    }

    with path.open("w", encoding="utf-8") as cache_file:
        json.dump(cache_payload, cache_file, indent=2)


def _find_latest_cache_path(
    endpoint_type: str,
    event_id: str | None = None,
    markets: list[str] | None = None,
) -> Path | None:
    """Find the newest cache file for an endpoint when today's call fails."""

    if not ODDS_CACHE_DIR.exists():
        return None

    cache_paths = sorted(
        ODDS_CACHE_DIR.glob(f"*__{endpoint_type}*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not event_id and not markets:
        return cache_paths[0] if cache_paths else None

    expected_event_part = _safe_cache_part(event_id) if event_id else None
    expected_market_part = _safe_cache_part(",".join(sorted(markets))) if markets else None

    for path in cache_paths:
        path_name = path.name

        if expected_event_part and expected_event_part not in path_name:
            continue

        if expected_market_part and expected_market_part not in path_name:
            continue

        return path

    return None


def _load_cached_payload(
    endpoint_type: str,
    event_id: str | None = None,
    markets: list[str] | None = None,
) -> tuple[object | None, str | None, str | None]:
    """Load today's cache first, then the latest older cache if available."""

    today_cache_path = _build_cache_path(endpoint_type, event_id, markets)
    cached_payload, cached_source = _read_cache_file(today_cache_path)

    if cached_payload is not None:
        return cached_payload, str(today_cache_path), cached_source

    latest_cache_path = _find_latest_cache_path(endpoint_type, event_id, markets)

    if latest_cache_path is None:
        return None, None, None

    cached_payload, cached_source = _read_cache_file(latest_cache_path)
    return cached_payload, str(latest_cache_path), cached_source


def _load_sample_payload(endpoint_type: str):
    """Load development sample data when live Odds API data is unavailable."""

    if endpoint_type == "events":
        return _read_json_file(SAMPLE_EVENTS_PATH) or []

    if endpoint_type == "hitter_props":
        return _read_json_file(SAMPLE_HITTER_PROPS_PATH) or {}

    return None


def get_today_cache_path() -> Path:
    """Return today's combined hitter props cache path."""

    return ODDS_DAILY_CACHE_DIR / f"{_today_cache_date()}_hitter_props.json"


def _get_latest_hitter_props_cache_path() -> Path:
    """Return the latest combined hitter props cache path."""

    return ODDS_DAILY_CACHE_DIR / "latest_hitter_props.json"


def odds_cache_exists_for_today() -> bool:
    """Return True when today's combined hitter props cache exists."""

    return get_today_cache_path().exists()


def _build_hitter_props_cache_payload(
    prop_rows: list[dict],
    data_source: str,
) -> dict:
    """Build the on-disk cache payload without storing API credentials."""

    return {
        "metadata": {
            "cache_date": _today_cache_date(),
            "last_refreshed": datetime.now(ZoneInfo(APP_TIMEZONE)).isoformat(),
            "data_source": data_source,
        },
        "hitter_props": prop_rows,
    }


def _read_hitter_props_cache(path: Path) -> dict:
    """Read a combined hitter props cache file."""

    cached_payload = _read_json_file(path)

    if not isinstance(cached_payload, dict):
        return {"hitter_props": [], "metadata": {}}

    return {
        "hitter_props": cached_payload.get("hitter_props", []),
        "metadata": cached_payload.get("metadata", {}),
    }


def load_cached_hitter_props() -> dict:
    """Load today's hitter props cache, or the latest cache if today is missing."""

    today_cache_path = get_today_cache_path()

    if today_cache_path.exists():
        cached_payload = _read_hitter_props_cache(today_cache_path)
        cached_payload["metadata"]["data_source"] = DATA_SOURCE_TODAY_CACHE
        cached_payload["metadata"]["cache_path"] = str(today_cache_path)
        _set_odds_data_source("hitter_props", DATA_SOURCE_TODAY_CACHE)
        LAST_ODDS_METADATA["hitter_props_last_refreshed"] = cached_payload[
            "metadata"
        ].get("last_refreshed", "")
        return cached_payload

    latest_cache_path = _get_latest_hitter_props_cache_path()

    if latest_cache_path.exists():
        cached_payload = _read_hitter_props_cache(latest_cache_path)
        cached_payload["metadata"]["data_source"] = DATA_SOURCE_LATEST_CACHE
        cached_payload["metadata"]["cache_path"] = str(latest_cache_path)
        _set_odds_data_source("hitter_props", DATA_SOURCE_LATEST_CACHE)
        LAST_ODDS_METADATA["hitter_props_last_refreshed"] = cached_payload[
            "metadata"
        ].get("last_refreshed", "")
        return cached_payload

    return {"hitter_props": [], "metadata": {}}


def save_hitter_props_cache(
    prop_rows: list[dict],
    data_source: str = DATA_SOURCE_LIVE,
) -> dict:
    """Save combined hitter props to today's cache and latest cache."""

    ODDS_DAILY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_payload = _build_hitter_props_cache_payload(prop_rows, data_source)

    for cache_path in [get_today_cache_path(), _get_latest_hitter_props_cache_path()]:
        with cache_path.open("w", encoding="utf-8") as cache_file:
            json.dump(cache_payload, cache_file, indent=2)

    LAST_ODDS_METADATA["hitter_props_last_refreshed"] = cache_payload["metadata"][
        "last_refreshed"
    ]
    _set_odds_data_source("hitter_props", data_source)

    return cache_payload


def cleanup_old_odds_cache(max_daily_files: int = 14) -> None:
    """Keep only the newest daily hitter prop cache files."""

    if not ODDS_DAILY_CACHE_DIR.exists():
        return

    daily_cache_files = sorted(
        ODDS_DAILY_CACHE_DIR.glob("*_hitter_props.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    daily_cache_files = [
        path
        for path in daily_cache_files
        if path.name != "latest_hitter_props.json"
    ]

    for cache_path in daily_cache_files[max_daily_files:]:
        cache_path.unlink(missing_ok=True)


def _set_odds_data_source(endpoint_type: str, data_source: str) -> None:
    """Remember the data source so Streamlit can display it."""

    LAST_ODDS_DATA_SOURCE[endpoint_type] = data_source


def get_odds_data_source_status() -> dict:
    """Return the most recent Odds API data source labels."""

    status = LAST_ODDS_DATA_SOURCE.copy()
    status.update(LAST_ODDS_METADATA)
    return status


def _is_out_of_usage_credits(error: OddsAPIError) -> bool:
    """Return True when The Odds API reports exhausted usage credits."""

    debug_info = getattr(error, "debug_info", {})
    response_body = str(debug_info.get("response_body_text", "")).lower()

    return (
        debug_info.get("error_category") == "quota exceeded"
        or "out_of_usage_credits" in response_body
    )


def _raise_for_api_error(
    response: requests.Response,
    url: str,
    params: dict,
) -> None:
    """Raise a friendly error when The Odds API returns a bad status code.

    The API can fail for several normal reasons, such as an invalid key or a
    rate limit. This helper translates those HTTP statuses into messages a user
    can understand and attaches redacted debug details for Streamlit.
    """

    if response.status_code < 400:
        return

    debug_info = _build_api_error_debug_info(response, url, params)
    error_category = debug_info["error_category"]

    if error_category == "invalid key":
        raise OddsAPIError(
            "The Odds API rejected the API key. The key is loaded, but it may "
            "be invalid, expired, or copied incorrectly.",
            debug_info=debug_info,
        )

    if error_category == "quota exceeded":
        raise OddsAPIError(
            "The Odds API quota or rate limit has been reached.",
            debug_info=debug_info,
        )

    if error_category == "unauthorized market/region/bookmaker":
        raise OddsAPIError(
            "The Odds API rejected this market, region, or authorization level. "
            "Your key may not have access to one of the requested prop markets.",
            debug_info=debug_info,
        )

    raise OddsAPIError(
        f"The Odds API returned an error with status code {response.status_code}.",
        debug_info=debug_info,
    )


def load_odds_api_key() -> str:
    """Load The Odds API key from the local .env file.

    load_dotenv() must run before os.getenv() reads ODDS_API_KEY. We point it at
    this project's .env file explicitly so the key still loads if Streamlit is
    started from a different working directory.
    """

    load_dotenv(dotenv_path=ENV_FILE_PATH)
    api_key = os.getenv(ODDS_API_KEY_ENV, "")
    cleaned_api_key = api_key.strip().strip("\"'")

    if not cleaned_api_key:
        raise MissingOddsAPIKeyError(
            f"Missing {ODDS_API_KEY_ENV}. Add it to your .env file."
        )

    return cleaned_api_key


def get_odds_api_key_debug_info() -> dict:
    """Return safe, redacted ODDS_API_KEY status for Streamlit debugging."""

    # This mirrors load_odds_api_key(): load the local .env before reading the
    # environment variable, then expose only non-secret metadata.
    load_dotenv(dotenv_path=ENV_FILE_PATH)
    api_key = os.getenv(ODDS_API_KEY_ENV, "")
    cleaned_key = api_key.strip().strip("\"'")

    return {
        "env_file_path": str(ENV_FILE_PATH),
        "env_file_exists": ENV_FILE_PATH.exists(),
        "odds_api_key_loaded": bool(cleaned_key),
        "key_length": len(cleaned_key),
        "first_4": cleaned_key[:4] if cleaned_key else "",
        "last_4": cleaned_key[-4:] if cleaned_key else "",
    }


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


def fetch_todays_mlb_events(force_refresh: bool = False) -> list[dict]:
    """Fetch today's MLB games from The Odds API.

    This only fetches the event schedule. It does not fetch odds, markets,
    sportsbook lines, or player props.
    """

    endpoint_type = "events"
    cache_path = _build_cache_path(endpoint_type)

    if not force_refresh:
        cached_events, cached_source = _read_cache_file(cache_path)

        if cached_events is not None:
            _set_odds_data_source(
                endpoint_type,
                cached_source
                if cached_source == DATA_SOURCE_SAMPLE
                else DATA_SOURCE_CACHED,
            )
            return [event for event in cached_events if is_event_today(event)]

        # Do not automatically call the live API on Streamlit reruns. The UI
        # passes force_refresh=True only after the user clicks a live refresh
        # button, which protects API credits during development.
        _set_odds_data_source(endpoint_type, "")
        return []

    api_key = load_odds_api_key()

    url = f"{ODDS_API_BASE_URL}/sports/{MLB_SPORT_KEY}/events"
    params = {"apiKey": api_key}

    try:
        response = requests.get(url, params=params, timeout=20)
    except requests.RequestException as error:
        raise OddsAPIError(
            "Unable to reach The Odds API. Check your internet connection and try again."
        ) from None

    try:
        _raise_for_api_error(response, url, params)
    except OddsAPIError as error:
        cached_events, _cache_location, cached_source = _load_cached_payload(
            endpoint_type
        )

        if _is_out_of_usage_credits(error) and cached_events is not None:
            _set_odds_data_source(
                endpoint_type,
                cached_source
                if cached_source == DATA_SOURCE_SAMPLE
                else DATA_SOURCE_CACHED,
            )
            return [event for event in cached_events if is_event_today(event)]

        if _is_out_of_usage_credits(error):
            sample_events = _load_sample_payload(endpoint_type)
            _write_cache_file(cache_path, sample_events, DATA_SOURCE_SAMPLE)
            _set_odds_data_source(endpoint_type, DATA_SOURCE_SAMPLE)
            return sample_events

        raise

    try:
        events = response.json()
    except ValueError as error:
        raise OddsAPIError(
            "The Odds API returned invalid JSON.",
            debug_info={
                "http_status_code": response.status_code,
                "response_body_text": _redact_api_key_from_text(
                    response.text,
                    params,
                ),
                "requested_url": _build_redacted_requested_url(url, params),
                "query_parameters": _redact_query_params(params),
            },
        ) from None

    if not isinstance(events, list):
        raise OddsAPIError(
            "The Odds API returned an unexpected response format.",
            debug_info={
                "http_status_code": response.status_code,
                "response_body_text": _redact_api_key_from_text(
                    response.text,
                    params,
                ),
                "requested_url": _build_redacted_requested_url(url, params),
                "query_parameters": _redact_query_params(params),
            },
        )

    _write_cache_file(cache_path, events, DATA_SOURCE_LIVE)
    _set_odds_data_source(endpoint_type, DATA_SOURCE_LIVE)

    # The events endpoint returns upcoming games. We filter to today locally so
    # the app displays only today's MLB schedule.
    return [event for event in events if is_event_today(event)]


def get_mlb_teams_playing_today_from_cache() -> set[str]:
    """Return MLB team codes playing today from cached Odds API events only.

    This helper is intentionally cache-only so Streamlit page refreshes do not
    burn Odds API credits just to decide whether a player has a game today.
    """

    endpoint_type = "events"
    cache_path = _build_cache_path(endpoint_type)
    cached_events, _cached_source = _read_cache_file(cache_path)

    if cached_events is None:
        return set()

    todays_events = [
        event
        for event in cached_events
        if isinstance(event, dict) and is_event_today(event)
    ]

    return build_teams_playing_today(todays_events)


def fetch_event_hitter_props(event_id: str, force_refresh: bool = False) -> dict:
    """Fetch hitter props for one MLB event.

    The Odds API returns props one event at a time. This function only requests
    hitter markets and does not request pitcher props, game lines, or team odds.
    """

    endpoint_type = "hitter_props"
    requested_markets = HITTER_PROP_MARKETS
    cache_path = _build_cache_path(
        endpoint_type,
        event_id=event_id,
        markets=requested_markets,
    )

    if not force_refresh:
        cached_props, cached_source = _read_cache_file(cache_path)

        if cached_props is not None:
            _set_odds_data_source(
                endpoint_type,
                cached_source
                if cached_source == DATA_SOURCE_SAMPLE
                else DATA_SOURCE_CACHED,
            )
            return cached_props

    api_key = load_odds_api_key()
    url = f"{ODDS_API_BASE_URL}/sports/{MLB_SPORT_KEY}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": ODDS_API_REGION,
        "markets": ",".join(requested_markets),
        "oddsFormat": ODDS_API_ODDS_FORMAT,
    }

    try:
        response = requests.get(url, params=params, timeout=20)
    except requests.RequestException:
        raise OddsAPIError(
            "Unable to reach The Odds API. Check your internet connection and try again."
        ) from None

    try:
        _raise_for_api_error(response, url, params)
    except OddsAPIError as error:
        cached_props, _cache_location, cached_source = _load_cached_payload(
            endpoint_type,
            event_id=event_id,
            markets=requested_markets,
        )

        if _is_out_of_usage_credits(error) and cached_props is not None:
            _set_odds_data_source(
                endpoint_type,
                cached_source
                if cached_source == DATA_SOURCE_SAMPLE
                else DATA_SOURCE_CACHED,
            )
            return cached_props

        if _is_out_of_usage_credits(error):
            sample_props = _load_sample_payload(endpoint_type)
            _write_cache_file(cache_path, sample_props, DATA_SOURCE_SAMPLE)
            _set_odds_data_source(endpoint_type, DATA_SOURCE_SAMPLE)
            return sample_props

        raise

    try:
        event_props = response.json()
    except ValueError:
        raise OddsAPIError(
            "The Odds API returned invalid JSON.",
            debug_info={
                "http_status_code": response.status_code,
                "response_body_text": _redact_api_key_from_text(
                    response.text,
                    params,
                ),
                "requested_url": _build_redacted_requested_url(url, params),
                "query_parameters": _redact_query_params(params),
            },
        ) from None

    if not isinstance(event_props, dict):
        raise OddsAPIError(
            "The Odds API returned an unexpected prop response format.",
            debug_info={
                "http_status_code": response.status_code,
                "response_body_text": _redact_api_key_from_text(
                    response.text,
                    params,
                ),
                "requested_url": _build_redacted_requested_url(url, params),
                "query_parameters": _redact_query_params(params),
            },
        )

    _write_cache_file(cache_path, event_props, DATA_SOURCE_LIVE)
    _set_odds_data_source(endpoint_type, DATA_SOURCE_LIVE)

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


def fetch_todays_mlb_hitter_props(
    events: list[dict] | None = None,
    force_refresh: bool = False,
) -> list[dict]:
    """Fetch and flatten hitter props for today's MLB games.

    Development-safe behavior: by default this function reads the combined
    daily cache and does not call The Odds API. The API is called only when
    force_refresh=True, which should be tied to a user button click.
    """

    if not force_refresh:
        cached_payload = load_cached_hitter_props()
        cached_rows = cached_payload.get("hitter_props", [])
        data_source = cached_payload.get("metadata", {}).get("data_source", "")

        for prop_row in cached_rows:
            prop_row["odds_data_source"] = data_source

        return cached_rows

    todays_events = (
        events if events is not None else fetch_todays_mlb_events(force_refresh)
    )
    all_prop_rows = []

    for event in todays_events:
        event_id = event.get("id")

        # Some defensive programming: if an event has no id, skip it because the
        # event odds endpoint cannot be called without one.
        if not event_id:
            continue

        event_props = fetch_event_hitter_props(event_id, force_refresh)
        event_prop_rows = flatten_event_hitter_props(event_props)
        data_source = LAST_ODDS_DATA_SOURCE.get("hitter_props", "")

        for prop_row in event_prop_rows:
            prop_row["odds_data_source"] = data_source

        all_prop_rows.extend(event_prop_rows)

    if all_prop_rows:
        unique_sources = {
            prop_row.get("odds_data_source", "")
            for prop_row in all_prop_rows
            if prop_row.get("odds_data_source")
        }

        if len(unique_sources) > 1:
            _set_odds_data_source("hitter_props", "Mixed")
            save_hitter_props_cache(all_prop_rows, "Mixed")
        else:
            save_hitter_props_cache(
                all_prop_rows,
                LAST_ODDS_DATA_SOURCE.get("hitter_props", DATA_SOURCE_LIVE),
            )

        cleanup_old_odds_cache()
        return all_prop_rows

    # If the live pull produced no prop rows, keep the app usable with latest
    # cached odds if available. If no cache exists, use the sample development
    # data and label it clearly.
    cached_payload = load_cached_hitter_props()
    cached_rows = cached_payload.get("hitter_props", [])

    if cached_rows:
        return cached_rows

    sample_props = _load_sample_payload("hitter_props")
    sample_rows = flatten_event_hitter_props(sample_props)

    for prop_row in sample_rows:
        prop_row["odds_data_source"] = DATA_SOURCE_SAMPLE

    save_hitter_props_cache(sample_rows, DATA_SOURCE_SAMPLE)
    cleanup_old_odds_cache()

    return sample_rows


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
                "Data Source": LAST_ODDS_DATA_SOURCE.get("events", ""),
            }
        )

    return formatted_events
