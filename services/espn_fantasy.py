"""Experimental helpers for ESPN Fantasy Baseball data.

ESPN's fantasy endpoints are unofficial and can change without notice. This
module keeps all ESPN-specific request and parsing code in one place so the
rest of the dashboard can treat it as an optional integration.
"""

import os
import re
from urllib.parse import unquote

import requests
from dotenv import load_dotenv

from utils.name_matching import build_player_match_key, normalize_player_name


ESPN_BASE_URL = "https://fantasy.espn.com/apis/v3/games/flb"
ESPN_QUERY_VIEWS = [
    "mTeam",
    "mRoster",
    "mSettings",
    "kona_player_info",
]
ESPN_LEAGUE_ID_ENV = "ESPN_LEAGUE_ID"
ESPN_TEAM_ID_ENV = "ESPN_TEAM_ID"
ESPN_SEASON_YEAR_ENV = "ESPN_SEASON_YEAR"
ESPN_SWID_ENV = "ESPN_SWID"
ESPN_S2_ENV = "ESPN_S2"
ESPN_PLAYABLE_POSITIONS = {
    "C",
    "1B",
    "2B",
    "3B",
    "SS",
    "LF",
    "CF",
    "RF",
    "DH",
    "SP",
    "RP",
}
ESPN_NON_PLAYABLE_SLOTS = {
    "BE",
    "BENCH",
    "IL",
    "INJURED LIST",
    "UTIL",
    "IF",
    "OF",
    "P",
    "2B/SS",
    "1B/3B",
}
ESPN_SLOT_ID_TO_LABEL = {
    "0": "C",
    "1": "1B",
    "2": "2B",
    "3": "3B",
    "4": "SS",
    "5": "OF",
    "12": "UTIL",
    "13": "P",
    "14": "SP",
    "15": "RP",
    "16": "BE",
    "17": "IL",
    "20": "DH",
    "21": "LF",
    "22": "CF",
    "23": "RF",
}


class ESPNFantasyError(Exception):
    """Raised when the experimental ESPN integration cannot load data."""

    def __init__(self, message: str, debug_info: dict | None = None):
        """Store a friendly message plus optional safe debugging details."""

        super().__init__(message)
        self.debug_info = debug_info or {}


def load_espn_config_from_env() -> dict:
    """Load ESPN connection settings from a local .env file."""

    load_dotenv()

    return {
        "league_id": os.getenv(ESPN_LEAGUE_ID_ENV, ""),
        "team_id": os.getenv(ESPN_TEAM_ID_ENV, ""),
        "season": os.getenv(ESPN_SEASON_YEAR_ENV, ""),
        "swid": os.getenv(ESPN_SWID_ENV, ""),
        "espn_s2": os.getenv(ESPN_S2_ENV, ""),
    }


def connect_espn_baseball_league(
    league_id=None,
    season=None,
    espn_s2: str | None = None,
    swid: str | None = None,
):
    """Connect to an ESPN Fantasy Baseball league using espn-api.

    Explicit function arguments win. Missing values are loaded from `.env`.
    This is now the primary ESPN integration path for the app.
    """

    env_config = load_espn_config_from_env()
    selected_league_id = league_id or env_config["league_id"]
    selected_season = season or env_config["season"]
    selected_espn_s2 = espn_s2 or env_config["espn_s2"]
    selected_swid = swid or env_config["swid"]

    if not selected_league_id:
        raise ESPNFantasyError("Missing ESPN league ID.")

    if not selected_season:
        raise ESPNFantasyError("Missing ESPN season year.")

    try:
        from espn_api.baseball import League
    except ImportError:
        raise ESPNFantasyError(
            "`espn-api` is not installed. Run `pip install -r requirements.txt`."
        ) from None

    try:
        return League(
            league_id=int(selected_league_id),
            year=int(selected_season),
            espn_s2=_clean_cookie_value(selected_espn_s2) or None,
            swid=_clean_cookie_value(selected_swid) or None,
        )
    except Exception as error:
        raise ESPNFantasyError(
            "espn-api could not connect to this league. Check league ID, "
            "season, cookies, and ESPN availability."
        ) from error


def get_espn_teams(league) -> list[dict]:
    """Return normalized league team rows from an espn-api League object."""

    teams = getattr(league, "teams", []) or []

    return [
        {
            "team_id": _get_team_id(team),
            "team_name": _get_team_name_from_object(team),
            "owner": getattr(team, "owner", ""),
            "wins": getattr(team, "wins", ""),
            "losses": getattr(team, "losses", ""),
        }
        for team in teams
    ]


def get_my_espn_roster(league, team_id=None) -> list[dict]:
    """Return normalized roster rows for the selected ESPN fantasy team."""

    selected_team_id = team_id or load_espn_config_from_env()["team_id"]
    selected_team = _find_team_by_id(getattr(league, "teams", []) or [], selected_team_id)

    if selected_team is None:
        return []

    fantasy_team_name = _get_team_name_from_object(selected_team)
    roster = getattr(selected_team, "roster", []) or []

    return [
        normalize_espn_player(
            player,
            fantasy_team=fantasy_team_name,
            roster_spot=_get_player_roster_spot(player),
        )
        for player in roster
    ]


def get_espn_free_agents(
    league,
    position: str | None = None,
    size: int = 500,
) -> list[dict]:
    """Return normalized free-agent rows from espn-api."""

    try:
        # espn-api handles the ESPN request for us. We default to a larger pool
        # because the top overall free agents can include many pitchers before
        # enough streamable hitters appear.
        free_agents = league.free_agents(size=size, position=position)
    except Exception as error:
        raise ESPNFantasyError(
            "espn-api could not load free agents.",
            debug_info={"error": repr(error), "position": position, "size": size},
        ) from error

    # Free agents are not on your fantasy roster, so we label their roster spot
    # as Available before showing them in Streamlit.
    return [
        normalize_espn_player(player, roster_spot="Available")
        for player in free_agents
    ]


def get_espn_free_agent_player_objects(
    league,
    position: str | None = None,
    size: int = 100,
) -> list:
    """Return raw espn-api Player objects for temporary debugging."""

    try:
        return league.free_agents(size=size, position=position)
    except Exception as error:
        raise ESPNFantasyError(
            "espn-api could not load raw free-agent player objects.",
            debug_info={"error": repr(error), "position": position, "size": size},
        ) from error


def normalize_espn_player(
    player,
    fantasy_team: str | None = None,
    roster_spot: str = "",
) -> dict:
    """Normalize an espn-api Player object for the rest of the dashboard."""

    eligible_positions = _get_player_eligible_positions(player)
    player_name = _get_player_name(player)
    normalized_roster_spot = _normalize_espn_slot_label(
        roster_spot or _get_player_roster_spot(player)
    )

    return {
        "player": player_name,
        "normalized_player_name": normalize_player_name(player_name),
        "player_match_key": build_player_match_key(player_name),
        "espn_player_id": _get_player_id(player),
        "eligible_positions": ",".join(eligible_positions),
        "roster_spot": normalized_roster_spot,
        "player_type": _infer_player_type_from_positions(eligible_positions),
        "pro_team": _get_first_existing_attribute(
            player,
            ["proTeam", "pro_team", "team", "proTeamId"],
        ),
        "injury_status": _get_first_existing_attribute(
            player,
            ["injuryStatus", "injury_status"],
        ),
        "status": _get_first_existing_attribute(
            player,
            ["status", "playerStatus"],
        ),
        "player_notes": _get_first_existing_attribute(
            player,
            ["player_notes", "playerNotes", "news", "notes"],
        ),
        "lineup_status": _get_first_existing_attribute(
            player,
            ["lineup_status", "lineupStatus", "lineupSlot"],
        ),
        "fantasy_status": _get_first_existing_attribute(
            player,
            ["fantasy_status", "fantasyStatus", "availabilityStatus"],
        ),
        "fantasy_team": fantasy_team or "",
    }


def get_espn_player_position_debug(player) -> dict:
    """Return safe position-related fields from an espn-api Player object.

    ESPN player objects contain several fields that look like positions. Some
    are true fantasy eligibility, while others are default MLB position,
    current lineup slot, or broad ESPN lineup buckets. This debug helper makes
    those fields visible while we confirm which one matches ESPN's UI.
    """

    raw_fields = getattr(player, "__dict__", {})
    position_debug = {}
    known_position_fields = [
        "position",
        "lineupSlot",
        "eligibleSlots",
        "eligible_slots",
        "eligible_positions",
        "positions",
    ]

    for field_name in known_position_fields:
        if hasattr(player, field_name):
            position_debug[field_name] = getattr(player, field_name)

    for field_name, field_value in raw_fields.items():
        lowered_name = field_name.lower()

        if (
            "position" in lowered_name
            or "slot" in lowered_name
            or field_name in {"lineupSlot", "eligibleSlots"}
        ):
            position_debug[field_name] = field_value

    position_debug["eligibility_source_used"] = (
        "eligibleSlots exact playable entries only"
    )
    position_debug["normalized_eligible_positions"] = ",".join(
        _get_player_eligible_positions(player)
    )
    position_debug["normalized_roster_spot"] = _get_player_roster_spot(player)

    return position_debug


def build_espn_headers(
    espn_s2: str | None = None,
    swid: str | None = None,
    decode_espn_s2: bool = False,
) -> dict:
    """Build headers that look like a normal browser request.

    ESPN is picky about fantasy auth cookies. We send a plain Cookie header in
    the standard order and do not URL-encode either value.
    """

    headers = {
        "Accept": "application/json",
        "User-Agent": (
            "Mozilla/5.0 fantasy-baseball-dashboard "
            "(experimental ESPN integration)"
        ),
    }
    cookie_parts = []
    cleaned_swid = _clean_cookie_value(swid)
    cleaned_espn_s2 = _clean_cookie_value(espn_s2)

    if decode_espn_s2:
        cleaned_espn_s2 = unquote(cleaned_espn_s2)

    if cleaned_swid:
        cookie_parts.append(f"SWID={cleaned_swid}")

    if cleaned_espn_s2:
        cookie_parts.append(f"espn_s2={cleaned_espn_s2}")

    if cookie_parts:
        headers["Cookie"] = "; ".join(cookie_parts)

    return headers


def fetch_espn_league_data(
    league_id,
    season,
    espn_s2: str | None = None,
    swid: str | None = None,
    decode_espn_s2: bool = False,
) -> dict:
    """Fetch raw ESPN fantasy baseball league data.

    Private ESPN leagues usually require both espn_s2 and SWID cookies. The
    caller passes those cookies in memory only; this function does not store
    them anywhere.
    """

    if not league_id:
        raise ESPNFantasyError("Enter an ESPN league_id before testing the connection.")

    if not season:
        raise ESPNFantasyError("Enter a season year before testing the connection.")

    url = f"{ESPN_BASE_URL}/seasons/{season}/segments/0/leagues/{league_id}"
    params = [("view", view_name) for view_name in ESPN_QUERY_VIEWS]
    headers = build_espn_headers(
        espn_s2=espn_s2,
        swid=swid,
        decode_espn_s2=decode_espn_s2,
    )
    session = requests.Session()
    request = requests.Request(
        "GET",
        url,
        headers=headers,
        params=params,
    )
    prepared_request = session.prepare_request(request)
    original_requested_url = prepared_request.url
    original_request_headers = dict(prepared_request.headers)

    try:
        response = session.send(prepared_request, timeout=20)
    except requests.RequestException:
        raise ESPNFantasyError(
            "Could not reach ESPN Fantasy. Check your internet connection and try again."
        ) from None

    if response.status_code in {401, 403}:
        raise ESPNFantasyError(
            "ESPN denied access. Private leagues usually require valid espn_s2 "
            "and SWID cookies."
        )

    if response.status_code == 404:
        raise ESPNFantasyError(
            "ESPN could not find that league. Check the league_id and season."
        )

    if response.status_code >= 400:
        raise ESPNFantasyError(
            f"ESPN returned status code {response.status_code}. The unofficial "
            "endpoint may be unavailable or the request may need different cookies."
        )

    try:
        raw_json = response.json()
    except ValueError:
        raise ESPNFantasyError(
            "ESPN returned a response that was not valid JSON.",
            debug_info=_build_response_debug_info(
                response,
                original_requested_url,
                original_request_headers,
                decode_espn_s2,
            ),
        ) from None

    if not isinstance(raw_json, dict):
        raise ESPNFantasyError("ESPN returned an unexpected JSON shape.")

    return raw_json


def extract_teams_from_league_data(raw_json: dict) -> list[dict]:
    """Extract a simple table of teams from ESPN league JSON."""

    teams = raw_json.get("teams", [])

    if not isinstance(teams, list):
        return []

    team_rows = []

    for team in teams:
        if not isinstance(team, dict):
            continue

        team_rows.append(
            {
                "team_id": team.get("id"),
                "team_name": _build_team_name(team),
                "abbrev": team.get("abbrev", ""),
                "owners": ", ".join(team.get("owners", []) or []),
            }
        )

    return team_rows


def extract_roster_for_team(raw_json: dict, team_id) -> list[dict]:
    """Extract a selected team's roster from ESPN league JSON."""

    teams = raw_json.get("teams", [])

    if not isinstance(teams, list):
        return []

    selected_team = None
    selected_team_id = str(team_id)

    for team in teams:
        if str(team.get("id", "")) == selected_team_id:
            selected_team = team
            break

    if selected_team is None:
        return []

    roster_entries = selected_team.get("roster", {}).get("entries", [])

    if not isinstance(roster_entries, list):
        return []

    roster_rows = []

    for entry in roster_entries:
        player = entry.get("playerPoolEntry", {}).get("player", {})

        if not isinstance(player, dict):
            continue

        roster_rows.append(
            {
                "player_id": player.get("id"),
                "player": player.get("fullName", player.get("name", "")),
                "eligible_positions": ", ".join(extract_player_eligibility(player)),
                "lineup_slot_id": entry.get("lineupSlotId"),
                "acquisition_type": entry.get("acquisitionType", ""),
                "injury_status": player.get("injuryStatus", ""),
            }
        )

    return roster_rows


def extract_player_eligibility(player_json: dict) -> list[str]:
    """Translate ESPN eligible slot IDs into readable position labels."""

    slot_id_to_position = {
        0: "C",
        1: "1B",
        2: "2B",
        3: "3B",
        4: "SS",
        5: "OF",
        12: "UTIL",
        13: "P",
        14: "SP",
        15: "RP",
        16: "BE",
        17: "IL",
        20: "DH",
        21: "LF",
        22: "CF",
        23: "RF",
    }
    eligible_slot_ids = player_json.get("eligibleSlots", [])

    if not isinstance(eligible_slot_ids, list):
        return []

    positions = []

    for slot_id in eligible_slot_ids:
        position = slot_id_to_position.get(slot_id)

        if position:
            positions.append(position)

    return positions


def _build_team_name(team: dict) -> str:
    """Build a readable team name from ESPN's team fields."""

    location = team.get("location", "")
    nickname = team.get("nickname", "")

    if location and nickname:
        return f"{location} {nickname}"

    return team.get("name", nickname or location)


def _build_response_debug_info(
    response: requests.Response,
    original_requested_url: str,
    original_request_headers: dict,
    decode_espn_s2: bool,
) -> dict:
    """Build safe debugging details for non-JSON ESPN responses."""

    response_body = response.text or ""

    return {
        "status_code": response.status_code,
        "content_type": response.headers.get("Content-Type", ""),
        "response_kind": _classify_response(response),
        "body_appears_html": _body_appears_html(response),
        "original_requested_url": original_requested_url,
        "final_response_url": response.url,
        "request_headers": _redact_request_headers(original_request_headers),
        "espn_s2_decoded_before_send": decode_espn_s2,
        "body_preview": response_body[:1000],
    }


def _clean_cookie_value(cookie_value: str | None) -> str:
    """Strip whitespace and accidental wrapping quotes from a cookie value."""

    if not cookie_value:
        return ""

    return cookie_value.strip().strip("\"'")


def _classify_response(response: requests.Response) -> str:
    """Classify a failed response so the user knows what likely happened."""

    response_body = (response.text or "").lstrip()
    content_type = response.headers.get("Content-Type", "").lower()

    if response.history or 300 <= response.status_code < 400:
        return "Redirect"

    if not response_body:
        return "Empty"

    if _body_appears_html(response):
        return "HTML"

    if "json" in content_type or response_body.startswith(("{", "[")):
        return "JSON"

    return "Unknown"


def _body_appears_html(response: requests.Response) -> bool:
    """Return True when the response content looks like HTML."""

    response_body = (response.text or "").lstrip().lower()
    content_type = response.headers.get("Content-Type", "").lower()

    return "html" in content_type or response_body.startswith(
        ("<!doctype html", "<html")
    )


def _redact_request_headers(headers) -> dict:
    """Return request headers with ESPN cookie values safely redacted."""

    redacted_headers = dict(headers)
    cookie_header = redacted_headers.get("Cookie")

    if cookie_header:
        redacted_headers["Cookie"] = _redact_cookie_header(cookie_header)

    return redacted_headers


def _redact_cookie_header(cookie_header: str) -> str:
    """Redact espn_s2 and SWID values inside a Cookie header."""

    redacted_parts = []

    for cookie_part in cookie_header.split(";"):
        cookie_part = cookie_part.strip()

        if "=" not in cookie_part:
            redacted_parts.append(cookie_part)
            continue

        cookie_name, cookie_value = cookie_part.split("=", 1)

        if cookie_name in {"espn_s2", "SWID"}:
            cookie_value = _redact_secret(cookie_value)

        redacted_parts.append(f"{cookie_name}={cookie_value}")

    return "; ".join(redacted_parts)


def _redact_secret(secret_value: str) -> str:
    """Show only the first and last five characters of a secret value."""

    if len(secret_value) <= 10:
        return "*" * len(secret_value)

    return f"{secret_value[:5]}...{secret_value[-5:]}"


def _get_first_existing_attribute(source_object, attribute_names: list[str]):
    """Read the first non-empty attribute from an espn-api object."""

    for attribute_name in attribute_names:
        attribute_value = getattr(source_object, attribute_name, "")

        if attribute_value:
            return attribute_value

    return ""


def _get_team_id(team) -> int | str:
    """Read a team id from whichever attribute espn-api exposes."""

    return _get_first_existing_attribute(team, ["team_id", "teamId", "id"])


def _get_team_name_from_object(team) -> str:
    """Read a readable team name from an espn-api Team object."""

    return _get_first_existing_attribute(
        team,
        ["team_name", "teamName", "name"],
    ) or f"Team {_get_team_id(team)}"


def _find_team_by_id(teams: list, team_id):
    """Find an espn-api Team object by team id."""

    selected_team_id = str(team_id)

    for team in teams:
        if str(_get_team_id(team)) == selected_team_id:
            return team

    return None


def _get_player_name(player) -> str:
    """Read a readable player name from an espn-api Player object."""

    return _get_first_existing_attribute(
        player,
        ["name", "playerName", "fullName"],
    )


def _get_player_id(player):
    """Read ESPN player id from whichever attribute espn-api exposes."""

    return _get_first_existing_attribute(player, ["playerId", "player_id", "id"])


def _get_player_roster_spot(player) -> str:
    """Read roster spot or lineup slot from an espn-api Player object."""

    roster_spot = str(
        _get_first_existing_attribute(
            player,
            ["lineupSlot", "lineup_slot", "slot_position"],
        )
    )

    return _normalize_espn_slot_label(roster_spot)


def _get_player_eligible_positions(player) -> list[str]:
    """Read true playable position eligibility from an espn-api Player object.

    ESPN exposes both position eligibility and lineup/roster slots. They are
    not the same thing. For example, BE means bench, IL means injured list, UTIL
    means a lineup slot, and IF/OF/2B-SS/1B-3B are broad lineup buckets. We
    exclude those from eligible_positions because this league requires exact
    positions for optimizer decisions.

    The espn-api field we use is eligibleSlots. We intentionally do not fall
    back to position/default position because that is a player's primary MLB
    position, not the full ESPN fantasy eligibility shown in the app.
    """

    raw_positions = []

    for attribute_name in ["eligibleSlots", "eligible_slots"]:
        positions = getattr(player, attribute_name, None)

        if not positions:
            continue

        if isinstance(positions, list):
            raw_positions = [str(position) for position in positions]
            break

        raw_positions = [str(positions)]
        break

    return _normalize_eligible_positions(raw_positions)


def _normalize_eligible_positions(raw_positions: list[str]) -> list[str]:
    """Split and clean ESPN position values into exact position tokens.

    ESPN or espn-api may return combined strings like "1B/3B". The dashboard
    needs those stored as "1B,3B" so filters and exact eligibility checks do not
    accidentally treat "1B/3B" as a fake position.
    """

    normalized_positions = []

    for raw_position in raw_positions:
        for position in re.split(r",", str(raw_position)):
            cleaned_position = _normalize_espn_slot_label(position)

            # Only keep true playable positions. Do not infer 1B from IF, do
            # not infer LF/CF/RF from OF, do not infer anything from UTIL, and
            # do not expand ESPN combo lineup slots like 2B/SS or 1B/3B.
            if (
                cleaned_position in ESPN_NON_PLAYABLE_SLOTS
                or cleaned_position not in ESPN_PLAYABLE_POSITIONS
            ):
                continue

            if cleaned_position not in normalized_positions:
                normalized_positions.append(cleaned_position)

    return normalized_positions


def _normalize_espn_slot_label(value) -> str:
    """Convert ESPN slot ids or labels into a readable uppercase label."""

    if value is None:
        return ""

    raw_value = str(value).strip()

    if raw_value.lower() == "available":
        return "Available"

    cleaned_value = raw_value.upper()

    if cleaned_value in ESPN_SLOT_ID_TO_LABEL:
        return ESPN_SLOT_ID_TO_LABEL[cleaned_value]

    if cleaned_value == "BENCH":
        return "BE"

    if cleaned_value == "INJURED LIST":
        return "IL"

    return cleaned_value


def _infer_player_type_from_positions(eligible_positions: list[str]) -> str:
    """Infer broad player type for dashboard roster rules."""

    normalized_positions = {position.upper() for position in eligible_positions}

    if "SP" in normalized_positions:
        return "SP"

    if "RP" in normalized_positions:
        return "RP"

    return "Hitter"
