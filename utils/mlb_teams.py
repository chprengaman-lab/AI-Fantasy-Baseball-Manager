"""Helpers for matching MLB team names from different data sources.

The Odds API usually returns full names like "Baltimore Orioles". ESPN may
return an abbreviation like "BAL" or sometimes another string. We normalize both
formats to the same three-letter team code before comparing schedules.
"""


MLB_TEAM_ALIASES = {
    "ARI": {"ARI", "ARIZONA", "ARIZONA DIAMONDBACKS", "DIAMONDBACKS"},
    "ATH": {"ATH", "OAK", "OAKLAND", "ATHLETICS", "OAKLAND ATHLETICS"},
    "ATL": {"ATL", "ATLANTA", "ATLANTA BRAVES", "BRAVES"},
    "BAL": {"BAL", "BALTIMORE", "BALTIMORE ORIOLES", "ORIOLES"},
    "BOS": {"BOS", "BOSTON", "BOSTON RED SOX", "RED SOX"},
    "CHC": {"CHC", "CHICAGO CUBS", "CUBS"},
    "CIN": {"CIN", "CINCINNATI", "CINCINNATI REDS", "REDS"},
    "CLE": {"CLE", "CLEVELAND", "CLEVELAND GUARDIANS", "GUARDIANS"},
    "COL": {"COL", "COLORADO", "COLORADO ROCKIES", "ROCKIES"},
    "CWS": {
        "CWS",
        "CHW",
        "CHICAGO WHITE SOX",
        "WHITE SOX",
        "CHI WHITE SOX",
    },
    "DET": {"DET", "DETROIT", "DETROIT TIGERS", "TIGERS"},
    "HOU": {"HOU", "HOUSTON", "HOUSTON ASTROS", "ASTROS"},
    "KC": {"KC", "KCR", "KANSAS CITY", "KANSAS CITY ROYALS", "ROYALS"},
    "LAA": {
        "LAA",
        "ANA",
        "LOS ANGELES ANGELS",
        "LA ANGELS",
        "ANGELS",
    },
    "LAD": {
        "LAD",
        "LA DODGERS",
        "LOS ANGELES DODGERS",
        "DODGERS",
    },
    "MIA": {"MIA", "MIAMI", "MIAMI MARLINS", "MARLINS"},
    "MIL": {"MIL", "MILWAUKEE", "MILWAUKEE BREWERS", "BREWERS"},
    "MIN": {"MIN", "MINNESOTA", "MINNESOTA TWINS", "TWINS"},
    "NYM": {"NYM", "NEW YORK METS", "METS"},
    "NYY": {"NYY", "NEW YORK YANKEES", "YANKEES"},
    "PHI": {"PHI", "PHILADELPHIA", "PHILADELPHIA PHILLIES", "PHILLIES"},
    "PIT": {"PIT", "PITTSBURGH", "PITTSBURGH PIRATES", "PIRATES"},
    "SD": {"SD", "SDP", "SAN DIEGO", "SAN DIEGO PADRES", "PADRES"},
    "SEA": {"SEA", "SEATTLE", "SEATTLE MARINERS", "MARINERS"},
    "SF": {"SF", "SFG", "SAN FRANCISCO", "SAN FRANCISCO GIANTS", "GIANTS"},
    "STL": {"STL", "ST. LOUIS", "SAINT LOUIS", "ST. LOUIS CARDINALS", "CARDINALS"},
    "TB": {"TB", "TBR", "TAMPA BAY", "TAMPA BAY RAYS", "RAYS"},
    "TEX": {"TEX", "TEXAS", "TEXAS RANGERS", "RANGERS"},
    "TOR": {"TOR", "TORONTO", "TORONTO BLUE JAYS", "BLUE JAYS"},
    "WSH": {"WSH", "WAS", "WASHINGTON", "WASHINGTON NATIONALS", "NATIONALS"},
}

_ALIAS_TO_TEAM_CODE = {
    alias: team_code
    for team_code, aliases in MLB_TEAM_ALIASES.items()
    for alias in aliases
}


def normalize_mlb_team(value) -> str:
    """Return a three-letter MLB team code when the team can be identified."""

    if value is None:
        return ""

    cleaned_value = str(value).strip().upper()

    if not cleaned_value:
        return ""

    return _ALIAS_TO_TEAM_CODE.get(cleaned_value, "")


def build_teams_playing_today(events: list[dict]) -> set[str]:
    """Build normalized team codes from today's Odds API event list."""

    teams = set()

    for event in events or []:
        for team_column in ["home_team", "away_team"]:
            team_code = normalize_mlb_team(event.get(team_column, ""))

            if team_code:
                teams.add(team_code)

    return teams
