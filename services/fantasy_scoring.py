"""Fantasy scoring calculations.

This module converts expected baseball statistics into fantasy points using the
league scoring rules from config.py. It does not create projections itself; it
only scores projection inputs that another future service will provide.
"""

from typing import Mapping

from config import LEAGUE_SCORING


# These are the hitting scoring rules for this league.
# The actual point values live in config.py, not here, so the scoring engine
# does not have hardcoded scoring numbers.
HITTING_SCORING = LEAGUE_SCORING["hitting"]


# This mapping translates future projection field names into league scoring
# category names.
#
# Example:
# - A projection engine may produce "expected_singles".
# - The league scoring dictionary uses "Single".
# - This mapping connects those two names.
#
# To support another future stat, add one row here and make sure the projection
# input includes the matching expected-stat key.
HITTING_STAT_TO_SCORING_CATEGORY = {
    "expected_singles": "Single",
    "expected_doubles": "Double",
    "expected_triples": "Triple",
    "expected_home_runs": "Home Run",
    "expected_walks": "Walk",
    "expected_runs": "Run",
    "expected_rbi": "RBI",
    "expected_stolen_bases": "Stolen Base",
    "expected_strikeouts": "Strikeout",
    "expected_extra_base_hits": "Extra Base Hit",
    "expected_game_winning_rbi": "Game Winning RBI",
    "expected_intentional_walks": "Intentional Walk",
    "expected_hit_by_pitch": "Hit By Pitch",
    "expected_sacrifices": "Sacrifice",
    "expected_caught_stealing": "Caught Stealing",
    "expected_ground_into_double_play": "Ground Into Double Play",
    "expected_hit_for_cycle": "Hit For The Cycle",
    "expected_grand_slam_home_runs": "Grand Slam Home Run",
}


def calculate_hitting_fantasy_points(expected_stats: Mapping[str, float]) -> float:
    """Calculate fantasy points from expected hitting statistics.

    Args:
        expected_stats: A dictionary-like object where each key is a projected
            stat name, such as "expected_singles", and each value is the
            expected amount of that stat.

    Returns:
        The total projected fantasy points from the provided hitting stats.

    This function is intentionally generic. It loops through the stat mapping
    above, finds each projected stat, multiplies it by the league point value,
    and adds it to the total.
    """

    # Start at zero and add each stat's fantasy point contribution.
    total_points = 0.0

    # Loop over every stat this scoring engine currently knows how to score.
    for expected_stat_name, scoring_category in HITTING_STAT_TO_SCORING_CATEGORY.items():
        # If a future projection does not include a stat yet, treat it as zero.
        expected_stat_value = expected_stats.get(expected_stat_name, 0)

        # Look up the point value from LEAGUE_SCORING["hitting"].
        points_per_stat = HITTING_SCORING[scoring_category]

        # Add this stat's contribution to the total.
        total_points += expected_stat_value * points_per_stat

    return total_points


def build_hitting_scoring_breakdown(expected_stats: Mapping[str, float]) -> dict:
    """Build a stat-by-stat scoring breakdown for future UI display.

    The dashboard will eventually need to explain why a player has a projected
    fantasy point total. Returning a breakdown now gives us a clean framework
    for that future feature without building the full projection engine today.
    """

    # Each scoring category will get its own small dictionary of details.
    breakdown = {}

    for expected_stat_name, scoring_category in HITTING_STAT_TO_SCORING_CATEGORY.items():
        expected_stat_value = expected_stats.get(expected_stat_name, 0)
        points_per_stat = HITTING_SCORING[scoring_category]

        breakdown[scoring_category] = {
            "expected_stat_name": expected_stat_name,
            "expected_value": expected_stat_value,
            "points_per_stat": points_per_stat,
            "fantasy_points": expected_stat_value * points_per_stat,
        }

    return breakdown


def calculate_fantasy_points(expected_stats: Mapping[str, float]) -> float:
    """Calculate total fantasy points for a player projection.

    For now, this only scores hitting stats because pitching scoring is still
    empty in LEAGUE_SCORING. Later, this function can combine hitting, pitching,
    baserunning, or other scoring sections into one player total.
    """

    return calculate_hitting_fantasy_points(expected_stats)
