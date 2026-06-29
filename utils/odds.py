"""Utility functions for odds-related calculations.

Utilities are small reusable helpers that do not own external API calls or app
state.
"""


def american_odds_to_implied_probability(american_odds: int | float | None) -> float | None:
    """Convert American odds into implied probability.

    Args:
        american_odds: The American odds value, such as +150 or -120.

    Returns:
        A decimal probability between 0 and 1. For example, +150 becomes 0.40.
        If the odds value is missing, return None so the UI can show a blank.

    American odds use two formulas:
    - Positive odds: 100 / (odds + 100)
    - Negative odds: abs(odds) / (abs(odds) + 100)
    """

    if american_odds is None:
        return None

    try:
        american_odds = float(american_odds)
    except (TypeError, ValueError):
        return None

    if american_odds > 0:
        return 100 / (american_odds + 100)

    if american_odds < 0:
        absolute_odds = abs(american_odds)
        return absolute_odds / (absolute_odds + 100)

    # American odds should not be zero, so treat zero as invalid/missing.
    return None
