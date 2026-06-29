"""Reusable player-name matching helpers.

Names are one of the hardest parts of sports data integration. ESPN,
sportsbooks, and stat providers may spell the same player differently. Accents,
punctuation, suffixes, and abbreviations can all break exact matching, so these
helpers create stable comparison keys for future data joins.
"""

from difflib import SequenceMatcher
import re
import unicodedata


def clean_player_name_encoding(name) -> str:
    """Fix escaped UTF-8 or mojibake player names without changing normal names.

    Some stat sources can return text like ``Heriberto Hern\\xc3\\xa1ndez`` or
    ``Heriberto HernÃ¡ndez``. The first is escaped UTF-8 and the second is
    mojibake. Both should display as ``Heriberto Hernández``.
    """

    if not isinstance(name, str):
        return ""

    cleaned_name = name.strip()

    if not cleaned_name:
        return ""

    if "\\x" in cleaned_name:
        try:
            unescaped_name = cleaned_name.encode("utf-8").decode("unicode_escape")
            cleaned_name = unescaped_name.encode("latin1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass

    if any(marker in cleaned_name for marker in ["Ã", "Â"]):
        try:
            cleaned_name = cleaned_name.encode("latin1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass

    return cleaned_name


def normalize_player_name(name) -> str:
    """Return a lowercase, punctuation-free version of a player name."""

    if not isinstance(name, str):
        return ""

    name = clean_player_name_encoding(name)

    # Convert accented characters into plain ASCII equivalents.
    normalized = unicodedata.normalize("NFKD", name)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")

    # Lowercase first so suffix handling works consistently.
    normalized = normalized.lower()

    # Replace punctuation with spaces before suffix removal. This lets "Jr."
    # and "Jr" be handled the same way.
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)

    # Suffixes can differ between providers, so remove the common ones.
    normalized = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", normalized)

    # Collapse repeated spaces created by the cleanup steps.
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def build_player_match_key(name) -> str:
    """Build a compact key for exact joins after normalizing a player name."""

    return normalize_player_name(name).replace(" ", "")


def fuzzy_match_player_name(
    name,
    candidate_names,
    min_score: int = 85,
) -> tuple[str | None, int]:
    """Return the closest candidate name when it clears the minimum score.

    The score is 0-100. This foundation will help later when sportsbook names
    and ESPN names are close but not identical.
    """

    normalized_name = normalize_player_name(name)

    if not normalized_name:
        return None, 0

    best_candidate = None
    best_score = 0

    for candidate_name in candidate_names:
        normalized_candidate = normalize_player_name(candidate_name)

        if not normalized_candidate:
            continue

        score = round(
            SequenceMatcher(None, normalized_name, normalized_candidate).ratio()
            * 100
        )

        if score > best_score:
            best_candidate = candidate_name
            best_score = score

    if best_score < min_score:
        return None, best_score

    return best_candidate, best_score
