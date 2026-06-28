"""Helpers for a future hitter lineup optimizer.

This module does not build the full optimizer yet. It only defines the position
eligibility rules the optimizer will need.
"""

from itertools import combinations
import re

import pandas as pd

from config import LEAGUE_RULES
from services.player_stats import normalize_player_name


def normalize_position(position) -> str:
    """Normalize one position value into the format used by league rules."""

    if not isinstance(position, str):
        return ""

    return position.strip().upper()


def parse_eligible_positions(value) -> list[str]:
    """Parse an eligible_positions value into a list of exact positions.

    Uploaded CSVs may use commas or slashes, such as "1B,3B" or "2B/SS".
    We split on both separators and normalize each position.
    """

    if not isinstance(value, str) or not value.strip():
        return []

    positions = re.split(r"[,/]", value)

    return [
        normalize_position(position)
        for position in positions
        if normalize_position(position)
    ]


def can_play_position(eligible_positions, roster_slot) -> bool:
    """Return True when a player can fill a specific roster slot.

    LF, CF, and RF must be treated as exact positions in this league. A player
    with LF eligibility should not automatically count as CF or RF.

    DH must also be exact. Do not assume a hitter can fill DH unless DH appears
    in the uploaded eligible_positions value.
    """

    normalized_roster_slot = normalize_position(roster_slot)
    parsed_positions = parse_eligible_positions(eligible_positions)

    return normalized_roster_slot in parsed_positions


def get_hitter_starting_slots() -> list[str]:
    """Return the daily hitter starting slots for this league."""

    return LEAGUE_RULES["hitter_starting_slots"]


def validate_hitter_eligibility(player_row, roster_slot) -> bool:
    """Validate whether a player row can be assigned to a hitter roster slot.

    Args:
        player_row: A pandas Series or dictionary containing eligible_positions.
        roster_slot: The exact lineup slot we want to test, such as "SS" or "RF".
    """

    if isinstance(player_row, pd.Series):
        eligible_positions = player_row.get("eligible_positions", "")
    elif isinstance(player_row, dict):
        eligible_positions = player_row.get("eligible_positions", "")
    else:
        eligible_positions = ""

    return can_play_position(eligible_positions, roster_slot)


def _empty_lineup_result(players_df: pd.DataFrame, warning_message: str):
    """Return an empty lineup result with all players on the bench."""

    lineup_df = pd.DataFrame(
        columns=[
            "roster_slot",
            "player",
            "eligible_positions",
            "projected_fantasy_points",
            "projection_source",
            "projection_confidence",
        ]
    )
    bench_df = players_df.copy()

    return lineup_df, bench_df, warning_message


def _is_truthy(value) -> bool:
    """Convert common CSV truthy values into a boolean."""

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    if isinstance(value, (int, float)):
        return value == 1

    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1"}

    return False


def _is_il_roster_spot(roster_spot) -> bool:
    """Return True when a roster_spot value means injured list."""

    if not isinstance(roster_spot, str):
        return False

    return normalize_position(roster_spot) == "IL"


def is_il_player(row) -> bool:
    """Return True when a roster row is on an IL roster spot."""

    return _is_il_roster_spot(row.get("roster_spot", ""))


def _is_bench_roster_spot(roster_spot) -> bool:
    """Return True when a roster_spot value means bench."""

    if not isinstance(roster_spot, str):
        return False

    return normalize_position(roster_spot) == "BENCH"


def infer_player_type(row) -> str:
    """Infer player type from CSV data.

    If player_type exists, use it. If it is missing, infer from
    eligible_positions:
    - SP means starting pitcher
    - RP means relief pitcher
    - otherwise treat the player as a hitter
    """

    explicit_player_type = row.get("player_type", "")

    if isinstance(explicit_player_type, str) and explicit_player_type.strip():
        return normalize_position(explicit_player_type)

    eligible_positions = parse_eligible_positions(row.get("eligible_positions", ""))

    if "SP" in eligible_positions:
        return "SP"

    if "RP" in eligible_positions:
        return "RP"

    return "HITTER"


def count_active_roster_spots(roster_df: pd.DataFrame) -> int:
    """Count active roster spots, excluding IL players."""

    if roster_df is None or roster_df.empty:
        return 0

    return int((~roster_df.apply(is_il_player, axis=1)).sum())


def count_il_players(roster_df: pd.DataFrame) -> int:
    """Count players stored on IL roster spots."""

    if roster_df is None or roster_df.empty:
        return 0

    return int(roster_df.apply(is_il_player, axis=1).sum())


def count_player_types(roster_df: pd.DataFrame) -> dict:
    """Count active hitters, SP, and RP from the roster CSV.

    IL players do not count toward the active roster, so they are excluded from
    these type counts. If player_type is missing, infer_player_type uses exact
    eligible_positions to identify SP and RP.
    """

    player_type_counts = {
        "HITTER": 0,
        "SP": 0,
        "RP": 0,
    }

    if roster_df is None or roster_df.empty:
        return player_type_counts

    active_roster_df = roster_df[~roster_df.apply(is_il_player, axis=1)]

    for _, player_row in active_roster_df.iterrows():
        player_type = infer_player_type(player_row)

        if player_type in player_type_counts:
            player_type_counts[player_type] += 1

    return player_type_counts


def get_roster_flexibility_summary(roster_df: pd.DataFrame) -> dict:
    """Summarize active roster space and pitcher counts for pickup decisions."""

    active_roster_limit = LEAGUE_RULES["active_roster_spots"]
    il_limit = LEAGUE_RULES["il_spots"]
    sp_limit = LEAGUE_RULES["max_sp"]
    rp_limit = LEAGUE_RULES["max_rp"]
    active_roster_count = count_active_roster_spots(roster_df)
    il_count = count_il_players(roster_df)
    player_type_counts = count_player_types(roster_df)
    sp_count = player_type_counts["SP"]
    rp_count = player_type_counts["RP"]
    open_active_spots = max(active_roster_limit - active_roster_count, 0)
    flexibility_notes = []

    if open_active_spots > 0:
        flexibility_notes.append(
            "You have open active roster space, so the app can recommend "
            "add-only hitter pickups."
        )
    else:
        flexibility_notes.append(
            "Your active roster is full, so adding a hitter requires dropping "
            "someone."
        )

    if sp_count < sp_limit:
        flexibility_notes.append(
            "You are carrying fewer than 5 SP, which may create hitter "
            "streaming flexibility."
        )

    if rp_count >= rp_limit:
        flexibility_notes.append("You are at the 2 RP maximum.")

    return {
        "active_roster_count": active_roster_count,
        "active_roster_limit": active_roster_limit,
        "open_active_spots": open_active_spots,
        "il_count": il_count,
        "il_limit": il_limit,
        "sp_count": sp_count,
        "sp_limit": sp_limit,
        "rp_count": rp_count,
        "rp_limit": rp_limit,
        "can_add_without_drop": open_active_spots > 0,
        "flexibility_note": " ".join(flexibility_notes),
    }


def _get_position_value(player_row) -> str:
    """Read eligible positions from either supported CSV position column."""

    eligible_positions = player_row.get("eligible_positions", "")

    if isinstance(eligible_positions, str) and eligible_positions.strip():
        return eligible_positions

    return player_row.get("position", "")


def _build_projection_lookup(projections_df: pd.DataFrame) -> dict:
    """Create a normalized-name lookup for projected points and source."""

    if projections_df is None or projections_df.empty or "player" not in projections_df:
        return {}

    projection_rows = projections_df.copy()
    projection_rows["normalized_player_name"] = projection_rows["player"].apply(
        normalize_player_name
    )
    projection_rows = projection_rows.sort_values(
        by="projected_fantasy_points",
        ascending=False,
    ).drop_duplicates(subset=["normalized_player_name"], keep="first")

    return {
        row["normalized_player_name"]: row.to_dict()
        for _, row in projection_rows.iterrows()
    }


def _enrich_players_with_projection(
    players_df: pd.DataFrame,
    projection_lookup: dict,
) -> pd.DataFrame:
    """Attach projected points to uploaded roster or available-player rows."""

    output_columns = [
        "player",
        "eligible_positions",
        "projected_fantasy_points",
        "projection_source",
    ]

    if players_df is None or players_df.empty or "player" not in players_df.columns:
        return pd.DataFrame(columns=output_columns)

    enriched_rows = []

    for _, player_row in players_df.iterrows():
        player_name = player_row.get("player", "")
        normalized_player_name = normalize_player_name(player_name)
        projection_row = projection_lookup.get(normalized_player_name, {})

        # Players without a projection are kept at 0 and marked Missing so the
        # replacement table does not silently treat missing data as real value.
        enriched_rows.append(
            {
                "player": player_name,
                "eligible_positions": _get_position_value(player_row),
                "projected_fantasy_points": projection_row.get(
                    "projected_fantasy_points",
                    0,
                ),
                "projection_source": projection_row.get(
                    "projection_source",
                    "Missing",
                ),
            }
        )

    return pd.DataFrame(enriched_rows, columns=output_columns)


def _get_best_player_for_position(
    players_df: pd.DataFrame,
    roster_slot: str,
) -> dict:
    """Return the highest projected eligible player for one exact position."""

    if players_df.empty:
        return {
            "player": "",
            "projected_fantasy_points": 0.0,
            "projection_source": "Missing",
        }

    eligible_players = players_df[
        players_df["eligible_positions"].apply(
            lambda positions: can_play_position(positions, roster_slot)
        )
    ].copy()

    if eligible_players.empty:
        return {
            "player": "",
            "projected_fantasy_points": 0.0,
            "projection_source": "Missing",
        }

    eligible_players["projected_fantasy_points"] = pd.to_numeric(
        eligible_players["projected_fantasy_points"],
        errors="coerce",
    ).fillna(0)
    best_player = eligible_players.sort_values(
        by="projected_fantasy_points",
        ascending=False,
    ).iloc[0]

    return {
        "player": best_player.get("player", ""),
        "projected_fantasy_points": float(
            best_player.get("projected_fantasy_points", 0)
        ),
        "projection_source": best_player.get("projection_source", "Missing"),
    }


def _interpret_replacement_gap(replacement_gap: float) -> str:
    """Convert a replacement gap into a readable position status."""

    if replacement_gap >= 3:
        return "Strong roster advantage"

    if replacement_gap >= 1:
        return "Moderate roster advantage"

    if replacement_gap > -1:
        return "Streamable position"

    return "Waiver upgrade available"


def calculate_replacement_value_by_position(
    roster_df: pd.DataFrame,
    available_df: pd.DataFrame,
    projections_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare rostered and available hitters at each exact starting position.

    This helps identify whether a position is worth protecting or whether
    waiver players are close enough in projected value to stream. Eligibility is
    exact: LF, CF, RF, and DH only match when that exact position is listed.
    """

    output_columns = [
        "position",
        "best_rostered_player",
        "best_rostered_projected_points",
        "best_rostered_projection_source",
        "best_available_player",
        "best_available_projected_points",
        "best_available_projection_source",
        "replacement_gap",
        "position_status",
    ]
    projection_lookup = _build_projection_lookup(projections_df)
    roster_players = _enrich_players_with_projection(roster_df, projection_lookup)
    available_players = _enrich_players_with_projection(available_df, projection_lookup)
    replacement_rows = []

    for roster_slot in get_hitter_starting_slots():
        best_rostered = _get_best_player_for_position(roster_players, roster_slot)
        best_available = _get_best_player_for_position(available_players, roster_slot)
        replacement_gap = (
            best_rostered["projected_fantasy_points"]
            - best_available["projected_fantasy_points"]
        )

        replacement_rows.append(
            {
                "position": roster_slot,
                "best_rostered_player": best_rostered["player"],
                "best_rostered_projected_points": best_rostered[
                    "projected_fantasy_points"
                ],
                "best_rostered_projection_source": best_rostered[
                    "projection_source"
                ],
                "best_available_player": best_available["player"],
                "best_available_projected_points": best_available[
                    "projected_fantasy_points"
                ],
                "best_available_projection_source": best_available[
                    "projection_source"
                ],
                "replacement_gap": replacement_gap,
                "position_status": _interpret_replacement_gap(replacement_gap),
            }
        )

    return pd.DataFrame(replacement_rows, columns=output_columns)


def has_open_active_roster_spot(roster_df: pd.DataFrame) -> bool:
    """Return True when the active roster has room for another player.

    Carrying fewer SP than the maximum can create hitter streaming flexibility
    because the active roster limit is 17 and you do not need to fill every
    pitcher maximum slot.
    """

    return get_roster_flexibility_summary(roster_df)["can_add_without_drop"]


def is_droppable_player(player_row) -> bool:
    """Return True when a roster player can safely be dropped.

    Rules:
    - If undroppable is true, never drop.
    - IL players are never dropped by this hitter pickup evaluator.
    - Core keepers are never dropped.
    - If droppable is true, allow the drop.
    - High long-term value players need droppable=TRUE before they can be dropped.
    - Streamers and Drop Candidates can be dropped when not IL or undroppable.
    - If droppable is blank or missing, protect the player by default.
    """

    drop_status, _ = get_drop_protection_status(player_row)

    return drop_status == "Droppable"


def _clean_roster_label(value) -> str:
    """Normalize optional roster strategy labels from uploaded CSVs."""

    if not isinstance(value, str):
        return ""

    return re.sub(r"\s+", " ", value.strip()).upper()


def get_drop_protection_status(player_row) -> tuple[str, str]:
    """Explain whether a roster player can be dropped.

    The Top Pickups page uses this helper to make the optimizer's drop logic
    visible. Keeping the explanation here helps the review table and the actual
    add/drop evaluator stay consistent.
    """

    if _is_truthy(player_row.get("undroppable", "")):
        return "Undroppable", "Marked undroppable in roster CSV"

    if is_il_player(player_row):
        return "IL - Do Not Drop", "Player is on IL and should not be dropped"

    keeper_status = _clean_roster_label(player_row.get("keeper_status", ""))
    long_term_value = _clean_roster_label(player_row.get("long_term_value", ""))

    if keeper_status == "CORE":
        return "Undroppable", "Keeper status is Core"

    if _is_truthy(player_row.get("droppable", "")):
        return "Droppable", "Marked droppable in roster CSV"

    if long_term_value == "HIGH":
        return (
            "Protected by Default",
            "High long-term value player needs droppable TRUE before app suggests a drop",
        )

    if keeper_status in {"DROP CANDIDATE", "STREAMER"}:
        return "Droppable", f"Keeper status is {player_row.get('keeper_status')}"

    return (
        "Protected by Default",
        "No droppable flag provided, so app protects this player by default",
    )


def get_drop_risk(player_row) -> str:
    """Return a plain risk label for a proposed dropped player."""

    keeper_status = _clean_roster_label(player_row.get("keeper_status", ""))
    long_term_value = _clean_roster_label(player_row.get("long_term_value", ""))

    if long_term_value == "HIGH" or keeper_status == "CORE":
        return "High"

    if long_term_value == "MEDIUM" or keeper_status == "HOLD":
        return "Medium"

    if long_term_value == "LOW" or keeper_status in {"DROP CANDIDATE", "STREAMER"}:
        return "Low"

    return ""


def optimize_hitter_lineup(players_df: pd.DataFrame):
    """Choose the best valid daily hitter lineup from current roster hitters.

    This first version uses a brute-force recursive search. The roster is small,
    so it is okay to try possible player-slot assignments and keep the best
    scoring lineup. We can replace this with a faster optimizer later.

    Returns:
        lineup_df: Starting hitter assignments.
        bench_df: Roster hitters not used in the starting lineup.
        warning_message: Empty when a full lineup is possible, otherwise a
            friendly explanation of which slots could not be filled.
    """

    required_columns = {"player", "eligible_positions", "projected_fantasy_points"}

    if players_df is None or players_df.empty:
        return _empty_lineup_result(
            pd.DataFrame(columns=list(required_columns)),
            "No roster hitters were provided for lineup optimization.",
        )

    missing_columns = required_columns - set(players_df.columns)

    if missing_columns:
        return _empty_lineup_result(
            players_df,
            "Lineup optimization needs these missing columns: "
            + ", ".join(sorted(missing_columns)),
        )

    roster_players = players_df.copy().reset_index(drop=True)
    roster_players["_optimizer_player_id"] = roster_players.index
    hitter_slots = get_hitter_starting_slots()

    best_assignments = []
    best_total_points = -1

    def search(slot_index: int, used_player_ids: set, current_assignments: list):
        """Try valid assignments one lineup slot at a time."""

        nonlocal best_assignments
        nonlocal best_total_points

        current_total_points = sum(
            assignment["projected_fantasy_points"] for assignment in current_assignments
        )

        # Prefer the lineup with more filled slots. Use projected points as the
        # tiebreaker when two partial lineups fill the same number of slots.
        if (
            len(current_assignments) > len(best_assignments)
            or (
                len(current_assignments) == len(best_assignments)
                and current_total_points > best_total_points
            )
        ):
            best_assignments = current_assignments.copy()
            best_total_points = current_total_points

        if slot_index >= len(hitter_slots):
            return

        roster_slot = hitter_slots[slot_index]

        # Option 1: try every unused player who is eligible for this exact slot.
        for _, player_row in roster_players.iterrows():
            player_id = player_row["_optimizer_player_id"]

            if player_id in used_player_ids:
                continue

            if not validate_hitter_eligibility(player_row, roster_slot):
                continue

            next_assignment = {
                "roster_slot": roster_slot,
                "player": player_row["player"],
                "eligible_positions": player_row["eligible_positions"],
                "projected_fantasy_points": player_row["projected_fantasy_points"],
                "projection_source": player_row.get("projection_source", ""),
                "projection_confidence": player_row.get("projection_confidence", ""),
                "_optimizer_player_id": player_id,
            }

            search(
                slot_index + 1,
                used_player_ids | {player_id},
                current_assignments + [next_assignment],
            )

        # Option 2: skip this slot. This lets us return the best partial lineup
        # when a complete valid lineup cannot be created.
        search(slot_index + 1, used_player_ids, current_assignments)

    search(0, set(), [])

    lineup_df = pd.DataFrame(best_assignments)

    if lineup_df.empty:
        used_player_ids = set()
    else:
        used_player_ids = set(lineup_df["_optimizer_player_id"])
        lineup_df = lineup_df.drop(columns=["_optimizer_player_id"])

    bench_df = roster_players[
        ~roster_players["_optimizer_player_id"].isin(used_player_ids)
    ].drop(columns=["_optimizer_player_id"])

    filled_slots = set(lineup_df["roster_slot"]) if not lineup_df.empty else set()
    missing_slots = [slot for slot in hitter_slots if slot not in filled_slots]

    if missing_slots:
        warning_message = (
            "A full valid hitter lineup could not be created. Missing slots: "
            + ", ".join(missing_slots)
            + "."
        )
    else:
        warning_message = ""

    return lineup_df, bench_df, warning_message


def _get_starting_total(lineup_df: pd.DataFrame) -> float:
    """Return total projected fantasy points for a lineup dataframe."""

    if lineup_df.empty:
        return 0.0

    return float(lineup_df["projected_fantasy_points"].sum())


def compare_lineups(
    current_lineup_df: pd.DataFrame,
    new_lineup_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare two optimized lineups and return changed slots only."""

    output_columns = [
        "roster_slot",
        "current_player",
        "current_projected_points",
        "new_player",
        "new_projected_points",
        "point_difference",
    ]

    if current_lineup_df is None:
        current_lineup_df = pd.DataFrame()

    if new_lineup_df is None:
        new_lineup_df = pd.DataFrame()

    current_by_slot = current_lineup_df.set_index("roster_slot")
    new_by_slot = new_lineup_df.set_index("roster_slot")
    changed_rows = []

    for roster_slot in get_hitter_starting_slots():
        current_player = ""
        current_points = 0.0
        new_player = ""
        new_points = 0.0

        if roster_slot in current_by_slot.index:
            current_player = current_by_slot.loc[roster_slot].get("player", "")
            current_points = current_by_slot.loc[roster_slot].get(
                "projected_fantasy_points",
                0,
            )

        if roster_slot in new_by_slot.index:
            new_player = new_by_slot.loc[roster_slot].get("player", "")
            new_points = new_by_slot.loc[roster_slot].get(
                "projected_fantasy_points",
                0,
            )

        if current_player != new_player or current_points != new_points:
            changed_rows.append(
                {
                    "roster_slot": roster_slot,
                    "current_player": current_player,
                    "current_projected_points": current_points,
                    "new_player": new_player,
                    "new_projected_points": new_points,
                    "point_difference": new_points - current_points,
                }
            )

    return pd.DataFrame(changed_rows, columns=output_columns)


def _get_pickup_recommendation(projected_gain: float) -> str:
    """Convert projected gain into a simple recommendation label."""

    if projected_gain >= 3:
        return "Strong add"

    if projected_gain >= 1:
        return "Marginal add"

    return "No move"


def calculate_risk_adjusted_score(row) -> float:
    """Score a move after accounting for roster risk and projection confidence.

    projected_gain is today's raw projected point improvement. The
    risk_adjusted_score starts with that gain, then subtracts penalties for
    risky drops and lower-confidence projections.
    """

    risk_adjusted_score = row.get("projected_gain", 0)

    if pd.isna(risk_adjusted_score):
        risk_adjusted_score = 0

    if row.get("move_type") != "Add Only":
        drop_risk = row.get("drop_risk", "")

        if drop_risk == "Low":
            risk_adjusted_score -= 0.25
        elif drop_risk == "Medium":
            risk_adjusted_score -= 1.0
        elif drop_risk == "High":
            risk_adjusted_score -= 3.0

    projection_confidence = row.get(
        "projection_confidence",
        row.get("add_projection_confidence", ""),
    )
    add_projection_confidence = row.get("add_projection_confidence", "")

    for confidence_value in {projection_confidence, add_projection_confidence}:
        if confidence_value == "Low":
            risk_adjusted_score -= 0.5
        elif confidence_value == "Unknown":
            risk_adjusted_score -= 1.0

    return round(float(risk_adjusted_score), 2)


def _get_risk_adjusted_label(risk_adjusted_score: float) -> str:
    """Convert a risk-adjusted score into a readable label."""

    if risk_adjusted_score >= 3:
        return "Excellent"

    if risk_adjusted_score >= 1.5:
        return "Good"

    if risk_adjusted_score > 0:
        return "Thin edge"

    return "Avoid"


def _get_multi_add_risk_adjusted_label(risk_adjusted_score: float) -> str:
    """Convert a multi-add risk-adjusted score into a readable label."""

    if risk_adjusted_score >= 5:
        return "Excellent"

    if risk_adjusted_score >= 3:
        return "Good"

    if risk_adjusted_score > 0:
        return "Thin edge"

    return "Avoid"


def _get_multi_add_recommendation(projected_gain: float) -> str:
    """Convert multi-add projected gain into a simple recommendation label."""

    if projected_gain >= 5:
        return "Strong multi-add"

    if projected_gain >= 3:
        return "Good multi-add"

    if projected_gain > 0:
        return "Thin multi-add"

    return "No move"


def _build_pickup_recommendation_row(row_data: dict) -> dict:
    """Add shared labels and risk-adjusted score to a recommendation row."""

    risk_adjusted_score = calculate_risk_adjusted_score(row_data)
    row_data["risk_adjusted_score"] = risk_adjusted_score
    row_data["risk_adjusted_label"] = _get_risk_adjusted_label(risk_adjusted_score)

    return row_data


def _calculate_multi_add_risk_adjusted_score(
    projected_gain: float,
    added_players: pd.DataFrame,
) -> float:
    """Score a multi-add move without drop risk.

    Multi-add moves do not drop anyone, so there is no season-long drop
    penalty. We still subtract for lower-confidence projections because those
    adds are less certain.
    """

    risk_adjusted_score = projected_gain

    for _, player_row in added_players.iterrows():
        projection_confidence = player_row.get("projection_confidence", "")

        if projection_confidence == "Low":
            risk_adjusted_score -= 0.5
        elif projection_confidence == "Unknown":
            risk_adjusted_score -= 1.0

    return round(float(risk_adjusted_score), 2)


def evaluate_multi_add_hitter_scenarios(
    roster_df: pd.DataFrame,
    available_df: pd.DataFrame,
    max_adds: int | None = None,
) -> pd.DataFrame:
    """Evaluate adding multiple hitters when open roster spots exist.

    This intentionally uses a brute-force combination search capped at three
    adds. That keeps the first version easy to understand and fast enough for a
    small fantasy roster.
    """

    output_columns = [
        "move_type",
        "add_players",
        "add_eligible_positions",
        "number_of_adds",
        "current_active_roster_count",
        "open_active_spots",
        "current_starting_total",
        "new_starting_total",
        "projected_gain",
        "risk_adjusted_score",
        "risk_adjusted_label",
        "recommendation",
    ]

    if roster_df is None or available_df is None or roster_df.empty or available_df.empty:
        return pd.DataFrame(columns=output_columns)

    available_hitters_df = available_df[
        available_df.apply(lambda row: infer_player_type(row) == "HITTER", axis=1)
    ].copy()

    if available_hitters_df.empty:
        return pd.DataFrame(columns=output_columns)

    current_active_roster_count = count_active_roster_spots(roster_df)
    open_active_spots = LEAGUE_RULES["active_roster_spots"] - current_active_roster_count

    if open_active_spots <= 0:
        return pd.DataFrame(columns=output_columns)

    if max_adds is None:
        max_adds = min(open_active_spots, 3)
    else:
        max_adds = min(max_adds, open_active_spots, 3)

    current_lineup_df, _, _ = optimize_hitter_lineup(roster_df)
    current_starting_total = _get_starting_total(current_lineup_df)
    scenario_rows = []

    # Try every available-player combination from one add up to max_adds.
    # This is simple and readable; if the available pool grows large later, we
    # can replace it with a smarter search.
    for number_of_adds in range(1, max_adds + 1):
        for player_indexes in combinations(available_hitters_df.index, number_of_adds):
            added_players = available_hitters_df.loc[list(player_indexes)].copy()
            roster_with_adds = pd.concat(
                [roster_df, added_players],
                ignore_index=True,
            )
            new_lineup_df, _, _ = optimize_hitter_lineup(roster_with_adds)
            new_starting_total = _get_starting_total(new_lineup_df)
            projected_gain = new_starting_total - current_starting_total
            risk_adjusted_score = _calculate_multi_add_risk_adjusted_score(
                projected_gain,
                added_players,
            )

            scenario_rows.append(
                {
                    "move_type": "Multi Add",
                    "add_players": ", ".join(
                        added_players["player"].fillna("").astype(str)
                    ),
                    "add_eligible_positions": " | ".join(
                        added_players["eligible_positions"].fillna("").astype(str)
                    ),
                    "number_of_adds": number_of_adds,
                    "current_active_roster_count": current_active_roster_count,
                    "open_active_spots": open_active_spots,
                    "current_starting_total": current_starting_total,
                    "new_starting_total": new_starting_total,
                    "projected_gain": projected_gain,
                    "risk_adjusted_score": risk_adjusted_score,
                    "risk_adjusted_label": _get_multi_add_risk_adjusted_label(
                        risk_adjusted_score
                    ),
                    "recommendation": _get_multi_add_recommendation(projected_gain),
                }
            )

    return pd.DataFrame(scenario_rows, columns=output_columns).sort_values(
        by="risk_adjusted_score",
        ascending=False,
    )


def evaluate_single_hitter_pickups(
    roster_df: pd.DataFrame,
    available_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate one hitter pickup move at a time.

    The evaluator:
    1. Optimizes the current roster.
    2. Adds one available hitter.
    3. Drops one safe droppable roster player only when the active roster is full.
    4. Optimizes the new roster.
    5. Measures the projected gain.

    Pitchers are intentionally ignored for now.
    """

    output_columns = [
        "move_type",
        "add_player",
        "add_eligible_positions",
        "drop_player",
        "drop_eligible_positions",
        "current_active_roster_count",
        "current_starting_total",
        "new_starting_total",
        "projected_gain",
        "add_projection_source",
        "add_projection_confidence",
        "drop_projection_source",
        "drop_projection_confidence",
        "drop_long_term_value",
        "drop_keeper_status",
        "drop_risk",
        "risk_adjusted_score",
        "risk_adjusted_label",
        "recommendation",
    ]

    if roster_df is None or available_df is None or roster_df.empty or available_df.empty:
        return pd.DataFrame(columns=output_columns)

    current_lineup_df, _, _ = optimize_hitter_lineup(roster_df)
    current_starting_total = _get_starting_total(current_lineup_df)
    current_active_roster_count = count_active_roster_spots(roster_df)

    recommendations = []

    if has_open_active_roster_spot(roster_df):
        # Add-only moves are allowed when there are fewer than 17 active players.
        # Carrying fewer pitchers than the maximum can create this kind of
        # hitter bench flexibility.
        for _, available_player in available_df.iterrows():
            roster_with_add = pd.concat(
                [
                    roster_df,
                    pd.DataFrame([available_player.to_dict()]),
                ],
                ignore_index=True,
            )

            new_lineup_df, _, _ = optimize_hitter_lineup(roster_with_add)
            new_starting_total = _get_starting_total(new_lineup_df)
            projected_gain = new_starting_total - current_starting_total

            recommendations.append(
                _build_pickup_recommendation_row(
                    {
                        "move_type": "Add Only",
                        "add_player": available_player.get("player", ""),
                        "add_eligible_positions": available_player.get(
                            "eligible_positions",
                            "",
                        ),
                        "drop_player": "",
                        "drop_eligible_positions": "",
                        "current_active_roster_count": current_active_roster_count,
                        "current_starting_total": current_starting_total,
                        "new_starting_total": new_starting_total,
                        "projected_gain": projected_gain,
                        "add_projection_source": available_player.get(
                            "projection_source",
                            "",
                        ),
                        "add_projection_confidence": available_player.get(
                            "projection_confidence",
                            "",
                        ),
                        "drop_projection_source": "",
                        "drop_projection_confidence": "",
                        "drop_long_term_value": "",
                        "drop_keeper_status": "",
                        "drop_risk": "",
                        "recommendation": _get_pickup_recommendation(projected_gain),
                    }
                )
            )
    else:
        droppable_roster_df = roster_df[
            roster_df.apply(is_droppable_player, axis=1)
        ].copy()

        for _, available_player in available_df.iterrows():
            for _, drop_player in droppable_roster_df.iterrows():
                roster_without_drop = roster_df[
                    roster_df["player"] != drop_player["player"]
                ].copy()
                roster_with_add = pd.concat(
                    [
                        roster_without_drop,
                        pd.DataFrame([available_player.to_dict()]),
                    ],
                    ignore_index=True,
                )

                new_lineup_df, _, _ = optimize_hitter_lineup(roster_with_add)
                new_starting_total = _get_starting_total(new_lineup_df)
                projected_gain = new_starting_total - current_starting_total

                recommendations.append(
                    _build_pickup_recommendation_row(
                        {
                            "move_type": "Add/Drop",
                            "add_player": available_player.get("player", ""),
                            "add_eligible_positions": available_player.get(
                                "eligible_positions",
                                "",
                            ),
                            "drop_player": drop_player.get("player", ""),
                            "drop_eligible_positions": drop_player.get(
                                "eligible_positions",
                                "",
                            ),
                            "current_active_roster_count": current_active_roster_count,
                            "current_starting_total": current_starting_total,
                            "new_starting_total": new_starting_total,
                            "projected_gain": projected_gain,
                            "add_projection_source": available_player.get(
                                "projection_source",
                                "",
                            ),
                            "add_projection_confidence": available_player.get(
                                "projection_confidence",
                                "",
                            ),
                            "drop_projection_source": drop_player.get(
                                "projection_source",
                                "",
                            ),
                            "drop_projection_confidence": drop_player.get(
                                "projection_confidence",
                                "",
                            ),
                            "drop_long_term_value": drop_player.get(
                                "long_term_value",
                                "",
                            ),
                            "drop_keeper_status": drop_player.get(
                                "keeper_status",
                                "",
                            ),
                            "drop_risk": get_drop_risk(drop_player),
                            "recommendation": _get_pickup_recommendation(projected_gain),
                        }
                    )
                )

    return pd.DataFrame(recommendations, columns=output_columns).sort_values(
        by="risk_adjusted_score",
        ascending=False,
    )
