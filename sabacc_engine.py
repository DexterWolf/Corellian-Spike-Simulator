from __future__ import annotations

import sys
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional, Dict, Any, TypedDict

from sabacc_cards import (
    Card, SUITS,
    build_full_deck, remove_values_from_deck,
    hand_sum, hand_values,
    hand_abs_values_sorted_desc, hand_positives_sorted_desc, hand_is_suited_nonzero,
)
from sabacc_hands import (
    NAMED_ORDERS,
    named_detect_active,
)
from sabacc_policies import (
    full_value_counts,
    known_add, known_clear, known_remove,
    ensure_discard_after_gain,
    _bot_safe_draw, _drawtozero_core, _swaptozero_core,
    _hunter_core, _minimizer_core,
    forced_discard_draw_index_min_abs,
)

# Interactive play pauses (seconds). No CLI flags; set here.
PAUSE_AFTER_ACTION = 1.0  # default 1.0s after actions and dice rolls
PAUSE_BEFORE_RESULTS = 2.0  # default 2.0s before printing results
def fmt_hand_inline(hand: List[Card]) -> str:
    return "[{}]".format(", ".join(str(c) for c in hand))

def fmt_hand_positions(hand: List[Card]) -> str:
    # "  1: [2]  2: [-2]  3: [-2]   (sum=-2)"
    parts = []
    for i, c in enumerate(hand, start=1):
        parts.append(f"{i}: [{c}]")
    return "  ".join(parts)

def pause_after_action():
    if PAUSE_AFTER_ACTION > 0:
        time.sleep(PAUSE_AFTER_ACTION)

def pause_before_results():
    if PAUSE_BEFORE_RESULTS > 0:
        time.sleep(PAUSE_BEFORE_RESULTS)

# =====================
# Simulation Bookkeeping
# =====================



@dataclass
class DiceStats:
    doubles_in_game: int = 0
    spike_doubles_in_game: int = 0

@dataclass
class RunCounters:
    reshuffles_total: int = 0
    reshuffles_dice: int = 0
    doubles_hist: Counter = field(default_factory=Counter)       # (d1,d2) -> count
    spike_doubles_hist: Counter = field(default_factory=Counter) # (d1,d2) in spike mode
    true_ties_games: int = 0      # games where single-card draw was used
    total_games: int = 0          # total games actually played in this run
    tiebreaker_hist: Counter = field(default_factory=Counter)  # int code -> count

# Compact integer codes for the last effective tiebreaker used in a game.
TIEBREAKER_CODES = {
    "single_zero_named": 1,
    "single_zero_unnamed": 2,
    "best_named_among_zeros": 3,
    "named_key": 4,
    "closest_to_zero": 5,
    "positive_closest_to_zero": 6,
    "cards_first": 7,
    "sum_abs": 8,
    "high_abs": 9,
    "positive_cards": 10,
    "suited": 11,
    "single_card_draw": 12,
}

TIEBREAKER_LABELS = {
    1: "Sabacc (named hand)",
    2: "Sabacc",
    3: "Sabacc (best named)",
    4: "Sabacc (named key)",
    5: "Nulrhek",
    6: "Nulrhek → positive",
    7: "Most cards",
    8: "Highest Σ|cards|",
    9: "Highest |card|",
    10: "Highest positive card",
    11: "Suited",
    12: "Draw",
}


def record_tiebreak(counters: Optional[RunCounters], key: str) -> None:
    """Increment global non-string tiebreak counter on the run counters."""
    if counters is None:
        return
    code = TIEBREAKER_CODES.get(key)
    if code is None:
        return
    counters.tiebreaker_hist[code] += 1

@dataclass
class DiceRoll:
    d1: int
    d2: int
    redraw_sizes: List[int]
    is_doubles: bool
    is_spike_double: bool
    dice_events: List[str] = field(default_factory=list)

@dataclass
class ReplayMove:
    player_index: int
    mode_label: str
    before: List[int]
    action: str
    after: List[int]
    reshuffle_events: List[str] = field(default_factory=list)

# =====================
# Replay Output  [EXPERIMENTAL — schema and format not yet stable]
# =====================
class ReplayFileWriter:
    def __init__(self, path, fmt, size_spec):
        # JSON used for JSONL mode
        import json as _json  # ensure available even if caller forgot at discard_top-level
        self._json = _json

        self.enabled = bool(path)
        self.path = path
        self.fmt = (fmt or "jsonl").lower()

        # parse size limit
        self.limit = None
        if size_spec:
            s = str(size_spec).strip().lower()
            if s == "all":
                self.limit = None
            else:
                try:
                    self.limit = int(s)
                except Exception:
                    raise SystemExit("--replay-size must be an integer or 'all'")

        # targets
        self.fp = None
        self.pq_writer = None
        self.pa = None
        self.pq = None

        if not self.enabled:
            return

        if self.fmt == "jsonl":
            self.fp = open(self.path, "w", encoding="utf-8")

        elif self.fmt == "parquet":
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except Exception:
                print("Parquet output requires pyarrow. Please install pyarrow or use --replay-format jsonl.")
                raise

            self.pa = pa
            self.pq = pq

            # Fully nested schema (structs/lists) matching file_data shape in play_one_game(...)
            move_struct = pa.struct([
                pa.field("player_index", pa.int32()),
                pa.field("mode_label", pa.string()),
                pa.field("before", pa.list_(pa.int32())),
                pa.field("before_suits", pa.list_(pa.string())),
                pa.field("action", pa.string()),
                pa.field("after", pa.list_(pa.int32())),
                pa.field("after_suits", pa.list_(pa.string())),
                pa.field("reshuffle_events", pa.list_(pa.string())),
            ])

            dice_struct = pa.struct([
                pa.field("d1", pa.int32()),
                pa.field("d2", pa.int32()),
                pa.field("redraw_sizes", pa.list_(pa.int32())),
                pa.field("is_doubles", pa.bool_()),
                pa.field("is_spike_double", pa.bool_()),
                pa.field("dice_events", pa.list_(pa.string())),
            ])

            final_struct = pa.struct([
                pa.field("values", pa.list_(pa.int32())),
                pa.field("suits", pa.list_(pa.string())),
                pa.field("sum", pa.int32()),
                pa.field("name", pa.string()).with_nullable(True),
            ])

            schema = pa.schema([
                pa.field("game_index", pa.int32()),

                pa.field("modes", pa.list_(pa.string())),
                pa.field("starts", pa.list_(pa.list_(pa.int32()))),
                pa.field("starts_suits", pa.list_(pa.list_(pa.string()))),

                pa.field("first_discard", pa.struct([
                    pa.field("value", pa.int32()),
                    pa.field("suit", pa.string()),
                ])),

                # rounds_moves is: list(round) of list(move-struct)
                pa.field("rounds_moves", pa.list_(pa.list_(move_struct))),

                # list(dice-struct)
                pa.field("dice_rolls", pa.list_(dice_struct)),

                # list(final-struct)
                pa.field("finals", pa.list_(final_struct)),

                pa.field("winner", pa.int32()),
                pa.field("winner_reason", pa.string()),
                pa.field("doubles_in_game", pa.int32()),
                pa.field("spike_doubles_in_game", pa.int32()),
            ])

            self.schema = schema
            self.pq_writer = pq.ParquetWriter(self.path, schema)

        else:
            raise SystemExit("--replay-format must be jsonl or parquet")

    def will_write(self, game_index: int) -> bool:
        if not self.enabled:
            return False
        if self.limit is None:
            return True
        return int(game_index) <= self.limit

    def _to_json(self, obj):
        return self._json.dumps(obj, ensure_ascii=False)

    def write_game(self, game_index, data: dict):
        if not self.enabled:
            return

        if self.fmt == "jsonl":
            rec = data.copy()
            rec["game_index"] = int(game_index)
            self.fp.write(self._to_json(rec) + "\n")
            return

        # Parquet (fully nested)
        pa = self.pa

        def _suits(lst):
            # normalize None -> "" to satisfy pa.string() element type
            return [("" if s is None else str(s)) for s in (lst or [])]

        # Normalize DiceRoll instances to dicts for dice_rolls
        dice_rolls = []
        for item in data.get("dice_rolls", []):
            dice_rolls.append({
                "d1": int(item.d1),
                "d2": int(item.d2),
                "redraw_sizes": list(item.redraw_sizes or []),
                "is_doubles": bool(item.is_doubles),
                "is_spike_double": bool(item.is_spike_double),
                "dice_events": list(item.dice_events or []),
            })

        # rounds_moves already contains dicts; ensure suit lists are normalized
        rounds_moves = []
        for round_moves in data.get("rounds_moves", []):
            rounds_moves.append([
                {
                    "player_index": int(mv.get("player_index")),
                    "mode_label": str(mv.get("mode_label")),
                    "before": list(mv.get("before") or []),
                    "before_suits": _suits(mv.get("before_suits")),
                    "action": str(mv.get("action")),
                    "after": list(mv.get("after") or []),
                    "after_suits": _suits(mv.get("after_suits")),
                    "reshuffle_events": list(mv.get("reshuffle_events") or []),
                }
                for mv in (round_moves or [])
            ])

        finals = []
        for f in data.get("finals", []):
            finals.append({
                "values": list(f.get("values") or []),
                "suits": _suits(f.get("suits")),
                "sum": int(f.get("sum") or 0),
                "name": (None if (f.get("name") in (None, "")) else str(f.get("name"))),
            })

        # starts_suits can contain None; normalize them too
        starts_suits = []
        for pair in (data.get("starts_suits") or []):
            starts_suits.append(_suits(pair))

        row = {
            "game_index": int(game_index),
            "modes": list(data.get("modes") or []),
            "starts": list(data.get("starts") or []),
            "starts_suits": starts_suits,
            "first_discard": {
                "value": int((data.get("first_discard") or {}).get("value", 0)),
                "suit": str((data.get("first_discard") or {}).get("suit", "") or ""),
            },
            "rounds_moves": rounds_moves,
            "dice_rolls": dice_rolls,
            "finals": finals,
            "winner": int(data.get("winner", 0)),
            "winner_reason": str(data.get("winner_reason", "")),
            "doubles_in_game": int(data.get("doubles_in_game", 0)),
            "spike_doubles_in_game": int(data.get("spike_doubles_in_game", 0)),
        }

        # IMPORTANT: build with the exact writer schema to avoid inference mismatches
        tbl = pa.Table.from_pylist([row], schema=self.schema)
        self.pq_writer.write_table(tbl)


    def close(self):
        try:
            if self.fp:
                self.fp.close()
        finally:
            self.fp = None
        try:
            if self.pq_writer:
                self.pq_writer.close()
        finally:
            self.pq_writer = None



# =====================
# Game State Type
# =====================

class SimConfig(TypedDict, total=False):
    """Simulation configuration built from CLI args and passed as state["opts"].

    All keys are optional (total=False) because the --compare code path builds
    a subset dict containing only the four tiebreaker-relevant keys.  In the
    normal simulation path all keys are present.
    """
    # --- Game setup ---
    n_games: int               # -n: number of games to simulate
    num_players: int           # --num-players: 2–8
    seed_given: bool           # True when --seed was provided explicitly
    human_player: Optional[int]  # --human-player: 1-indexed seat, or None
    # --- Dice ---
    no_dice: bool              # --no-dice: disable dice phase entirely
    dice_mode: str             # --dice-mode: "spike" (default) | "classic"
    spike_cheat: bool          # --spike-cheat: interactive override of dice rolls
    # --- Card actions ---
    allow_discard_gain: bool   # --allow-discard-gain: enable gaining from discard
    # --- Starting hands ---
    random_starts: bool        # --random-starts: randomise starts once per run
    randomize_all: bool        # --randomize-all: new random starts every game
    rotate_first: bool         # --rotate-first: rotate first-to-act seat each game
    # --- Named hand & tiebreaker settings ---
    named_order: str           # --named-order: "default" | "old" | "galedge"
    named_low_wins: bool       # --named-low-wins: lower rank key wins named tiebreak
    high_abs: bool             # True by default; --no-high-abs disables highest-|card| tiebreak step
    use_suits: bool            # True by default; --no-suits disables suited tiebreak step
    # --- Replay / output ---
    keep_games: int            # --keep-games: how many games to keep full replay for
    no_replay: bool            # --no-replay: disable all replay capture
    replay_file: Optional[str] # --replay-file: path for JSONL/Parquet output
    replay_format: str         # --replay-format: "jsonl" | "parquet"
    replay_size: Optional[str] # --replay-size: N or "all"


class GameState(TypedDict, total=False):
    """Shared mutable state dict threaded through the entire simulation.

    All keys are typed as optional (total=False) because the dict is built
    up incrementally: run-level keys are set in main(), per-game keys are
    reset each iteration of the game loop, and per-hand keys are set inside
    play_one_game().
    """
    # --- Run-level: set once in main(), persist for the whole run ---
    rng: random.Random
    opts: SimConfig
    counters: RunCounters
    recycle: Callable[..., None]           # engine_recycle(state, during_dice)
    abort: Callable[[str], None]           # engine_abort(msg) — exits the process
    interactive: bool
    known_in_hand_by_player: Dict[int, List[Card]]
    fixed_starts: Dict[int, List[int]]
    starts_run_vals: List[Optional[List[int]]]
    modes_run: List[Any]                   # str | ("hunter", target_name) per seat
    start_seat_index: int
    record_replay: bool
    replay_writer: Optional[ReplayFileWriter]
    # --- Per-game: reset each game in main()'s game loop ---
    record_file_replay: bool
    game_num: int
    # --- Per-hand: set (and mutated) inside play_one_game() ---
    hands: List[List[Card]]
    discard: List[Card]
    draw: List[Card]
    order: List[int]
    discard_full: List[Card]               # alias for discard; used by public_counts helpers
    round_index: int
    curr_round_moves: List[ReplayMove]
    file_curr_round_moves: List[Dict[str, Any]]
    human_index: Optional[int]
    dice_events: List[str]
    dealer_started_new_discard: Optional[Card]  # temporary; popped after use


# =====================
# Tiebreakers & Winner Resolution
# =====================

def compare_lex_desc(a: List[int], b: List[int]) -> int:
    """Return +1 if a>b, -1 if a<b, 0 if equal (lexicographic descending)."""
    for x,y in zip(a,b):
        if x > y: return +1
        if x < y: return -1
    if len(a) > len(b): return +1
    if len(a) < len(b): return -1
    return 0

def compare_lex_asc(a: List[int], b: List[int]) -> int:
    """Return +1 if a<b (since lower is better), -1 if a>b, 0 if equal; we map to +1 means a better."""
    for x,y in zip(a,b):
        if x < y: return +1
        if x > y: return -1
    if len(a) < len(b): return +1
    if len(a) > len(b): return -1
    return 0

def winner_and_reason(hands: List[List[Card]], opts: dict, rng: random.Random,
                      discard: List[Card], draw: List[Card],
                      counters: RunCounters, replay_trace: Optional[dict]) -> Tuple[int, str, bool]:
    """Decide winner among hands (list per seat).
    Returns: (winner_index (0-based), reason_string, used_true_tie_bool)
    The function may mutate draw/discard piles during the final 'single-card draw' tiebreaker.
    """
    n = len(hands)
    sums = [hand_sum(h) for h in hands]
    sabacc_seats = [i for i,s in enumerate(sums) if s == 0]

    active_order = NAMED_ORDERS[opts["named_order"]]
    hand_ranks = [None]*n
    for i in range(n):
        if sums[i] == 0:
            hand_ranks[i] = named_detect_active(hands[i], active_order)

    # 1) Exactly one zero wins
    if len(sabacc_seats) == 1:
        i = sabacc_seats[0]
        named_hand = hand_ranks[i][1] if hand_ranks[i] else None
        if named_hand:
            record_tiebreak(counters, "single_zero_named")
            reason = f"Sabacc: {hand_ranks[i][1]}"
        else:
            record_tiebreak(counters, "single_zero_unnamed")
            reason = "Sabacc"
        return i, reason, False

    # 2) Two or more sabacc_seats
    if len(sabacc_seats) >= 2:
        sabacc_seats = sabacc_seats[:]
        # If one is named and the other(s) not -> named wins
        named_sabacc_seats = [i for i in sabacc_seats if hand_ranks[i] is not None]
        if len(named_sabacc_seats) == 1:
            i = named_sabacc_seats[0]
            record_tiebreak(counters, "single_zero_named")
            return i, f"Sabacc → {hand_ranks[i][1]}", False
        if len(named_sabacc_seats) >= 2:
            # compare by (rank index lower better, then key lex (desc by default or asc if named_low_wins))
            def named_sort_key(i):
                rank, name, key = hand_ranks[i]
                return (rank, key)
            # find all best with lowest rank
            best_rank = min(hand_ranks[i][0] for i in named_sabacc_seats)
            candidates = [i for i in named_sabacc_seats if hand_ranks[i][0] == best_rank]
            if len(candidates) == 1:
                i = candidates[0]
                record_tiebreak(counters, "best_named_among_zeros")
                return i, f"Sabacc → {hand_ranks[i][1]}", False
            # same named type -> compare keys
            lower_key_wins = opts["named_low_wins"]
            # keys same length? Regardless, compare lex
            def better(a,b):
                key_a = hand_ranks[a][2]
                key_b = hand_ranks[b][2]
                if lower_key_wins:
                    res = compare_lex_asc(key_a,key_b)
                else:
                    res = compare_lex_desc(key_a,key_b)
                return res
            best = candidates[0]
            tie_group = [best]
            for candidate in candidates[1:]:
                r = better(candidate, best)
                if r > 0:
                    best = candidate
                    tie_group = [best]
                elif r == 0:
                    tie_group.append(candidate)

            i = tie_group[0]
            if len(tie_group) == 1:
                record_tiebreak(counters, "named_key")
                key_str = ",".join(str(k) for k in hand_ranks[i][2])
                return i, f"Sabacc: {hand_ranks[i][1]} → key {key_str}", False
            # still tied -> generic tiebreakers among these
            return generic_tiebreak(hands, tie_group, opts, rng, discard, draw, counters, replay_trace, chain=f"Sabacc: {hand_ranks[i][1]}")

        # none named -> generic tiebreakers among sabacc_seats
        return generic_tiebreak(hands, sabacc_seats, opts, rng, discard, draw, counters, replay_trace, chain="Sabacc")

    # 3) No sabacc_seats: smallest |sum| wins; if tie -> generic
    abs_sums = [abs(s) for s in sums]
    best_val = min(abs_sums)
    tied_seats = [i for i,a in enumerate(abs_sums) if a == best_val]
    if len(tied_seats) == 1:
        i = tied_seats[0]
        record_tiebreak(counters, "closest_to_zero")
        return i, "Nulrhek", False
    # tied on |sum| -> generic among tied_seats
    return generic_tiebreak(hands, tied_seats, opts, rng, discard, draw, counters, replay_trace, chain="Nulrhek")

def generic_tiebreak(hands: List[List[Card]], tied_seats: List[int], opts: dict, rng: random.Random,
                     discard: List[Card], draw: List[Card], counters: RunCounters, replay_trace: Optional[dict], chain: Optional[str] = None) -> Tuple[int,str,bool]:
    """Apply generic tiebreakers in order (see spec)."""
    # 1) Positive beats negative if both non-zero
    sums = [hand_sum(h) for h in hands]
    cabs = [abs(s) for s in sums]
    # Step 1 applies only when tied on |sum| and both non-zero (we call generic when tied on |sum| or as part of sabacc_seats logic)
    # We'll apply steps generically on tied_seats list.

    def reduce_by(tied_seats, fn, desc):
        """Return new tied_seats reduced by comparator fn returning best >0, 0 equal, -1 worse. Also returns reason suffix if decided."""
        best = tied_seats[0]
        ties = [best]
        for candidate in tied_seats[1:]:
            r = fn(candidate, best)
            if r > 0:
                best = candidate
                ties = [best]
            elif r == 0:
                ties.append(candidate)
        if len(ties) == 1:
            return ties, desc
        return ties, None


    # 1) Positive beats negative (if both non-zero)
    def cmp_pos_neg(a,b):
        sum_a, sum_b = sums[a], sums[b]
        # if either zero, no preference here
        if sum_a == 0 or sum_b == 0:
            return 0
        # both non-zero, both same |sum| context
        # prefer positive over negative
        if sum_a > 0 and sum_b < 0:
            return +1
        if sum_a < 0 and sum_b > 0:
            return -1
        return 0

    if not any(sums[i] == 0 for i in tied_seats):
        tied_seats, r = reduce_by(tied_seats, cmp_pos_neg, "positive")
        if r:
            record_tiebreak(counters, "positive_closest_to_zero")
            return tied_seats[0], (f"{chain} → {r}" if chain else r), False
        chain = "positive"


    # 2) Compare hand sizes (more cards wins).
    def cmp_cards(a,b):
        size_a = len(hands[a]); size_b = len(hands[b])
        if size_a > size_b: return +1
        if size_a < size_b: return -1
        return 0
    tied_seats, r = reduce_by(tied_seats, cmp_cards, "most cards")
    if r:
        record_tiebreak(counters, "cards_first")
        return tied_seats[0], (f"{chain} → {r}" if chain else r), False
    chain = "most cards"

    # 3) Sum of absolute values of all cards; higher sum wins.
    sums_abs = [sum(abs(c.v) for c in h) for h in hands]
    def cmp_sumabs(a,b):
        if sums_abs[a] > sums_abs[b]: return +1
        if sums_abs[a] < sums_abs[b]: return -1
        return 0
    tied_seats, r = reduce_by(tied_seats, cmp_sumabs, "highest Σ|cards|")
    if r:
        record_tiebreak(counters, "sum_abs")
        return tied_seats[0], (f"{chain} → {r}" if chain else r), False
    chain = "highest Σ|cards|"

    # 4) Unless --no-high-abs, compare absolute values lexicographically (desc).
    if opts["high_abs"]:
        abs_lists = {i: hand_abs_values_sorted_desc(hands[i]) for i in tied_seats}
        def cmp_highabs(a,b):
            return compare_lex_desc(abs_lists[a], abs_lists[b])
        tied_seats, r = reduce_by(tied_seats, cmp_highabs, "highest |card|")
        if r:
            record_tiebreak(counters, "high_abs")
            return tied_seats[0], (f"{chain} → {r}" if chain else r), False
        chain = "highest |card|"

    # 5) Compare positive cards lexicographically (desc).
    pos_lists = {i: hand_positives_sorted_desc(hands[i]) for i in tied_seats}
    def cmp_poslist(a,b):
        return compare_lex_desc(pos_lists[a], pos_lists[b])
    tied_seats, r = reduce_by(tied_seats, cmp_poslist, "highest positive card")
    if r:
        record_tiebreak(counters, "positive_cards")
        return tied_seats[0], (f"{chain} → {r}" if chain else r), False
    chain = "highest positive card"

    # 6) Unless --no-suits, a hand with all non-zero cards the same suit wins.
    if opts["use_suits"]:
        def cmp_suited(a,b):
            suited_a = hand_is_suited_nonzero(hands[a])
            suited_b = hand_is_suited_nonzero(hands[b])
            if suited_a and not suited_b: return +1
            if suited_b and not suited_a: return -1
            return 0
        tied_seats, r = reduce_by(tied_seats, cmp_suited, "suited")
        if r:
            record_tiebreak(counters, "suited")
            return tied_seats[0], (f"{chain} → {r}" if chain else r), False
        chain = "suited"

    # 8) True tie -> single-card draw tiebreak
    # Each of tied players draws one blind card in seat order; smallest |sum| wins; repeat if tie.
    # This mutates draw/discard piles.
    # Log into replay_trace if provided.
    # We count a "true tie" game in counters.true_ties_games
    while True:
        # ensure draw available; recycle if needed handled by caller during draw action
        # We'll implement a small helper here to draw with recycling (reusing the engine's function would be ideal, but we inline minimal logic).
        # This true-tie happens at the very end; we consider draw pile and discard.
        def draw_one() -> Optional[Card]:
            nonlocal draw, discard
            if draw:
                return draw.pop()
            # recycle policy: keep discard_top discard; shuffle rest into draw
            if len(discard) >= 2:
                discard_top = discard.pop()
                rng.shuffle(discard)
                draw = discard
                discard = [discard_top]
                # This is a reshuffle
                counters.reshuffles_total += 1
                # This is not necessarily during dice, so no RESHUFFLES_DICE
                return draw.pop() if draw else None
            return None

        # each candidate draws one
        drawn = {}
        for i in tied_seats:
            c = draw_one()
            if c is None:
                engine_abort("Bug: deck exhausted during single-card draw tiebreak.")
            drawn[i] = c

        # compare new |sum| including the drawn card
        newsabs = {}
        best_val = None
        bests = []
        for i in tied_seats:
            s = abs(sum(c.v for c in hands[i]) + drawn[i].v)
            newsabs[i] = s
            if best_val is None or s < best_val:
                best_val = s
                bests = [i]
            elif s == best_val:
                bests.append(i)

        # place drawn cards onto discard discard_top (visible)
        for i in tied_seats:
            discard.append(drawn[i])

        if len(bests) == 1:
            counters.true_ties_games += 1
            record_tiebreak(counters, "single_card_draw")
            return bests[0], (f"{chain} → draw" if chain else "draw"), True
        # else loop again

# =====================
# Game Engine
# =====================

def engine_recycle(state: GameState, during_dice: bool):
    """Recycle discard into draw when needed (preserve discard_top visible), count reshuffles, and log/announce."""
    draw = state["draw"]; discard = state["discard"]
    if len(discard) >= 2:
        discard_top = discard.pop()
        state["rng"].shuffle(discard)
        draw.extend(discard)
        discard.clear()
        discard.append(discard_top)
        state["counters"].reshuffles_total += 1
        if during_dice:
            state["counters"].reshuffles_dice += 1
            try:
                state.setdefault("dice_events", []).append("Draw Pile exhausted, reshuffling Discard Pile")
            except Exception:
                pass
        # log
        if state["record_replay"] and not during_dice:
            state["curr_round_moves"][-1].reshuffle_events.append("Draw Pile exhausted, reshuffling Discard Pile")
            try:
                if state.get("record_file_replay"):
                    state.setdefault("file_curr_round_moves", [])
                    if state["file_curr_round_moves"]:
                        state["file_curr_round_moves"][-1].setdefault("reshuffle_events", []).append("Draw Pile exhausted, reshuffling Discard Pile")
            except Exception:
                pass
        if state["interactive"] and not during_dice:
            print("Draw Pile exhausted, reshuffling Discard Pile")
    else:
        # impossible
        pass

def engine_abort(msg: str):
    print(msg)
    sys.exit(1)

def perform_human_action(state: GameState, seat, action_label):
    hand = state["hands"][seat]
    discard = state["discard"]; draw = state["draw"]
    if action_label == "stand":
        if state["interactive"]:
            print(f"  -> stand")
        return "stand"
    elif action_label == "draw":
        if not draw:
            state["recycle"](state, during_dice=False)
        if not draw:
            state["abort"]("Draw impossible.")
        c = draw.pop()
        hand.append(c)
        named_hand = named_detect_active(hand, NAMED_ORDERS[state["opts"]["named_order"]])
        named_label = f" [{named_hand[1]}]" if named_hand else ""
        print(f"  -> draw: + [{c}]  |  new hand: {fmt_hand_positions(hand)}   (sum={hand_sum(hand)}){named_label}")
        return "draw"
    elif action_label == "discard_draw":
        # prompt for index was already done; index stored in state["human_index"]
        idx = state["human_index"]
        if idx is None or idx < 0 or idx >= len(hand):
            state["abort"]("Invalid discard index.")
        c = hand.pop(idx)
        discard.append(c)
        known_remove(state, c)
        if not draw:
            state["recycle"](state, during_dice=False)
        if not draw:
            state["abort"]("Draw impossible.")
        d = draw.pop()
        hand.append(d)
        named_hand = named_detect_active(hand, NAMED_ORDERS[state["opts"]["named_order"]])
        named_label = f" [{named_hand[1]}]" if named_hand else ""
        print(f"  -> discard [{c}], draw + [{d}]  |  new hand: {fmt_hand_positions(hand)}   (sum={hand_sum(hand)}){named_label}")
        return "discard_draw"
    elif action_label == "swap":
        idx = state["human_index"]
        if idx is None or idx < 0 or idx >= len(hand):
            state["abort"]("Invalid swap index.")
        if not discard:
            state["abort"]("No discard to swap with.")
        discard_top = discard[-1]
        swapped_out = hand[idx]
        hand[idx] = discard_top
        discard[-1] = swapped_out
        known_add(state, seat, discard_top)
        known_remove(state, swapped_out)
        named_hand = named_detect_active(hand, NAMED_ORDERS[state["opts"]["named_order"]])
        named_label = f" [{named_hand[1]}]" if named_hand else ""
        print(f"  -> swap: took [{discard_top}], discarded [{swapped_out}]  |  new hand: {fmt_hand_positions(hand)}   (sum={hand_sum(hand)}){named_label}")
        return "swap"
    elif action_label == "gain_discard":
        if not state["opts"].get("allow_discard_gain"):
            state["abort"]("Discard-gain not enabled.")
        if not discard:
            state["abort"]("No discard to gain from.")
        c = discard.pop()
        hand.append(c)
        known_add(state, seat, c)  # everyone saw you take this exact card
        ensure_discard_after_gain(state)
        named_hand = named_detect_active(hand, NAMED_ORDERS[state["opts"]["named_order"]])
        named_label = f" [{named_hand[1]}]" if named_hand else ""
        print(f"  -> gain-from-discard: + [{c}]  |  new hand: {fmt_hand_positions(hand)}   (sum={hand_sum(hand)}){named_label}")
        new_discard_top = state.pop("dealer_started_new_discard", None)
        if new_discard_top is not None and state["interactive"]:
            print(f"Dealer starts new discard with [{new_discard_top}]")
            pause_after_action()
        return "gain_discard"

    else:
        state["abort"]("Unknown human action.")

def play_one_game(state: GameState) -> dict:
    """Run a single game, return per-game report including replay snippet data."""
    rng = state["rng"]
    opts = state["opts"]
    num_players = opts["num_players"]
    # Build deck and starts for this game depending on starts mode
    # Determine starting values for each seat for this game
    starting_hands = []
    if opts["randomize_all"]:
        # deal new random starts each game
        # we'll remove values from a fresh deck below; but we need just values for reporting
        # We'll sample from full counts without replacement per value assignment
        counts = full_value_counts()
        for _ in range(num_players):
            # draw two values uniformly from remaining multiset
            # We'll build a temp deck and pop discard_top, but simpler: build full deck and rng.shuffle
            # For clarity, we will deal from a fresh deck at deal time; here we just placeholder None; real starts assigned below.
            starting_hands.append(None)
    elif state["fixed_starts"]:
        # hybrid: fixed seats as given; others random per game
        starting_hands = [None]*num_players
        for seat_idx, vals in state["fixed_starts"].items():
            if 1 <= seat_idx <= num_players:
                starting_hands[seat_idx-1] = vals[:]
    else:
        # constant starts per run
        starting_hands = [vals[:] for vals in state["starts_run_vals"]]

    # Build live deck
    deck = build_full_deck()
    # Remove starting values (choose suits randomly)
    # For randomize_all or hybrid randomize_others, we need to generate random starts now by popping from deck.
    hands = [[] for _ in range(num_players)]
    if opts["randomize_all"]:
        # shuffle deck, then deal two cards per player by popping discard_top
        rng.shuffle(deck)
        for i in range(num_players):
            c1 = deck.pop(); c2 = deck.pop()
            hands[i].append(c1); hands[i].append(c2)
            starting_hands[i] = [c1.v, c2.v]
    else:
        # For starting_hands entries that are None -> randomize (hybrid)
        # First, remove fixed/known values from deck
        known_vals = []
        for vals in starting_hands:
            if vals is not None:
                known_vals.extend(vals)
        # Remove known values with suits chosen randomly
        remove_values_from_deck(rng, deck, known_vals)
        rng.shuffle(deck)
        # Fixed-start cards get a randomly chosen suit.  Suits are only relevant
        # for the suited tiebreaker (disabled via --no-suits), so this approximation is fine.
        for i in range(num_players):
            if starting_hands[i] is not None:
                v1, v2 = starting_hands[i]
                hands[i].append(Card(v1, None if v1==0 else rng.choice(SUITS)))
                hands[i].append(Card(v2, None if v2==0 else rng.choice(SUITS)))
        # For the remaining seats, deal from deck
        for i in range(num_players):
            if starting_hands[i] is None:
                c1 = deck.pop(); c2 = deck.pop()
                hands[i].append(c1); hands[i].append(c2)
                starting_hands[i] = [c1.v, c2.v]

    # Face-up discard: pop discard_top
    discard = []
    first_discard = deck.pop()
    discard.append(first_discard)
    draw = deck  # remaining are draw pile

    # Expose piles and hands in state for policies
    state["hands"] = hands
    state["discard"] = discard
    state["draw"] = draw

    # Prepare modes for this game
    seat_modes = []
    for i in range(num_players):
        m = state["modes_run"][i]
        if m == "newmodes":
            # Randomize only among the new simple bots
            seat_modes.append(rng.choice(["minimizer","swaptozero","drawtozero"]))
        else:
            # Deterministic: use the mode as provided
            seat_modes.append(m)
 

    # Rotate first seat if option enabled
    start_seat = state["start_seat_index"]
    order = list(range(num_players))
    if opts["rotate_first"]:
        order = order[start_seat:] + order[:start_seat]

    
    # For replay
    record_replay = state["record_replay"]
    record_file_replay = state.get("record_file_replay", False)
    rounds_moves = []
    dice_rolls = []
    if record_replay:
        state["curr_round_moves"] = []  # will be replaced per round
    if record_file_replay:
        file_rounds = []
        file_dice_rolls = []


    # Live "full discard" for public counts includes buried cards too
    state["order"] = order
    state["discard_full"] = discard  # reference to same list

    # Print game banner for interactive play
    if state["interactive"]:
        game_num = state.get("game_num", "?")
        modes_display = ", ".join(
            f"P{i+1}: {'You' if (opts['human_player'] and i == opts['human_player']-1) else str(seat_modes[i])}"
            for i in range(num_players)
        )
        print(f"\n{'='*50}")
        print(f"  Game {game_num}   |   {modes_display}")
        print(f"  First face-up discard: {first_discard.v}")
        print(f"{'='*50}")

    # 3 rounds
    dice_stats = DiceStats()
    for round_num in range(1,4):
        state["round_index"] = round_num
        if record_replay:
            state["curr_round_moves"] = []
        if state.get("record_file_replay"):
            state["file_curr_round_moves"] = []

        if state["interactive"]:
            print(f"\n--- Round {round_num} ---")

        # Each player acts in order
        for seat in order:
            # Human?
            is_human = (opts["human_player"] is not None and (seat == opts["human_player"]-1))
            mode = seat_modes[seat]
            # trace before
            before_vals = hand_values(hands[seat])[:]
            before_suits = [c.s for c in hands[seat]]

            if is_human and state["interactive"]:
                # Prompt
                print(f"Your turn (Game {state.get('game_num', '?')}) (Round {round_num}):")
                nd = named_detect_active(hands[seat], NAMED_ORDERS[opts['named_order']])
                print(f"  Your hand:                 {fmt_hand_positions(hands[seat])}   (sum={hand_sum(hands[seat])})" + (f" [{nd[1]}]" if nd else ""))
                print(f"  Top of discard:            {discard[-1].v if discard else 'None'}")
                print(f"  Draw pile size: {len(draw)}")
                # print("  Actions: [s]tand, [d]raw, [x] discard+draw, s[w]ap-with-discard, [exit]")
                
                extra = " [g] gain from discard," if opts.get("allow_discard_gain") else ""
                print(f"  Actions: [s] stand, [d] draw, [x] discard+redraw, [w] swap with discard,{extra} [exit]")
                prompt = "  Choose action (s/d/x/w" + ("/g" if opts.get("allow_discard_gain") else "") + "/exit): "
                while True:
                    choice = input(prompt).strip().lower()
                    if choice == "exit":
                        print("Exiting.")
                        sys.exit(0)
                    if choice == "s":
                        act = "stand"; state["human_index"] = None; break
                    elif choice == "d":
                        act = "draw"; state["human_index"] = None; break
                    elif choice == "x":
                        # ask index
                        idx = None
                        while True:
                            try:
                                idx = int(input(f"  Discard which card [1..{len(hands[seat])}]? ").strip()) - 1
                                if 0 <= idx < len(hands[seat]):
                                    break
                            except:
                                pass
                            print("  Invalid index.")
                        state["human_index"] = idx
                        act = "discard_draw"; break
                    elif choice == "w":
                        if not discard:
                            print("  No discard to swap with; standing.")
                            act = "stand"; state["human_index"] = None; break
                        else:
                            idx = None
                            while True:
                                try:
                                    idx = int(input(f"  Swap which card [1..{len(hands[seat])}]? ").strip()) - 1
                                    if 0 <= idx < len(hands[seat]):
                                        break
                                except:
                                    pass
                                print("  Invalid index.")
                            state["human_index"] = idx
                            act = "swap"; break
                    elif choice == "g":
                        if not opts.get("allow_discard_gain"):
                            print("  Discard-gain not enabled.")
                            continue
                        act = "gain_discard"; state["human_index"] = None; break

                    else:
                        print(f"  Invalid choice. Options: s (stand), d (draw), x (discard+redraw), w (swap with discard),{extra} or exit.")
                        continue
                # Execute
                act_label = perform_human_action(state, seat, act)
                if record_replay:
                    state["curr_round_moves"].append(ReplayMove(seat, "Human", before_vals, act_label, hand_values(hands[seat])[:]))
                if state.get("record_file_replay"):
                    after_vals = hand_values(hands[seat])[:]
                    after_suits = [c.s for c in hands[seat]]
                    state["file_curr_round_moves"].append({
                        "player_index": seat,
                        "mode_label": "Human",
                        "before": before_vals,
                        "before_suits": before_suits,
                        "action": act_label,
                        "after": after_vals,
                        "after_suits": after_suits,
                        "reshuffle_events": [],
                    })
                pause_after_action()
            else:
                # Bot
                prev_discard_top = discard[-1].v if discard else None
                # Select policy implementation
                if mode == "draw":
                    act_label = _bot_safe_draw(state, seat)
                elif mode == "stand":
                    act_label = "stand"
                elif mode == "drawtozero":
                    act_label = _drawtozero_core(state, seat)
                elif mode == "minimizer":
                    act_label = _minimizer_core(state, seat)
                elif mode == "swaptozero":
                    act_label = _swaptozero_core(state, seat)
                elif (isinstance(mode, tuple) and mode and mode[0] == "hunter") or mode == "hunter":
                    act_label = _hunter_core(state, seat)
                else:
                    act_label = _drawtozero_core(state, seat)


                # Announce
                if state["interactive"]:
                    if act_label == "swap":
                        took = prev_discard_top if prev_discard_top is not None else "?"
                        disc = (discard[-1].v if discard else "?")
                        print(f"P{seat+1} ({mode}) swapped with discard (took {took}, discarded {disc}); hand size now {len(hands[seat])}")
                    elif act_label == "discard_draw":
                        disc = (discard[-1].v if discard else "?")
                        print(f"P{seat+1} ({mode}) discarded {disc} and drew (blind); hand size now {len(hands[seat])}")
                    elif act_label == "draw":
                        print(f"P{seat+1} ({mode}) drew (blind); hand size now {len(hands[seat])}")
                    elif act_label == "gain_discard":
                        took = prev_discard_top if prev_discard_top is not None else "?"
                        print(f"P{seat+1} ({mode}) took from discard {took}; hand size now {len(hands[seat])}")
                        new_discard_top = state.pop("dealer_started_new_discard", None)
                        if new_discard_top is not None:
                            print(f"Dealer starts new discard with [{new_discard_top}]")
                            
                    elif act_label == "stand":
                        print(f"P{seat+1} ({mode}) stood; hand size now {len(hands[seat])}")

                # Known-in-hand maintenance: if a swap taken from discard discard_top -> that specific card is now known in hand

                # Replay record
                if record_replay:
                    state["curr_round_moves"].append(ReplayMove(seat, str(mode), before_vals, act_label, hand_values(hands[seat])[:]))
                if state.get("record_file_replay"):
                    after_vals = hand_values(hands[seat])[:]
                    after_suits = [c.s for c in hands[seat]]
                    state["file_curr_round_moves"].append({
                        "player_index": seat,
                        "mode_label": str(mode),
                        "before": before_vals,
                        "before_suits": before_suits,
                        "action": act_label,
                        "after": after_vals,
                        "after_suits": after_suits,
                        "reshuffle_events": [],
                    })

                if state["interactive"]:
                    pause_after_action()

        # Dice phase
        if not opts["no_dice"]:
            state["dice_events"] = []

            if state["interactive"] and opts["dice_mode"] == "spike" and opts.get("spike_cheat"):
                # Ask every dice phase what to do
                while True:
                    try:
                        val = int(input("  Spike cheat — enter 1..6 for doubles, or 0 for NO double: ").strip())
                    except Exception:
                        val = -1
                    if 0 <= val <= 6:
                        break
                    print("  Please enter a number 0..6.")
                if val == 0:
                    # Force a non-double
                    d1 = rng.randint(1,6)
                    d2 = rng.randint(1,6)
                    if d2 == d1:
                        d2 = 1 + (d1 % 6)
                else:
                    # Force doubles of the chosen pip
                    d1 = val
                    d2 = val
            else:
                d1 = rng.randint(1,6)
                d2 = rng.randint(1,6)

            is_doubles = (d1 == d2)
            is_spike_double = (opts["dice_mode"] == "spike" and d1 == d2 and d1 == 1)
            redraw_sizes = None
            if is_doubles:
                dice_stats.doubles_in_game += 1
            if is_spike_double:
                dice_stats.spike_doubles_in_game += 1
            # Printing interactive/replay messages
            if state["interactive"]:
                if opts["dice_mode"] == "spike" and is_doubles and d1 == 1:
                    print(f"Dice: {d1} + {d2} -- SPIKE DOUBLES! Everyone discards and redraws")
                elif is_doubles and opts["dice_mode"] == "classic":
                    print(f"Dice: {d1} + {d2} -- DOUBLES! Everyone discards and redraws")
                elif is_doubles:
                    if opts["dice_mode"] == "spike":
                        print(f"Dice: {d1} + {d2} -- DOUBLES! Forced discard+draw for each player")
                    else:
                        print(f"Dice: {d1} + {d2} -- DOUBLES! Everyone discards and redraws")
                else:
                    print(f"Dice: {d1} + {d2}")
                pause_after_action()
            else:
                # For replay later we just record rolls; template prints in replay
                pass

            # Apply behavior
            if is_doubles:
                if opts["dice_mode"] == "classic" or (opts["dice_mode"] == "spike" and d1 == 1):
                    # full-hand wipe/redraw
                    redraw_sizes = [len(h) for h in hands]
                    # everyone dumps to discard
                    known_clear(state)
                    for i in range(num_players):
                        while hands[i]:
                            discard.append(hands[i].pop())
                    # redraw same counts in seat order
                    for i in range(num_players):
                        need = redraw_sizes[i]
                        for _ in range(need):
                            if not draw:
                                engine_recycle(state, during_dice=True)
                            if not draw:
                                state["abort"]("Draw impossible during wipe redraw.")
                            hands[i].append(draw.pop())
                    # after redraw, print sizes in interactive mode
                    if state["interactive"] and redraw_sizes:
                        print(f"    (Redraw sizes: {redraw_sizes})")

                else:
                    # spike: forced discard_draw per player (non-snake-eyes doubles)
                    # Human must choose their discard; bots will show which card they discard.
                    for i in order:
                        if hands[i]:
                            is_human = (opts["human_player"] is not None and (i == opts["human_player"]-1))
                            if is_human and state["interactive"]:
                                print("DOUBLES: forced discard+draw your turn")
                                print(f"  Your hand:                 {fmt_hand_positions(hands[i])}   (sum={hand_sum(hands[i])})")
                                idx = None
                                while True:
                                    try:
                                        idx = int(input(f"  Discard which card [1..{len(hands[i])}]? ").strip()) - 1
                                        if 0 <= idx < len(hands[i]):
                                            break
                                    except Exception:
                                        pass
                                    print("  Invalid index.")
                                c = hands[i].pop(idx)
                                discard.append(c)
                                if not draw:
                                    engine_recycle(state, during_dice=True)
                                if not draw:
                                    state["abort"]("Draw impossible during spike forced replacement.")
                                dcard = draw.pop()
                                hands[i].append(dcard)
                                named_hand = named_detect_active(hands[i], NAMED_ORDERS[opts["named_order"]])
                                named_label = f" [{named_hand[1]}]" if named_hand else ""
                                print(f"  -> discard [{c}], draw + [{dcard}]  |  new hand: {fmt_hand_positions(hands[i])}   (sum={hand_sum(hands[i])}){named_label}")
                            else:
                                # Bot forced discard+draw: choose index by helper (damage control) and print the discarded card
                                idx = forced_discard_draw_index_min_abs(state, i, rng, mode_label=seat_modes[i])

                                c = hands[i].pop(idx)
                                discarded_val = c.v
                                discard.append(c)
                                if not draw:
                                    engine_recycle(state, during_dice=True)
                                if not draw:
                                    state["abort"]("Draw impossible during spike forced replacement.")
                                hands[i].append(draw.pop())
                                if state["interactive"]:
                                    mode_lbl = seat_modes[i]
                                    print(f"P{i+1} ({mode_lbl}) forced discard {discarded_val} and drew (blind); hand size now {len(hands[i])}")


            if state["interactive"] and state.get("dice_events"):
                for evt in state["dice_events"]:
                    print(f"    {evt}")
            if record_replay:
                dice_rolls.append(DiceRoll(d1, d2, redraw_sizes, is_doubles, is_spike_double, list(state.get("dice_events", []))))
            if state.get("record_file_replay"):
                file_dice_rolls.append(DiceRoll(d1, d2, redraw_sizes, is_doubles, is_spike_double, list(state.get("dice_events", []))))

        # end round
        if record_replay:
            rounds_moves.append(state["curr_round_moves"])
        if state.get("record_file_replay"):
            file_rounds.append(state.get("file_curr_round_moves", []))

    # Final evaluation
    winner_i, reason, used_true_tie = winner_and_reason(hands, opts, rng, discard, draw, state["counters"], None)

    # Build per-game result
    
    res = {
        "modes": seat_modes,
        "starts": starting_hands,
        "first_discard": first_discard.v,
        "finals": [(hand_values(h), hand_sum(h), (lambda nd: nd[1] if nd else None)(named_detect_active(h, NAMED_ORDERS[opts["named_order"]]))) for h in hands],
        "winner": winner_i,
        "winner_reason": reason.strip(),
        "dice_stats": (dice_stats.doubles_in_game, dice_stats.spike_doubles_in_game),
        "rounds_moves": rounds_moves,
        "dice_rolls": dice_rolls,
        "used_true_tie": used_true_tie,
    }
    # Write to replay file (with suits) if requested
    if state.get("replay_writer") and state["replay_writer"].will_write(state.get("game_num", 0) or 0):
        file_data = {
            "modes": seat_modes,
            "starts": starting_hands,
            "starts_suits": [[c.s for c in hands[i][:2]] for i in range(num_players)],
            "first_discard": {"value": first_discard.v, "suit": first_discard.s},
            "rounds_moves": file_rounds,
            "dice_rolls": file_dice_rolls,
            "finals": [{"values": hand_values(h), "suits": [c.s for c in h], "sum": hand_sum(h), "name": (lambda nd: nd[1] if nd else None)(named_detect_active(h, NAMED_ORDERS[opts['named_order']]))} for h in hands],
            "winner": winner_i,
            "winner_reason": reason.strip(),
            "doubles_in_game": dice_stats.doubles_in_game,
            "spike_doubles_in_game": dice_stats.spike_doubles_in_game,
        }
        try:
            state["replay_writer"].write_game(state.get("game_num", 0) or 0, file_data)
        except Exception as _e:
            # Non-fatal
            pass
    return res

