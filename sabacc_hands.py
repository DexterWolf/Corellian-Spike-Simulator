from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from sabacc_cards import Card, hand_values, hand_sum

# =====================
# Starting Hand Parsing
# =====================

def parse_starts_string(s: str) -> Dict[int, List[int]]:
    """Parse "p1:a,b;p3:0,10;..." into {1:[a,b],3:[0,10],...}"""
    res = {}
    if not s.strip():
        return res
    parts = s.split(";")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Malformed starts segment: {part}")
        left, right = part.split(":", 1)
        left = left.strip().lower()
        if not left.startswith("p"):
            raise ValueError(f"Starts seat must be 'pN': {left}")
        try:
            seat = int(left[1:])
        except:
            raise ValueError(f"Starts seat malformed: {left}")
        vals = [x.strip() for x in right.split(",")]
        if len(vals) != 2:
            raise ValueError(f"Starts for {left} must have exactly 2 values.")
        try:
            v1 = int(vals[0]); v2 = int(vals[1])
        except:
            raise ValueError(f"Non-integer value in starts: {right}")
        if not (-10 <= v1 <= 10 and -10 <= v2 <= 10):
            raise ValueError(f"Values out of range in starts for {left}: {right}")
        res[seat] = [v1, v2]
    return res

def parse_fixed_starts_string(s: str) -> Dict[int, List[int]]:
    return parse_starts_string(s)

# =====================
# Named Hand Detection
# =====================

# Utility: count by abs value
def abs_counter(hand: List[Card]) -> Counter:
    return Counter(abs(c.v) for c in hand)

def sign_counts_by_abs(hand: List[Card]) -> Dict[int, Counter]:
    """Return {abs_val: Counter({+1: count_pos, -1: count_neg})}"""
    d = defaultdict(Counter)
    for c in hand:
        if c.v == 0:
            continue
        a = abs(c.v)
        sgn = 1 if c.v > 0 else -1
        d[a][sgn] += 1
    return d

def consecutive_abs_values(vals: List[int]) -> bool:
    """Distinct abs values are consecutive (e.g., 3,4,5,6)"""
    if not vals:
        return False
    sv = sorted(set(vals))
    if len(sv) != len(vals):
        return False
    lo, hi = min(sv), max(sv)
    return hi - lo + 1 == len(vals)

@dataclass
class NamedResult:
    ok: bool
    name: str = ""
    key: Optional[List[int]] = None
    rank: Optional[int] = None  # index in active order (0 best)

def is_sum_zero(hand: List[Card]) -> bool:
    return hand_sum(hand) == 0

# Named hand detectors (sum==0 required)
def detect_pure_sabacc(hand: List[Card]) -> NamedResult:
    """Exactly two Sylops (0, 0) — the rarest possible Sabacc."""
    if hand_sum(hand) != 0:
        return NamedResult(False)
    card_values = sorted(hand_values(hand))
    if card_values == [0,0] and len(card_values) == 2:
        return NamedResult(True, "Pure Sabacc", [0])
    return NamedResult(False)

def detect_full_sabacc(hand: List[Card]) -> NamedResult:
    """5 cards: two +10s, two −10s, one Sylop."""
    if hand_sum(hand) != 0 or len(hand) != 5:
        return NamedResult(False)
    card_values = sorted(hand_values(hand))
    if card_values.count(10) == 2 and card_values.count(-10) == 2 and card_values.count(0) == 1:
        return NamedResult(True, "Full Sabacc", [10])
    return NamedResult(False)

def detect_fleet(hand: List[Card]) -> NamedResult:
    """5 cards: four copies of the same |value| plus one Sylop."""
    if hand_sum(hand) != 0 or len(hand) != 5:
        return NamedResult(False)
    if 0 not in hand_values(hand):
        return NamedResult(False)
    zero_ct = hand_values(hand).count(0)
    if zero_ct != 1:
        return NamedResult(False)
    # four of the same abs among non-sabacc_seats
    non_zero_cards = [c for c in hand if c.v != 0]
    non_zero_counts = abs_counter(non_zero_cards)
    if len(non_zero_cards) == 4 and any(v == 4 for v in non_zero_counts.values()):
        a = max(k for k,v in non_zero_counts.items() if v == 4)
        return NamedResult(True, "Fleet", [a])
    return NamedResult(False)

def detect_squadron(hand: List[Card]) -> NamedResult:
    """4 cards: four copies of the same |value|, no Sylops."""
    if hand_sum(hand) != 0 or len(hand) != 4:
        return NamedResult(False)
    if any(c.v == 0 for c in hand):
        return NamedResult(False)
    value_counts = abs_counter(hand)
    if any(v == 4 for v in value_counts.values()):
        a = max(k for k,v in value_counts.items() if v == 4)
        return NamedResult(True, "Squadron", [a])
    return NamedResult(False)

def detect_rhylet(hand: List[Card]) -> NamedResult:
    """5 cards: a strict triplet (all same sign) and pair (opposite sign, different |value|)."""
    if hand_sum(hand) != 0 or len(hand) != 5:
        return NamedResult(False)
    # three of one sign same abs + two of the opposite sign same abs
    sign_counts = sign_counts_by_abs(hand)  # {abs_val: Counter({+1: x, -1: y})}
    # counts must be concentrated in exactly two abs values with totals 3 and 2
    totals = {a: (cnt[+1] + cnt[-1]) for a, cnt in sign_counts.items()}
    if sorted(totals.values()) != [2, 3]:
        return NamedResult(False)

    triplet_abs = max(a for a, t in totals.items() if t == 3)
    pair_abs = max(a for a, t in totals.items() if t == 2)

    # STRICT sign conditions:
    triplet_signs = sign_counts[triplet_abs]  # Counter for the abs with total 3
    pair_signs = sign_counts[pair_abs]  # Counter for the abs with total 2

    # triplet must be all + or all -
    if not ((triplet_signs[+1] == 3 and triplet_signs[-1] == 0) or (triplet_signs[-1] == 3 and triplet_signs[+1] == 0)):
        return NamedResult(False)
    # pair must be all + or all -
    if not ((pair_signs[+1] == 2 and pair_signs[-1] == 0) or (pair_signs[-1] == 2 and pair_signs[+1] == 0)):
        return NamedResult(False)
    # and the signs must be opposite between the triplet and the pair
    triplet_sign = +1 if triplet_signs[+1] == 3 else -1
    pair_sign    = +1 if pair_signs[+1] == 2 else -1
    if triplet_sign == pair_sign:
        return NamedResult(False)

    return NamedResult(True, "Rhylet", [max(triplet_abs, pair_abs), min(triplet_abs, pair_abs)])

def detect_lesser_rhylet(hand: List[Card]) -> NamedResult:
    """5 cards: triplet + pair by |value| with mixed signs allowed (Wild Rhylet)."""
    if hand_sum(hand) != 0 or len(hand) != 5:
        return NamedResult(False)
    value_counts = abs_counter(hand)
    if sorted(value_counts.values()) == [2,3]:
        triplet_abs = max(a for a,v in value_counts.items() if v==3)
        pair_abs = max(a for a,v in value_counts.items() if v==2)
        return NamedResult(True, "Wild Rhylet", [max(triplet_abs,pair_abs), min(triplet_abs,pair_abs)])
    return NamedResult(False)

def detect_straight_khyron_base(hand: List[Card], need_zero: bool) -> NamedResult:
    """Shared logic for Straight Khyron variants; need_zero=True adds a required Sylop."""
    if hand_sum(hand) != 0:
        return NamedResult(False)
    card_values = [c.v for c in hand]
    if need_zero:
        if card_values.count(0) != 1 or len(hand) != 5:
            return NamedResult(False)
        abs_values = [abs(v) for v in card_values if v!=0]
        if len(abs_values) != 4:
            return NamedResult(False)
        if len(set(abs_values)) != 4:
            return NamedResult(False)
        if not consecutive_abs_values(abs_values):
            return NamedResult(False)
        key_values = sorted(set(abs_values), reverse=True)
        return NamedResult(True, "Sylop Straight Khyron", key_values)
    else:
        if any(v==0 for v in card_values):
            return NamedResult(False)
        if len(hand) != 4:
            return NamedResult(False)
        abs_values = [abs(v) for v in card_values]
        if len(set(abs_values)) != 4:
            return NamedResult(False)
        if not consecutive_abs_values(abs_values):
            return NamedResult(False)
        key_values = sorted(set(abs_values), reverse=True)
        return NamedResult(True, "Straight Khyron", key_values)

def detect_straight_khyron(hand: List[Card]) -> NamedResult:
    """4 cards: four consecutive |values|, no Sylops."""
    return detect_straight_khyron_base(hand, need_zero=False)

def detect_sylop_straight_khyron(hand: List[Card]) -> NamedResult:
    """5 cards: four consecutive |values| plus one Sylop."""
    return detect_straight_khyron_base(hand, need_zero=True)

def detect_gee_whiz(hand: List[Card]) -> NamedResult:
    """5 cards: ±10 paired with the four cards of the opposite sign {1,2,3,4}."""
    if hand_sum(hand) != 0 or len(hand) != 5:
        return NamedResult(False)
    card_values = sorted(hand_values(hand))
    if card_values == [-4, -3, -2, -1, 10] or card_values == [-10, 1, 2, 3, 4]:
        return NamedResult(True, "Gee Whiz!", [10,4,3,2,1])
    return NamedResult(False)

def detect_five_card_straight(hand: List[Card]) -> NamedResult:
    """5 cards: five consecutive |values|, no Sylops (Full Straight)."""
    if hand_sum(hand) != 0 or len(hand) != 5:
        return NamedResult(False)
    if any(c.v == 0 for c in hand):
        return NamedResult(False)
    abs_values = [abs(c.v) for c in hand]
    if len(set(abs_values)) != 5:
        return NamedResult(False)
    if not consecutive_abs_values(abs_values):
        return NamedResult(False)
    key_values = sorted(set(abs_values), reverse=True)
    return NamedResult(True, "Full Straight", key_values)

def detect_five_card_squad(hand: List[Card]) -> NamedResult:
    """5 cards: four of the same |value| plus one other card, no Sylops (Five Card Squad)."""
    # sum==0 implies the 5th card has double that abs with a matching sign-count structure.
    if hand_sum(hand) != 0 or len(hand) != 5:
        return NamedResult(False)
    if any(c.v == 0 for c in hand):
        return NamedResult(False)
    value_counts = abs_counter(hand)
    if 4 not in value_counts.values():
        return NamedResult(False)
    a = max(k for k,v in value_counts.items() if v==4)
    return NamedResult(True, "Five Card Squad", [a])

def detect_rule_of_two(hand: List[Card]) -> NamedResult:
    """4–5 cards: exactly two abs-value pairs (Sylops may be among the non-pair card)."""
    if hand_sum(hand) != 0:
        return NamedResult(False)

    # Allow sabacc_seats to participate as a normal abs value (0).
    value_counts = abs_counter(hand)
    count_values = list(value_counts.values())

    # 4 cards: exactly two abs values, each occurring twice
    if len(hand) == 4 and sorted(count_values) == [2, 2]:
        # Exclude the special case: {0,0,+a,-a} which is now Idiots Rule
        if value_counts.get(0, 0) == 2:
            return NamedResult(False)

        pairs = sorted([a for a, v in value_counts.items() if v == 2], reverse=True)
        if len(pairs) == 2:
            return NamedResult(True, "Rule of Two", pairs)

    # 5 cards: two abs-value pairs plus one extra card (any abs, including 0)
    if len(hand) == 5 and sorted(count_values) == [1, 2, 2]:
        pairs = sorted([a for a, v in value_counts.items() if v == 2], reverse=True)
        if len(pairs) == 2:
            return NamedResult(True, "Rule of Two", pairs)

    return NamedResult(False)


def detect_idiots_rule(hand: List[Card]) -> NamedResult:
    """
    Idiots Rule:
      4 cards, sum == 0,
      exactly two Sylops (sabacc_seats) and a balanced pair (+a, -a).
      Pattern in values: multiset {0, 0, +a, -a}.
    """
    if hand_sum(hand) != 0 or len(hand) != 4:
        return NamedResult(False)

    card_values = hand_values(hand)
    if card_values.count(0) != 2:
        return NamedResult(False)

    nonzero = [v for v in card_values if v != 0]
    if len(nonzero) != 2:
        return NamedResult(False)

    # must be +a and -a
    if nonzero[0] + nonzero[1] != 0:
        return NamedResult(False)
    if abs(nonzero[0]) != abs(nonzero[1]):
        return NamedResult(False)

    pair_val = abs(nonzero[0])
    return NamedResult(True, "Idiots Rule", [pair_val])


def detect_sylop_rule_of_two(hand: List[Card]) -> NamedResult:
    """5 cards: exactly one Sylop plus two abs-value pairs among the non-zero cards."""
    if hand_sum(hand) != 0 or len(hand) != 5:
        return NamedResult(False)
    if hand_values(hand).count(0) != 1:
        return NamedResult(False)
    non_zero_cards = [c for c in hand if c.v != 0]
    value_counts = abs_counter(non_zero_cards)
    if sorted(value_counts.values()) == [2,2]:
        pairs = sorted([a for a,v in value_counts.items() if v==2], reverse=True)
        return NamedResult(True, "Sylop Rule of Two", pairs)
    return NamedResult(False)

def detect_banthas_wild(hand: List[Card]) -> NamedResult:
    """3–5 cards: a triplet by |value| plus 1–2 other distinct |values| (may include a Sylop)."""
    if hand_sum(hand) != 0:
        return NamedResult(False)
    value_counts = abs_counter(hand)
    # need a triplet by abs plus one or two other values; one other may be 0
    if 3 not in value_counts.values():
        return NamedResult(False)
    trip_val = max(a for a,v in value_counts.items() if v==3)
    # Gather other values (by abs), including 0
    others = []
    seen = set()
    for c in hand:
        a = abs(c.v)
        if a == trip_val:
            continue
        if a not in seen:
            seen.add(a)
            others.append(a)
    # Banthas Wild allows 1 or 2 others
    if len(others) not in (1,2):
        return NamedResult(False)
    others_sorted = sorted([o for o in others if o != 0], reverse=True)
    if 0 in others:
        others_sorted.append(0)
    return NamedResult(True, "Banthas Wild", [trip_val])

def detect_yee_haa(hand: List[Card]) -> NamedResult:
    """3 cards: a balanced pair (+a, −a) plus one Sylop."""
    # 1) Only zero-sum hands can be named
    if hand_sum(hand) != 0:
        return NamedResult(False)

    card_values = hand_values(hand)

    # Yee-Haa is now ONLY the 3-card pattern:
    #   pair (by abs, opposite signs) + one zero  -> e.g. (10, -10, 0)
    if len(card_values) != 3:
        return NamedResult(False)

    if card_values.count(0) != 1:
        return NamedResult(False)

    nonzero = [v for v in card_values if v != 0]
    if len(nonzero) != 2:
        return NamedResult(False)

    # Must be a balanced pair: +a and -a
    if nonzero[0] + nonzero[1] != 0:
        return NamedResult(False)
    if abs(nonzero[0]) != abs(nonzero[1]):
        return NamedResult(False)

    pair_val = abs(nonzero[0])
    return NamedResult(True, "Yee-Haa", [pair_val])




def detect_pair(hand: List[Card]) -> NamedResult:
    """Any zero-sum hand containing at least one abs-value pair among the non-zero cards."""
    # Only zero-sum hands can be named
    if hand_sum(hand) != 0:
        return NamedResult(False)

    # Ignore Sylops as pair candidates; they are fillers here.
    non_zero_cards = [c for c in hand if c.v != 0]
    value_counts = abs_counter(non_zero_cards)  # counts by abs value of non-zero cards

    # Require at least one non-zero abs-value pair
    if not any(v >= 2 for v in value_counts.values()):
        return NamedResult(False)

    pair_val = max(a for a, v in value_counts.items() if v >= 2)
    return NamedResult(True, "Pair", [pair_val])



# Registry of checkers (name -> function)

NAMED_CHECKERS = {
    "Pure Sabacc": detect_pure_sabacc,
    "Full Sabacc": detect_full_sabacc,
    "Fleet": detect_fleet,
    "Squadron": detect_squadron,
    "Rhylet": detect_rhylet,
    "Wild Rhylet": detect_lesser_rhylet,
    "Straight Khyron": detect_straight_khyron,
    "Sylop Straight Khyron": detect_sylop_straight_khyron,
    "Gee Whiz!": detect_gee_whiz,
    "Full Straight": detect_five_card_straight,
    "Five Card Squad": detect_five_card_squad,
    "Rule of Two": detect_rule_of_two,
    "Sylop Rule of Two": detect_sylop_rule_of_two,
    "Banthas Wild": detect_banthas_wild,
    "Idiots Rule": detect_idiots_rule,
    "Yee-Haa": detect_yee_haa,
    "Pair": detect_pair,
}


# Orders
NAMED_ORDERS = {
    "old": [
        "Full Sabacc",
        "Fleet",
        "Rhylet",
        "Wild Rhylet",
        "Squadron",
        "Five Card Squad",
        "Sylop Straight Khyron",
        "Gee Whiz!",
        "Full Straight",
        "Straight Khyron",
        "Sylop Rule of Two",
        "Banthas Wild",
        "Pure Sabacc",
        "Idiots Rule",
        "Rule of Two",
        "Yee-Haa",
        "Pair",
    ],
    "galedge": [
        "Pure Sabacc",
        "Full Sabacc",
        "Fleet",
        "Yee-Haa",
        "Rhylet",
        "Squadron",
        "Gee Whiz!",
        "Straight Khyron",
        "Banthas Wild",
        "Rule of Two",
        "Pair",
    ],
    "default": [
        "Full Sabacc",
        "Fleet",
        "Rhylet",
        "Wild Rhylet",
        "Gee Whiz!",
        "Full Straight",
        "Sylop Straight Khyron",
        "Five Card Squad",
        "Squadron",
        "Sylop Rule of Two",
        "Banthas Wild",
        "Pure Sabacc",
        "Straight Khyron",
        "Idiots Rule",
        "Rule of Two",
        "Yee-Haa",
        "Pair",
    ],
}

def named_detect_active(hand: List[Card], active_order: List[str]) -> Optional[Tuple[int,str,List[int]]]:
    """Return (rank_index, name, key) if hand is named (sum==0) given active order; else None."""
    if hand_sum(hand) != 0:
        return None
    for idx, name in enumerate(active_order):
        checker = NAMED_CHECKERS.get(name)
        if not checker:
            continue
        res = checker(hand)
        if res.ok:
            return (idx, res.name, res.key or [])
    return None
