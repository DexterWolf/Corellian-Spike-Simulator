from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

# =====================
# Cards & Deck
# =====================

SUITS = ("Circle", "Square", "Triangle")  # for non-zero
Sylop = None  # suit for sabacc_seats

@dataclass(frozen=True)
class Card:
    v: int
    s: Optional[str]  # None for sabacc_seats

    def __str__(self):
        return f"{self.v}"

def build_full_deck() -> List[Card]:
    """Return a full 62-card deck as Card objects with suits assigned for non-zero values.
    - ±1..±10 each appear 3x (with 3 suits total across signs).
    - sabacc_seats appear 2x (no suit).
    Implementation: for each abs value a in 1..10 produce 3 cards at +a with the 3 suits,
    and 3 cards at -a with the same 3 suits (total 6 per abs value). Zeros: two Card(0,None).
    """
    deck = []
    for a in range(1, 11):
        for suit in SUITS:
            deck.append(Card(+a, suit))
            deck.append(Card(-a, suit))
    deck.append(Card(0, Sylop))
    deck.append(Card(0, Sylop))
    assert len(deck) == 62
    return deck

def shuffle_deck(rng: random.Random, deck: List[Card]):
    rng.shuffle(deck)

def card_value_counts(cards: List[Card]) -> Counter:
    """Return counts by numeric value (ignore suits)."""
    c = Counter()
    for c0 in cards:
        c[c0.v] += 1
    return c

def remove_values_from_deck(rng: random.Random, deck: List[Card], values: List[int]):
    """Remove specific values (by value only) from the deck, choosing suits uniformly among available.
    Raises ValueError if not possible.
    """
    for v in values:
        idxs = [i for i, c in enumerate(deck) if c.v == v]
        if not idxs:
            raise ValueError(f"Cannot take value {v} from deck; not enough copies remain.")
        # pick any available card of that value; choose uniformly
        i = rng.choice(idxs)
        deck.pop(i)

def pop_top(deck: List[Card]) -> Card:
    return deck.pop()  # discard_top is end of list

def push_top(stack: List[Card], c: Card):
    stack.append(c)

def hand_values(hand: List[Card]) -> List[int]:
    return [c.v for c in hand]

def hand_sum(hand: List[Card]) -> int:
    return sum(c.v for c in hand)

def hand_abs_values_sorted_desc(hand: List[Card]) -> List[int]:
    return sorted([abs(c.v) for c in hand], reverse=True)

def hand_positives_sorted_desc(hand: List[Card]) -> List[int]:
    return sorted([c.v for c in hand if c.v > 0], reverse=True)

def hand_is_suited_nonzero(hand: List[Card]) -> bool:
    suits = [c.s for c in hand if c.v != 0]
    if not suits:
        return False
    return all(s == suits[0] for s in suits)
