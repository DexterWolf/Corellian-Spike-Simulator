from __future__ import annotations

from collections import Counter
from typing import List, Optional

from sabacc_cards import Card, hand_sum
from sabacc_hands import NAMED_ORDERS, named_detect_active
from sabacc_combos import COMBOS_BY_NAME

# =====================
# Public Information & Reachability
# =====================

# Maximum |sum| correction achievable in a single turn: swap/replace any card
# value v with a card drawn from the range [-10, +10], so the largest possible
# change in |sum| per card action is |(-10) - 10| = 20.
MAX_SINGLE_CARD_CORRECTION = 20

def full_value_counts() -> Counter:
    c = Counter()
    for a in range(1,11):
        c[+a] += 3
        c[-a] += 3
    c[0] += 2
    return c

def public_counts_for_player(all_players_hands: List[List[Card]], my_index: int,
                             discard_stack: List[Card],
                             known_in_hand: List[Card]) -> Counter:
    """Return the counts of cards still unknown to seat my_index (i.e. possibly in the draw pile).

    Subtracts the bot's own hand, the entire visible discard stack, and all
    known-in-hand cards (cards publicly taken from the discard by any player
    and not yet discarded back).  Does NOT account for turn-order: it treats all
    known cards as permanently unavailable.  Use public_counts_for_threshold
    instead when timing matters (e.g. reachability checks).
    """
    counts = full_value_counts()

    # Subtract my hand
    for c in all_players_hands[my_index]:
        counts[c.v] -= 1

    # Subtract entire discard stack
    for c in discard_stack:
        counts[c.v] -= 1

    # Subtract known-in-hand
    for c in known_in_hand:
        counts[c.v] -= 1

    # Safety: no negatives
    for v in list(counts.keys()):
        if counts[v] < 0:
            counts[v] = 0
    return counts




def _future_sequence_after_my_action(state: GameState, seat):
    """
    Remaining turn order (players) AFTER my current action, through end of round 3.
    Requires state["order"] and state["round_index"] to be current.
    """
    order = state.get("order")
    if not order:
        # fall back to linear seat order starting from start seat
        n = state["opts"]["num_players"]
        start = state.get("start_seat_index", 0)
        order = list(range(n))
        order = order[start:] + order[:start]
    round_num = int(state.get("round_index", 1))  # 1..3
    if seat not in order:
        return []
    pos = order.index(seat)
    seq = []
    # remainder of this round after seat:
    seq.extend(order[pos+1:])
    # full future rounds:
    for _ in range(max(0, 3 - round_num)):
        seq.extend(order)
    return seq

def _holder_exposes_before_me_again(state: GameState, seat, holder):
    """
    True if `holder` acts at least once later in the remaining sequence,
    and I (seat) still act AFTER that at least once.
    """
    if holder == seat:
        return False
    seq = _future_sequence_after_my_action(state, seat)
    seen_holder = False
    for p in seq:
        if not seen_holder and p == holder:
            seen_holder = True
        elif seen_holder and p == seat:
            return True
    return False

def public_counts_for_threshold(state: GameState, seat):
    """Like public_counts_for_player, but turn-order aware: re-adds known cards that
    could become available to the current seat before it acts again.

    A known-in-hand card (taken from the discard by another player) is added back
    to the counts if that player will take at least one more turn AND the current
    seat still acts after them — meaning they might discard that card back to the
    pile before the current seat has to decide.  Used for reachability checks
    where timing matters (e.g. _drawing_now_makes_zero_impossible).
    """
    # Start from counts that subtract flat known-in-hand (back-compat path)
    counts = public_counts_for_player(state["hands"], seat, state["discard_full"], known_flat(state))
    known_by_seat = state.get("known_in_hand_by_player", {}) or {}
    for holder, cards in known_by_seat.items():
        if _holder_exposes_before_me_again(state, seat, holder):
            for c in cards:
                counts[c.v] += 1
    # Also, there may be flat known cards without an owner recorded; treat them as NOT available (already subtracted).
    # Clip negatives defensively.
    for v in list(counts.keys()):
        if counts[v] < 0:
            counts[v] = 0
    return counts
def _public_extrema_counts(state: GameState, seat):
    """Return (counts, V_min, V_max) based on public info at this moment (horizon-aware)."""
    counts = public_counts_for_threshold(state, seat)
    present = [v for v, ct in counts.items() if ct > 0]
    if not present:
        return counts, 0, 0
    return counts, min(present), max(present)

def _one_turn_upper_bound_after_state(hand_vals, discard_top_val, allow_gain, V_min, V_max):
    """
    Upper bound on the maximum |delta| you can correct in ONE future action
    using swap/gain/dd/draw with the given hand/public extrema.
    """
    if not hand_vals:
        return 0
    A_min = min(hand_vals); A_max = max(hand_vals)

    # Discard+draw upper bound (replace one card with any remaining value in [V_min, V_max])
    if V_min == 0 and V_max == 0:
        T_dd = 0
    else:
        T_dd = max(A_max - V_min, V_max - A_min)

    # Swap with discard discard_top
    T_swap = 0
    if discard_top_val is not None:
        T_swap = max(A_max - discard_top_val, discard_top_val - A_min)

    # Gain-from-discard
    T_gain = abs(discard_top_val) if (allow_gain and discard_top_val is not None) else 0

    # Blind draw (magnitude at most max(|V_min|, V_max))
    T_draw = max(abs(V_min), abs(V_max))

    return max(T_dd, T_swap, T_gain, T_draw)

def _turns_left_after_my_action(state: GameState):
    round_num = int(state.get("round_index", 1))  # 1..3
    # After we take an action now, how many future actions do we still have this hand?
    return max(0, 3 - round_num)

def _drawing_now_makes_zero_impossible(state: GameState, seat):
    """
    True iff for every possible draw value v (consistent with public counts),
    after drawing v it is still IMPOSSIBLE to reach 0 with the remaining actions
    even under best-case future moves.
    """
    hand = state["hands"][seat]
    s = hand_sum(hand)
    discard_top_val = state["discard"][-1].v if state["discard"] else None
    allow_gain = bool(state["opts"].get("allow_discard_gain"))

    counts, V_min, V_max = _public_extrema_counts(state, seat)
    # Build the set of candidate draw values
    candidates = sorted(v for v, ct in counts.items() if ct > 0)
    if not candidates:
        # If nothing to draw, drawing isn't possible; treat as impossible -> force dd if available
        return True

    t_left = _turns_left_after_my_action(state)  # after we draw now

    for v in candidates:
        s2 = s + v
        # Updated hand after hypothetical draw
        hand2 = [c.v for c in hand] + [v]
        # Updated public extrema after consuming v
        ct_v = counts[v]
        if ct_v <= 1:
            present = [x for x, ct in counts.items() if (x != v and ct > 0)]
            if present:
                Vmin2, Vmax2 = min(present), max(present)
            else:
                Vmin2, Vmax2 = 0, 0
        else:
            Vmin2, Vmax2 = V_min, V_max

        T_one = _one_turn_upper_bound_after_state(hand2, discard_top_val, allow_gain, Vmin2, Vmax2)
        T_total = (T_one + MAX_SINGLE_CARD_CORRECTION * (t_left - 1)) if t_left >= 1 else 0
        if abs(s2) <= T_total:
            return False
    return True

def can_still_reach_min_size(state: GameState, seat, min_sz: int) -> bool:
    """Feasibility: at most one growth per remaining round, limited by public cards."""
    try:
        round_num = int(state.get("round_index", 1))  # 1..3
    except Exception:
        round_num = 1
    rounds_left = max(0, 3 - round_num)
    n = len(state["hands"][seat])
    need = max(0, int(min_sz) - n)
    if need == 0:
        return True
    draw = state.get("draw") or []
    discard = state.get("discard") or []
    public_cards = len(draw) + len(discard)
    possible_grows = min(rounds_left, public_cards)
    return possible_grows >= need

# =====================
# Known Card Tracking
# =====================

def known_flat(state: GameState) -> List[Card]:
    """Return a flat list of all publicly-known cards across all players' hands."""
    return [c for cards in state["known_in_hand_by_player"].values() for c in cards]


def known_add(state: GameState, player_index, card):
    """Record a card as publicly known to be in a player's hand.

    A card becomes "known in hand" when it is taken face-up from the discard
    pile (swap or gain action), making it visible to all other players.  Cards
    drawn from the face-down draw pile are never known.  The card stays known
    until it is discarded back (known_remove) or a dice wipe resets all hands
    (known_clear).
    """
    known_by_seat = state.setdefault("known_in_hand_by_player", {})
    lst = known_by_seat.get(player_index)
    if lst is None:
        known_by_seat[player_index] = [card]
    else:
        lst.append(card)


def known_clear(state: GameState):
    """Reset all known-in-hand tracking after a dice wipe (Spike) redeals all hands.

    After a Spike the entire table draws fresh cards from the draw pile, so
    nothing previously known about anyone's hand is still valid.
    """
    for lst in state["known_in_hand_by_player"].values():
        lst.clear()


def known_remove(state: GameState, card):
    """Remove a card from known-in-hand tracking when it is discarded back to the pile.

    Once a previously-known card leaves a player's hand it is no longer
    publicly traceable (it joins the anonymous discard pile), so other players
    can no longer reason about it specifically.
    """
    for lst in state["known_in_hand_by_player"].values():
        try:
            lst.pop(lst.index(card))
            break
        except ValueError:
            continue

def ensure_discard_after_gain(state: GameState):
    """Enforce the dealer rule: if gaining the top discard empties the pile, flip a new card.

    When a player takes the last card from the discard pile the dealer
    immediately turns over the top card of the draw pile to start a fresh
    discard.  This keeps the discard pile non-empty so subsequent players
    always have a swap/gain target.  The newly flipped card is stored in
    state['dealer_started_new_discard'] for deferred printing by the caller.
    """
    draw = state["draw"]; discard = state["discard"]
    # clear any stale flag
    state.pop("dealer_started_new_discard", None)
    if not discard and draw:
        new_discard_top = draw.pop()
        discard.append(new_discard_top)
        state["dealer_started_new_discard"] = new_discard_top
        return new_discard_top
    return None

# =====================
# Bot Policies
# =====================

def mode_label_str(mode):
    return str(mode)

def best_swap_index_for_abs_reduction(hand: List[Card], discard_top: Optional[Card]) -> Optional[int]:
    if discard_top is None:
        return None
    best_i = None
    best_val = abs(hand_sum(hand))
    for i,c in enumerate(hand):
        new_sum = hand_sum(hand) - c.v + discard_top.v
        if abs(new_sum) < best_val:
            best_val = abs(new_sum)
            best_i = i
    return best_i

def best_discard_index_for_abs_reduction(hand: List[Card]) -> Optional[int]:
    best_i = None
    best_val = abs(hand_sum(hand))
    for i,c in enumerate(hand):
        new_sum = hand_sum(hand) - c.v
        if abs(new_sum) < best_val:
            best_val = abs(new_sum)
            best_i = i
    return best_i

def _bot_safe_draw(state: GameState, seat) -> str:
    """Draws one card, recycling if needed; returns 'draw'."""
    hand = state["hands"][seat]
    draw = state["draw"]
    if not draw:
        state["recycle"](state, during_dice=False)
    if not draw:
        state["abort"]("Draw impossible during bot action.")
    c = draw.pop()
    hand.append(c)
    return "draw"



def _drawtozero_core(state: GameState, seat) -> str:
    """
    Behave like `draw`, but:
      1) If swapping ANY card with discard discard_top makes sum==0, do that.
      2) If allow_discard_gain and gaining discard discard_top makes sum==0, do that.
      3) If allow_discard_gain and we're already sum==0 and discard discard_top is 0, gain it.
      4) If sum==0 (and #3 didn't trigger), stand.
      5) Otherwise, draw.
    """
    hand     = state["hands"][seat]
    discard  = state["discard"]
    opts     = state["opts"]
    curr_sum = hand_sum(hand)

    # 1) Already at zero and a 0 is on discard_top? Gain it (if allowed).
    if opts.get("allow_discard_gain") and discard and discard[-1].v == 0:
        # Only do this special case if we're already at zero:
        if curr_sum == 0:
            c = discard.pop()
            hand.append(c)
            known_add(state, seat, c)
            ensure_discard_after_gain(state)
            return "gain_discard"

    # 2) If we're zero (and didn't gain a 0 in step 3), stand
    if curr_sum == 0:
        return "stand"

    # 3) Try gain-from-discard to hit zero immediately (if allowed)
    if opts.get("allow_discard_gain") and discard:
        if curr_sum + discard[-1].v == 0:
            c = discard.pop()
            hand.append(c)
            known_add(state, seat, c)
            ensure_discard_after_gain(state)
            return "gain_discard"

    # 4) Try swap to hit zero immediately
    if discard:
        discard_top = discard[-1]
        for i, c in enumerate(hand):
            if curr_sum - c.v + discard_top.v == 0:
                swapped_out = hand[i]
                hand[i] = discard_top
                discard[-1] = swapped_out
                known_add(state, seat, discard_top)
                known_remove(state, swapped_out)
                return "swap"



    # 5) Otherwise, consider whether drawing would make zero unreachable for sure.
    if _drawing_now_makes_zero_impossible(state, seat):
        discard_idx = best_discard_index_for_abs_reduction(hand)
        if discard_idx is not None:
            discarded = hand.pop(discard_idx)
            state["discard"].append(discarded)
            known_remove(state, discarded)
            if not state["draw"]:
                state["recycle"](state, during_dice=False)
            if not state["draw"]:
                state["abort"]("Draw impossible during bot action.")
            hand.append(state["draw"].pop())
            return "discard_draw"
    return _bot_safe_draw(state, seat)

def _swaptozero_core(state: GameState, seat) -> str:
    """
    Like drawtozero, but will also take a swap/gain that strictly reduces |sum|.
    Stands only at exactly zero.  Priority:
      A) If already sum==0 and allow_discard_gain and discard_top is 0 -> gain it.
      B) If sum==0 -> stand.
      C) Gain-to-zero or swap-to-zero.
      D) If allow_discard_gain reduces |sum| -> gain it.
      F) If swap reduces |sum| -> swap it.
      G) Otherwise draw.
    """
    hand    = state["hands"][seat]
    discard = state["discard"]
    draw    = state["draw"]
    opts    = state["opts"]
    hand_total = hand_sum(hand)

    # A) already zero and we can gain a 0
    if opts.get("allow_discard_gain") and hand_total == 0 and discard and discard[-1].v == 0:
        c = discard.pop()
        hand.append(c)
        known_add(state, seat, c)
        ensure_discard_after_gain(state)  # defined in engine; keeps discard alive
        return "gain_discard"

    # B) stand at zero (if not handled by A) or at "close" to zero
    if hand_total == 0:
        return "stand"

    # C1) gain-to-zero
    if opts.get("allow_discard_gain") and discard and (hand_total + discard[-1].v == 0):
        c = discard.pop()
        hand.append(c)
        known_add(state, seat, c)
        ensure_discard_after_gain(state)
        return "gain_discard"

    # C2) swap to immediate zero
    if discard:
        discard_top = discard[-1]
        for i, c in enumerate(hand):
            if hand_total - c.v + discard_top.v == 0:
                swapped_out = hand[i]
                hand[i] = discard_top
                discard[-1] = swapped_out
                known_add(state, seat, discard_top)
                known_remove(state, swapped_out)
                return "swap"

    # D) gain if it strictly reduces |sum|
    if opts.get("allow_discard_gain") and discard:
        new_s = hand_total + discard[-1].v
        if abs(new_s) < abs(hand_total):
            c = discard.pop()
            hand.append(c)
            known_add(state, seat, c)
            ensure_discard_after_gain(state)
            return "gain_discard"

    # F) swap if it strictly reduces |sum|
    if discard:
        i = best_swap_index_for_abs_reduction(hand, discard[-1])  # helper already present
        if i is not None:
            new_s = hand_total - hand[i].v + discard[-1].v
            if abs(new_s) < abs(hand_total):
                swapped_out = hand[i]
                discard_top = discard[-1]
                hand[i] = discard_top
                discard[-1] = swapped_out
                known_add(state, seat, discard_top)
                known_remove(state, swapped_out)
                return "swap"



    # G) otherwise draw (recycle if needed)
    # Before default draw, guard with reachability check: if drawing would make zero unreachable for sure, do discard+draw.
    if _drawing_now_makes_zero_impossible(state, seat):
        discard_idx = best_discard_index_for_abs_reduction(hand)
        if discard_idx is not None:
            discarded = hand.pop(discard_idx)
            state["discard"].append(discarded)
            known_remove(state, discarded)
            if not draw:
                state["recycle"](state, during_dice=False)
            if not draw:
                state["abort"]("Draw impossible during bot action.")
            hand.append(draw.pop())
            return "discard_draw"
    # G) Otherwise, just draw
    return _bot_safe_draw(state, seat)

# =====================
# Hunter Policy
# =====================

def _hunter_target_name(state: GameState, seat) -> Optional[str]:
    try:
        mode = state.get("modes_run", [None])[seat]
    except Exception:
        mode = None
    if isinstance(mode, tuple) and len(mode) >= 2 and mode[0] == "hunter":
        return mode[1]
    # If someone forced mode string "hunter" without tuple, no target -> treat as None
    return None

def _multiset(lst):
    return Counter(lst)

_HUNTER_BEST_COMBO_CACHE = {}
_HUNTER_NO_COMBO = object()

# Precompute zero-sum combos + their multisets and lengths per target for hunter
_HUNTER_PRECOMPUTED_COMBOS = {}

def _hunter_precomputed_combos(target_name):
    """
    Return a list of (combo_list, multiset_counter, combo_len) for this target,
    filtered to zero-sum combos only. Built once per target_name.
    """
    pre = _HUNTER_PRECOMPUTED_COMBOS.get(target_name)
    if pre is not None:
        return pre

    raw = COMBOS_BY_NAME.get(target_name) or []
    pre = []
    for tup in raw:
        combo = list(tup)
        if sum(combo) != 0:
            continue  # hunter only chases zero-sum patterns
        C = _multiset(combo)        # Counter of values in the combo
        clen = len(combo)           # total size of this combo
        pre.append((combo, C, clen))

    _HUNTER_PRECOMPUTED_COMBOS[target_name] = pre
    return pre

def _choose_best_target_combo(hand_vals, target_name):
    """
    Pick a zero-sum combo for target_name that best matches our current hand:
      minimize (missing_to_add, extras_to_drop, -kept_now).

    Performance notes:
      - We memoize per (target_name, sorted hand_vals).
      - We precompute combo multisets and lengths so we avoid rebuilding Counters
        and extra loops on every call.
      - NEW: we skip all combos smaller than the current hand size, because
        hand size can never shrink in this game.
    """
    combos = _hunter_precomputed_combos(target_name)
    if not combos:
        return None

    # Canonical key for "shape" of this hand (values only; suits are irrelevant here)
    key = tuple(sorted(hand_vals))
    cache_for_target = _HUNTER_BEST_COMBO_CACHE.setdefault(target_name, {})
    entry = cache_for_target.get(key, _HUNTER_NO_COMBO)
    if entry is not _HUNTER_NO_COMBO:
        # Could legitimately be None if no usable combo was found
        return entry

    H = _multiset(hand_vals)
    lenH = len(hand_vals)

    best = None
    best_key = (10**9, 10**9, -10**9)

    for combo, C, clen in combos:
        # 🔥 NEW: unreachable combos (too small) are skipped completely
        if clen < lenH:
            continue

        # kept = number of cards we can reuse from current hand
        kept = 0
        for v, need in C.items():
            have = H.get(v, 0)
            kept += have if have < need else need

        # Using size arithmetic instead of extra loops:
        missing = clen - kept      # cards we still need to add
        extras  = lenH - kept      # cards we have to drop
        key_tuple = (missing, extras, -kept)

        if key_tuple < best_key:
            best_key = key_tuple
            best = combo

    cache_for_target[key] = best  # may be None, we cache that too
    return best


def _index_to_discard_nonpattern(hand_vals, target_combo):
    """
    Return an index in hand_vals that is NOT needed by target_combo (by multiplicity).
    If all are needed, return index that exceeds multiplicity (overfill). Else None.
    """
    if not target_combo:
        return None
    hand_vals = list(hand_vals)
    C = _multiset(target_combo)
    used = {v: 0 for v in C}
    # First pass: prefer a value not in the combo at all
    for i, v in enumerate(hand_vals):
        if v not in C:
            return i
    # Second pass: prefer any value whose count in hand exceeds combo need
    H = _multiset(hand_vals)
    for i, v in enumerate(hand_vals):
        if H[v] > C.get(v, 0):
            return i
    return None

def _needs_value(hand_vals, target_combo, v) -> bool:
    """True if the combo still needs an extra 'v' beyond what we already hold."""
    C = _multiset(target_combo)
    H = _multiset(hand_vals)
    return H[v] < C.get(v, 0)

def _can_grow_to_combo_len(state: GameState, seat, target_combo) -> bool:
    """Respect your one-card-per-round growth and public-cards bound."""
    if not target_combo:
        return False
    min_sz, max_sz = len(target_combo), len(target_combo)  # combos are exact-size, already zero-sum
    return can_still_reach_min_size(state, seat, min_sz)

def _missing_and_extras(vals, target_combo):
    H = _multiset(vals)
    C = _multiset(target_combo)
    kept = sum(min(H[v], C[v]) for v in C)
    missing = sum((C[v] - min(H[v], C[v])) for v in C)
    extras  = sum((H[v] - min(H[v], C[v])) for v in H)
    return missing, extras

def _would_complete_combo_after_take(vals, combo, take_v, drop_index=None, grow=False):
    test = list(vals)
    if grow:
        test.append(take_v)
    else:
        if drop_index is None or not (0 <= drop_index < len(test)):
            return False
        test[drop_index] = take_v
    return _multiset(test) == _multiset(combo)

def _min_actions_needed_to_reach_combo(vals, combo):
    H = _multiset(vals); C = _multiset(combo)
    missing = sum(max(0, C[v] - H.get(v, 0)) for v in C)
    extras  = sum(max(0, H[v] - C.get(v, 0)) for v in H)
    size_now  = len(vals)
    size_goal = len(combo)

    # Growth to reach the combo's length can come from either DRAW or GAIN.
    growth_needed = max(0, size_goal - size_now)

    # Optimistically assume growth can cover up to 'growth_needed' of the missing
    # (a drawn/gained needed value doesn't create an extra).
    covered_by_growth = min(missing, growth_needed)
    remaining_missing = missing - covered_by_growth

    # SWAP compresses one missing + one extra into a single action
    pairs = min(remaining_missing, extras)
    swaps = pairs
    remaining_missing -= pairs
    extras -= pairs

    # Total minimal actions: growth steps + swaps + leftovers (each ~1 action)
    return growth_needed + swaps + remaining_missing + extras


def _turns_left_for_player(state: GameState, seat):
    """
    Remaining voluntary actions for this player including the current turn.
    With 3 rounds total and one action per round, this is simply 4 - round_index.
    If round_index is missing, assume we're on the last action (1).
    """
    try:
        round_num = int(state.get("round_index", 3))
    except Exception:
        round_num = 3
    # round_num ∈ {1,2,3}  -> returns {3,2,1} including this action
    return max(1, 4 - round_num)

def _nonpattern_extra_indices(vals, combo):
    """Return indices of cards in vals that are NOT needed by combo multiplicity."""
    C = Counter(combo)
    remaining = dict(C)
    extras = []
    for i, v in enumerate(vals):
        need = remaining.get(v, 0)
        if need > 0:
            remaining[v] = need - 1
        else:
            extras.append(i)  # not needed or over multiplicity
    return extras


def _hunter_core(state: GameState, seat) -> str:
    """
    Try to progress toward the chosen zero-sum target pattern with a cheap one-turn action:
      - SWAP the needed discard_top discard if we can drop a non-pattern card (and on last action only if it completes).
      - GAIN the needed discard_top discard (if allowed) when we can still grow to the target_combo size
        (on last action only if it completes).
      - DISCARD+DRAW a non-pattern (or overfill) card to search needed values
        (but on last action: abandon the chase and minimize |sum|).
      - Otherwise, DRAW unless doing so would make zero unreachable, in which case DISCARD+DRAW.
    If no valid target or it's impossible to finish in time, behave like drawtozero for this action.
    """
    hand    = state["hands"][seat]
    discard = state["discard"]
    draw    = state["draw"]
    opts    = state["opts"]

    target_name = _hunter_target_name(state, seat)
    if not target_name or target_name not in COMBOS_BY_NAME:
        return _drawtozero_core(state, seat)  # no valid target → zero-hunt

    card_values = [c.v for c in hand]
    target_combo = _choose_best_target_combo(card_values, target_name)
    if not target_combo:
        return _drawtozero_core(state, seat)  # nothing usable → zero-hunt

    # If we already ARE the exact target pattern (by multiset), stand immediately.
    # This works even for "unranked" targets like plain Sabacc zero-sum.
    if _multiset(card_values) == _multiset(target_combo):
        return "stand"

    # If we already ARE the exact target, stand.
    active = named_detect_active(hand, NAMED_ORDERS[opts["named_order"]])
    if active and active[1] == target_name:
        return "stand"

    # Turns left (including this turn)
    turns_left = _turns_left_for_player(state, seat)

    # Tight feasibility check: can we still finish in time?
    can_gain = bool(opts.get("allow_discard_gain"))
    can_add_card = _can_grow_to_combo_len(state, seat, target_combo)
    actions_needed = _min_actions_needed_to_reach_combo(card_values, target_combo)
    if actions_needed > turns_left:
        # Not finishable in remaining actions → play for zero this action.
        return _drawtozero_core(state, seat)

    # NOTE: If sum(card_values) == 0 but actions_needed <= turns_left, we intentionally KEEP CHASING.
    # (Do not delegate to drawtozero, which might insta-stand.)

    # --- SWAP: take needed discard_top discard if we can discard a non-pattern card.
    if discard:
        topv = discard[-1].v
        if _needs_value(card_values, target_combo, topv):
            discard_idx = _index_to_discard_nonpattern(card_values, target_combo)
            if discard_idx is not None:
                # On last action, only swap if this completes the target.
                if turns_left > 1 or _would_complete_combo_after_take(card_values, target_combo, topv, drop_index=discard_idx, grow=False):
                    swapped_out = hand[discard_idx]
                    discard_top = discard[-1]
                    hand[discard_idx] = discard_top
                    discard[-1] = swapped_out
                    known_add(state, seat, discard_top)
                    known_remove(state, swapped_out)
                    return "swap"

    # --- GAIN: take needed discard_top discard and grow, if allowed and feasible.
    if can_gain and discard:
        topv = discard[-1].v
        if _needs_value(card_values, target_combo, topv) and len(hand) < len(target_combo) and can_add_card:
            # On last action, only gain if this completes the target.
            if turns_left > 1 or _would_complete_combo_after_take(card_values, target_combo, topv, grow=True):
                c = discard.pop()
                hand.append(c)
                known_add(state, seat, c)
                ensure_discard_after_gain(state)
                return "gain_discard"

    # --- DISCARD+DRAW: if we still have extras vs target_combo, cycle one.
    discard_idx = _index_to_discard_nonpattern(card_values, target_combo)
    if discard_idx is not None:
        if turns_left > 1:
            # Continue chasing the pattern.
            discarded = hand.pop(discard_idx)
            state["discard"].append(discarded)
            known_remove(state, discarded)
            if not draw:
                state["recycle"](state, during_dice=False)
            if not draw:
                state["abort"]("Draw impossible during bot action.")
            hand.append(draw.pop())
            return "discard_draw"

        else:
            # Last action:
            # If there is EXACTLY ONE non-pattern extra, only abandon if discarding it would lock out zero.
            extras_idx = _nonpattern_extra_indices(card_values, target_combo)
            if len(extras_idx) == 1:
                j_only = extras_idx[0]
                S = sum(card.v for card in hand)
                after_drop_sum = S - hand[j_only].v
                need_v = -after_drop_sum

                # INFO-LIMITED view: public counts (no peeking into draw)
                available_cards = public_counts_for_player(state["hands"], seat, state["discard_full"], known_flat(state))
                zero_still_possible = available_cards.get(need_v, 0) > 0

                if zero_still_possible:
                    # Do NOT abandon: discard that single non-pattern extra and draw.
                    discarded = hand.pop(j_only)
                    state["discard"].append(discarded)
                    known_remove(state, discarded)
                    if not draw:
                        state["recycle"](state, during_dice=False)
                    if not draw:
                        state["abort"]("Draw impossible during bot action.")
                    hand.append(draw.pop())
                    return "discard_draw"

            # Abandon the chase; minimize |sum| (original behavior)
            j_abs = best_discard_index_for_abs_reduction(hand)
            if j_abs is None:
                j_abs = discard_idx  # fallback
            discarded = hand.pop(j_abs)
            state["discard"].append(discarded)
            known_remove(state, discarded)
            if not draw:
                state["recycle"](state, during_dice=False)
            if not draw:
                state["abort"]("Draw impossible during bot action.")
            hand.append(draw.pop())
            return "discard_draw"

    # --- Safety: if drawing now would make zero unreachable, do a discard+draw instead.
    if _drawing_now_makes_zero_impossible(state, seat):
        j2 = _index_to_discard_nonpattern(card_values, target_combo)
        if j2 is None:
            j2 = best_discard_index_for_abs_reduction(hand)
        if j2 is not None:
            discarded = hand.pop(j2)
            state["discard"].append(discarded)
            known_remove(state, discarded)
            if not draw:
                state["recycle"](state, during_dice=False)
            if not draw:
                state["abort"]("Draw impossible during bot action.")
            hand.append(draw.pop())
            return "discard_draw"

    # Default: just draw (safe). This growth step is accounted for by actions_needed logic.
    return _bot_safe_draw(state, seat)



def _minimizer_core(state: GameState, seat) -> str:
    """
    Like swaptozero, but stands at |sum| ≤ 1 (not just exactly zero).  Priority:
      A) If already sum==0 and allow_discard_gain and discard_top is 0 -> gain it.
      B) If |sum| ≤ 1 -> stand.
      C) Gain-to-zero or swap-to-zero.
      D) If allow_discard_gain reduces |sum| -> gain it.
      F) If swap reduces |sum| -> swap it.
      G) Otherwise draw.
    """
    hand    = state["hands"][seat]
    discard = state["discard"]
    draw    = state["draw"]
    opts    = state["opts"]
    hand_total = hand_sum(hand)

    # A) already zero and we can gain a 0
    if opts.get("allow_discard_gain") and hand_total == 0 and discard and discard[-1].v == 0:
        c = discard.pop()
        hand.append(c)
        known_add(state, seat, c)
        ensure_discard_after_gain(state)  # defined in engine; keeps discard alive
        return "gain_discard"

    # B) stand at zero (if not handled by A) or at "close" to zero
    if abs(hand_total) <= 1:
        return "stand"

    # C1) gain-to-zero
    if opts.get("allow_discard_gain") and discard and (hand_total + discard[-1].v == 0):
        c = discard.pop()
        hand.append(c)
        known_add(state, seat, c)
        ensure_discard_after_gain(state)
        return "gain_discard"

    # C2) swap to immediate zero
    if discard:
        discard_top = discard[-1]
        for i, c in enumerate(hand):
            if hand_total - c.v + discard_top.v == 0:
                swapped_out = hand[i]
                hand[i] = discard_top
                discard[-1] = swapped_out
                known_add(state, seat, discard_top)
                known_remove(state, swapped_out)
                return "swap"

    # D) gain if it strictly reduces |sum|
    if opts.get("allow_discard_gain") and discard:
        new_s = hand_total + discard[-1].v
        if abs(new_s) < abs(hand_total):
            c = discard.pop()
            hand.append(c)
            known_add(state, seat, c)
            ensure_discard_after_gain(state)
            return "gain_discard"

    # F) swap if it strictly reduces |sum|
    if discard:
        i = best_swap_index_for_abs_reduction(hand, discard[-1])  # helper already present
        if i is not None:
            new_s = hand_total - hand[i].v + discard[-1].v
            if abs(new_s) < abs(hand_total):
                swapped_out = hand[i]
                discard_top = discard[-1]
                hand[i] = discard_top
                discard[-1] = swapped_out
                known_add(state, seat, discard_top)
                known_remove(state, swapped_out)
                return "swap"



    # G) otherwise draw (recycle if needed)
    # Before default draw, guard with reachability check: if drawing would make zero unreachable for sure, do discard+draw.
    if _drawing_now_makes_zero_impossible(state, seat):
        discard_idx = best_discard_index_for_abs_reduction(hand)
        if discard_idx is not None:
            discarded = hand.pop(discard_idx)
            state["discard"].append(discarded)
            known_remove(state, discarded)
            if not draw:
                state["recycle"](state, during_dice=False)
            if not draw:
                state["abort"]("Draw impossible during bot action.")
            hand.append(draw.pop())
            return "discard_draw"
    # G) Otherwise, just draw
    return _bot_safe_draw(state, seat)




def forced_discard_draw_index_min_abs(state: GameState, seat, rng, mode_label=None):
    """Used when spike dice doubles forces a discard+draw.
    Choose the discard index in a policy-aware way:
      - hunter: discard a non-pattern extra for the target target_combo
      - otherwise: minimize |sum| after removal
    """
    hand = state["hands"][seat]
    if not hand:
        return None

    # Honor hunter targets: discard a non-pattern extra for the best
    # zero-sum target_combo of the target (keep pursuing the named hand).
    target_hand = _hunter_target_name(state, seat)
    if target_hand:
        card_values = [c.v for c in hand]
        target_combo = _choose_best_target_combo(card_values, target_hand)
        discard_idx = _index_to_discard_nonpattern(card_values, target_combo)
        if discard_idx is None:
            discard_idx = best_discard_index_for_abs_reduction(hand)
            if discard_idx is None:
                best_j = 0; best_val = float('inf')
                s0 = abs(hand_sum(hand))
                for i, c in enumerate(hand):
                    inc = abs(hand_sum(hand) - c.v) - s0
                    if inc < best_val:
                        best_val = inc; best_j = i
                return best_j
        return discard_idx

    # Minimize |sum| after removal
    discard_idx = best_discard_index_for_abs_reduction(hand)
    if discard_idx is None:
        best_j = 0; best_val = float('inf')
        s0 = abs(hand_sum(hand))
        for i, c in enumerate(hand):
            inc = abs(hand_sum(hand) - c.v) - s0
            if inc < best_val:
                best_val = inc; best_j = i
        return best_j
    return discard_idx
