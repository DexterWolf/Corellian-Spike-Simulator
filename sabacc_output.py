from __future__ import annotations

from sabacc_cards import Card
from sabacc_hands import NAMED_ORDERS, named_detect_active
from sabacc_engine import TIEBREAKER_LABELS

# =====================
# Output & Reporting
# =====================

def _sep(title):
    """Print a section title with a matching-length underline."""
    print(title)
    print("-" * len(title))


def _start_str(start_vals):
    """Format a starting hand as '[3, -2] (sum=1)'."""
    return f"{start_vals} (sum={sum(start_vals)})"


def _seat_start_display(i, starts_first, starts_same):
    """Return formatted start string for seat i, or 'randomized'."""
    if starts_first is not None and starts_same is not None and i < len(starts_first):
        if starts_same[i]:
            return _start_str(starts_first[i])
    return "randomized"


def print_interactive_header(opts, seed, modes_run):
    print("=" * 52)
    print("  Interactive Sabacc Session")
    print("=" * 52)
    seed_label = f"{seed} (given)" if opts["seed_given"] else f"{seed} (random)"
    print(f"  Seed    : {seed_label}")
    print(f"  Players : {opts['num_players']}   Human seat: {opts['human_player']}")
    modes_str = ", ".join(
        f"P{i+1}: {'You' if i == opts['human_player']-1 else str(modes_run[i])}"
        for i in range(opts['num_players'])
    )
    print(f"  Modes   : {modes_str}")
    dice_str = "OFF" if opts["no_dice"] else opts["dice_mode"]
    print(f"  Dice    : {dice_str}")
    print("=" * 52)
    print()


def print_results_game_block(game_num, game, opts):
    print(f"Results (Game {game_num})")
    modes_str = ", ".join(
        f"P{i+1}: {'You' if (opts['human_player'] and i == opts['human_player']-1) else m}"
        for i, m in enumerate(game['modes'])
    )
    print(f"  Modes: {modes_str}")
    starts_str = ", ".join(
        f"P{i+1}: {vals} (sum={sum(vals)})" for i, vals in enumerate(game['starts'])
    )
    print(f"  Starts: {starts_str}")
    print(f"  First face-up discard: {game['first_discard']}")
    # Rounds
    for ridx, moves in enumerate(game["rounds_moves"], start=1):
        print(f"  Round {ridx}:")
        for mv in moves:
            before = mv.before
            after = mv.after
            print(f"    P{mv.player_index+1} ({mv.mode_label}): {before}  ->[{mv.action}]->  {after}")
            for evt in getattr(mv, "reshuffle_events", []) or []:
                print(f"      {evt}")
        # Dice line for that round
        if ridx <= len(game["dice_rolls"]):
            roll = game["dice_rolls"][ridx-1]
            d1,d2,redraw_sizes,is_doubles,is_spike,dice_events = roll.d1,roll.d2,roll.redraw_sizes,roll.is_doubles,roll.is_spike_double,roll.dice_events
            if opts["dice_mode"] == "spike" and is_doubles and d1==1:
                print(f"    Dice: {d1} + {d2} -- SPIKE DOUBLES! Everyone discards and redraws")
                for evt in dice_events:
                    print(f"      [during dice] {evt}")
                if redraw_sizes:
                    print(f"    (Redraw sizes: {redraw_sizes})" )
            elif is_doubles and opts["dice_mode"]=="classic":
                print(f"    Dice: {d1} + {d2} -- DOUBLES! Everyone discards and redraws")
                for evt in dice_events:
                    print(f"      [during dice] {evt}")
                if redraw_sizes:
                    print(f"    (Redraw sizes: {redraw_sizes})")
            elif is_doubles:
                print(f"    Dice: {d1} + {d2} -- DOUBLES!")
                for evt in dice_events:
                    print(f"      [during dice] {evt}")
            else:
                print(f"    Dice: {d1} + {d2}")
                for evt in dice_events:
                    print(f"      [during dice] {evt}")
    # Final
    print("  Final:")
    for i,(vals,s, name) in enumerate(game["finals"], start=1):
        name_str = f" [{name}]" if name else ""
        print(f"    P{i}: {vals}, sum={s}{name_str}")
    print(f"    Winner: Player {game['winner']+1} ({game['winner_reason']})")
    d, sd = game['dice_stats']
    if d:
        print(f"    Doubles in game: {d}")
    if sd:
        print(f"    Spike Doubles in game: {sd}")
    print()


def print_simulation_summary(all_games, opts, seed, counters, per_player_stats, named_agg, starts_first, starts_same):
    n = counters.total_games or max(len(all_games), 1)

    # ── Run Setup ────────────────────────────────────────────────
    _sep("Simulation Summary")
    seed_label = f"{seed} (given)" if opts["seed_given"] else f"{seed} (random)"
    print(f"  Games: {n}   Seed: {seed_label}   Players: {opts['num_players']}")
    if opts["human_player"]:
        print(f"  Human player: seat {opts['human_player']}")
    print()

    # ── Starting Hands ───────────────────────────────────────────
    _sep("Starting Hands")
    if opts["randomize_all"]:
        print("  Fresh random hands dealt to every player before each game.")
    else:
        parts = []
        for i in range(opts["num_players"]):
            parts.append(f"P{i+1}: {_seat_start_display(i, starts_first, starts_same)}")
        print("  " + "   ".join(parts))
        if opts["random_starts"]:
            print("  (Same random hands used for every game in this run.)")
    print()

    # ── Dice ─────────────────────────────────────────────────────
    _sep("Dice")
    if opts["no_dice"]:
        print("  Dice phase: OFF")
    else:
        print(f"  Mode: {opts['dice_mode']}   Rotate first seat: {'ON' if opts['rotate_first'] else 'OFF'}")
    print()

    # ── Rules ────────────────────────────────────────────────────
    _sep("Rules")
    named_tie_str = "lower key wins (Galaxy's Edge mode)" if opts["named_low_wins"] else "higher key wins"
    print(f"  Named hand order : {opts['named_order']}   Named tie: {named_tie_str}")
    high_abs_str = "ON" if opts["high_abs"] else "OFF (--no-high-abs)"
    suits_str    = "ON" if opts["use_suits"] else "OFF (--no-suits)"
    print(f"  Highest |card| tiebreak: {high_abs_str}   Suited tiebreak: {suits_str}")
    print(f"  Discard gain: {'ON' if opts['allow_discard_gain'] else 'OFF'}")
    print()

    # ── Outcome table ────────────────────────────────────────────
    _sep("Outcome")

    # Build table rows first so we can compute column widths
    rows = []
    for i in range(opts["num_players"]):
        st = per_player_stats[i]
        seat   = f"P{i+1}"
        mode   = st["mode_label"]
        start  = _seat_start_display(i, starts_first, starts_same)
        win_pct = f"{st['wins']*100.0/n:0.2f} %"
        wins   = f"({st['wins']})"
        rows.append((seat, mode, start, win_pct, wins))

    w_mode  = max(len(r[1]) for r in rows)
    w_start = max(len(r[2]) for r in rows)
    w_pct   = max(len(r[3]) for r in rows)
    w_wins  = max(len(r[4]) for r in rows)

    for seat, mode, start, win_pct, wins in rows:
        print(f"  {seat}  {mode:<{w_mode}}  {start:<{w_start}}  {win_pct:>{w_pct}}  {wins:>{w_wins}}")

    print(f"  True ties (single-card draw): {counters.true_ties_games}")
    print()

    # ── Dice Statistics ──────────────────────────────────────────
    _sep("Dice Statistics")
    print(f"  Reshuffles: {counters.reshuffles_total} total ({counters.reshuffles_dice} during dice redraws)")
    avg_d  = sum(k * v for k, v in counters.doubles_hist.items()) / max(1, n)
    avg_sd = sum(k * v for k, v in counters.spike_doubles_hist.items()) / max(1, n)
    print(f"  Avg doubles / game: {avg_d:0.3f}")
    if opts["dice_mode"] == "spike" and not opts["no_dice"]:
        print(f"  Avg Spike doubles / game: {avg_sd:0.3f}")
    for k in range(0, 4):
        pct = (counters.doubles_hist.get(k, 0) * 100.0) / max(1, n)
        print(f"  {k} doubles : {pct:0.1f} % ({counters.doubles_hist.get(k, 0)})")
    if opts["dice_mode"] == "spike" and not opts["no_dice"]:
        for k in range(0, 4):
            pct = (counters.spike_doubles_hist.get(k, 0) * 100.0) / max(1, n)
            print(f"  {k} Spike doubles : {pct:0.1f} % ({counters.spike_doubles_hist.get(k, 0)})")
    print()

    # ── Player Statistics ────────────────────────────────────────
    _sep("Player Statistics")
    for i in range(opts["num_players"]):
        st = per_player_stats[i]
        print(f"  Player {i+1} ({st['mode_label']}):")
        avg_abs = (st["end_sum_total"] / st["end_sum_count"]) if st["end_sum_count"] else 0.0
        print(f"    Avg |sum| at end : {avg_abs:.2f}")
        exact_pct = st['exact_zero'] * 100.0 / n
        print(f"    Exact zero       : {exact_pct:0.2f} % ({st['exact_zero']})")
        zw = st['zero_wins']
        zt = st['exact_zero']
        zwp = (zw * 100.0 / zt) if zt else 0.0
        print(f"    Zero win rate    : {zwp:0.2f} % ({zw} of {zt} zero hands)")
        nt = sum(st['named_totals'].values())
        ntp = (nt * 100.0 / n) if n else 0.0
        nw = sum(st['named_wins'].values())
        nwp = (nw * 100.0 / n) if n else 0.0
        print(f"    Named hands      : {ntp:0.2f} % ({nt})   Named wins: {nwp:0.2f} % ({nw})")

        def _fmt_counter_inline(counter, opts):
            order = list(NAMED_ORDERS[opts["named_order"]])
            idx = {nm: i for i, nm in enumerate(order)}
            rows = sorted(counter.items(), key=lambda kv: idx.get(kv[0], len(order)))
            return ", ".join(f"{name}: {cnt}" for name, cnt in rows)

        wins_inline   = _fmt_counter_inline(st['named_wins'],   opts)
        totals_inline = _fmt_counter_inline(st['named_totals'], opts)
        if wins_inline:
            print(f"    Named wins       : {wins_inline}")
        if totals_inline:
            print(f"    Named totals     : {totals_inline}")
    print()

    # ── Tiebreakers ──────────────────────────────────────────────
    if counters.tiebreaker_hist:
        _sep("Tiebreakers")
        total_tb = sum(counters.tiebreaker_hist.values())
        display_order = [5, 6, 2, 1, 3, 4, 7, 8, 9, 10, 11, 12]
        for code in display_order:
            if code not in counters.tiebreaker_hist:
                continue
            cnt   = counters.tiebreaker_hist[code]
            label = TIEBREAKER_LABELS.get(code, f"code {code}")
            pct   = cnt * 100.0 / max(1, total_tb)
            print(f"  {label:<30} : {pct:6.2f} % ({cnt})")
        print()

    # ── Named Hand Stats ─────────────────────────────────────────
    _sep("Named Hand Stats")

    def _rank_name_by_order(name, opts):
        order     = list(NAMED_ORDERS[opts["named_order"]])
        order_idx = {n: i for i, n in enumerate(order)}
        return order_idx.get(name, len(order))

    def _print_named_counter_sorted(counter, opts, indent="  "):
        if not counter:
            print(f"{indent}(none)")
            return
        rows = sorted(counter.items(), key=lambda kv: _rank_name_by_order(kv[0], opts))
        w_name = max(len(n) for n, _ in rows)
        w_cnt  = max(len(str(c)) for _, c in rows)
        for name, cnt in rows:
            print(f"{indent}{name:<{w_name}} : {cnt:>{w_cnt}}")

    # Overall totals
    print("Overall (all players & games):")
    overall = named_agg["overall"]
    wins_c  = named_agg["wins"]
    if overall:
        order_list = NAMED_ORDERS[opts["named_order"]]
        rank_map = {name: i for i, name in enumerate(order_list)}
        rank_map["Sabacc"]  = len(order_list)
        rank_map["Nulrhek"] = len(order_list) + 1
        name_w = max(len(n) for n in overall.keys())
        cnt_w  = max(len(str(c)) for c in overall.values())
        win_w  = max(len(str(wins_c.get(n, 0))) for n in overall.keys())
        for name in sorted(overall.keys(), key=lambda n: rank_map.get(n, 10**6)):
            total = overall[name]
            wins  = wins_c.get(name, 0)
            print(f"  {name:<{name_w}} : {total:>{cnt_w}}   Win: {wins:>{win_w}}")
    else:
        print("  (none)")
    print()

    # Shared hands
    print("Shared named hand (same hand held by multiple players in one game):")
    _print_named_counter_sorted(named_agg["shared"], opts, indent="  ")
    print()

    # Single named hand — only one named hand appeared in the game
    print("Single named hand in a game (no other named hand present):")
    sg = named_agg["single_games"]
    if not sg:
        print("  (none)")
    else:
        order = list(NAMED_ORDERS[opts["named_order"]])
        def _rank(n):
            try: return order.index(n)
            except ValueError: return len(order)
        rows = sorted(sg.items(), key=lambda kv: _rank(kv[0]))
        w_name = max(len(name) for name, _ in rows)
        w_cnt  = max(len(str(cnt))  for _, cnt  in rows)
        for name, cnt in rows:
            print(f"  {name:<{w_name}} : {cnt:>{w_cnt}}")
    print()

    # Multiple named hands in the same game (constellations)
    print("Multiple named hands in the same game:")
    con = named_agg["constellations"]
    if not con:
        print("  (none)")
    else:
        def _lab_str(k):
            return " + ".join(k) if isinstance(k, tuple) else str(k)
        rows = sorted(con.items(), key=lambda kv: (-kv[1], _lab_str(kv[0])))
        w_lab = max(len(_lab_str(k)) for k, _ in rows)
        w_cnt = max(len(str(c))      for _, c  in rows)
        for label, cnt in rows:
            s = _lab_str(label)
            print(f"  {s:<{w_lab}} : {cnt:>{w_cnt}}")

    print()


_ACTION_LABELS = {
    "stand":        "stand",
    "draw":         "draw (blind)",
    "discard_draw": "discard + redraw",
    "swap":         "swap with discard",
    "gain_discard": "gain from discard",
}

def print_replays(all_games, opts):
    print("Replay")
    print("------")
    k = int(opts.get("keep_games", 8) or 0)
    for gi, game in enumerate(all_games[:k], start=1):
        print(f"\nGame {gi}")
        modes_str = ", ".join(
            f"P{i+1}: {m}" for i, m in enumerate(game['modes'])
        )
        print(f"  Modes: {modes_str}")
        starts_str = ", ".join(
            f"P{i+1}: {vals} (sum={sum(vals)})" for i, vals in enumerate(game['starts'])
        )
        print(f"  Starts: {starts_str}")
        print(f"  First face-up discard: {game['first_discard']}")
        # rounds
        for ridx, moves in enumerate(game["rounds_moves"], start=1):
            print(f"  Round {ridx}:")
            for mv in moves:
                action_str = _ACTION_LABELS.get(mv.action, mv.action)
                print(f"    P{mv.player_index+1} ({mv.mode_label}): {mv.before}  ->  {mv.after}  [{action_str}]")
                for evt in getattr(mv, "reshuffle_events", []) or []:
                    print(f"      {evt}")
            if ridx <= len(game["dice_rolls"]):
                roll = game["dice_rolls"][ridx-1]
                d1,d2,redraw_sizes,is_doubles,is_spike,dice_events = roll.d1,roll.d2,roll.redraw_sizes,roll.is_doubles,roll.is_spike_double,roll.dice_events
                if opts["dice_mode"] == "spike" and is_doubles and d1==1:
                    print(f"    Dice: {d1} + {d2} -- SPIKE DOUBLES! Everyone discards and redraws")
                    for evt in dice_events:
                        print(f"      [during dice] {evt}")
                    if redraw_sizes:
                        print(f"    (Redraw sizes: {redraw_sizes})" )
                elif is_doubles and opts["dice_mode"]=="classic":
                    print(f"    Dice: {d1} + {d2} -- DOUBLES! Everyone discards and redraws")
                    for evt in dice_events:
                        print(f"      [during dice] {evt}")
                    if redraw_sizes:
                        print(f"    (Redraw sizes: {redraw_sizes})")
                elif is_doubles:
                    print(f"    Dice: {d1} + {d2} -- DOUBLES!")
                    for evt in dice_events:
                        print(f"      [during dice] {evt}")
                else:
                    print(f"    Dice: {d1} + {d2}")
                    for evt in dice_events:
                        print(f"      [during dice] {evt}")
        print("  Final:")
        for i,(vals,s, name) in enumerate(game["finals"], start=1):
            name_str = f" [{name}]" if name else ""
            print(f"    P{i}: {vals}, sum={s}{name_str}")
        print(f"    Winner: Player {game['winner']+1} ({game['winner_reason']})")
        d, sd = game['dice_stats']
        if d:
            print(f"    Doubles in game: {d}")
        if sd:
            print(f"    Spike Doubles in game: {sd}")
