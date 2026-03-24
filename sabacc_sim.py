
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corellian Spike Sabacc Simulator
=================================

A Monte Carlo simulator for **Corellian Spike Sabacc**, a card game from the
Star Wars universe.  The ruleset implemented here is the casino-ready variant
described at https://corellian.spike.today (not the Galaxy's Edge version —
see that site for the differences).

Game basics
-----------
- 62-card deck: values −10 to +10 in three suits (Circle, Square, Triangle),
  three copies per value/sign combination; plus 2 zero-value Sylops (no suit).
- 2–8 players, each starting with 2 cards dealt face-down.
- 3 rounds.  Each round: card phase (each player acts once) → spike dice phase.
- Goal: hand summing to zero (Sabacc), or as close as possible (Nulrhek).
- Spike dice: non-1 doubles → forced discard+draw for each player;
  double-1 (Spike) → full hand wipe and redraw.
- Named hands (ranked by rarity): Full Sabacc, Fleet, Rhylet, Wild Rhylet,
  Gee Whiz!, Full Straight, Sylop Straight Khyron, Five Card Squad, Squadron,
  Sylop Rule of Two, Banthas Wild, Pure Sabacc, Straight Khyron, Idiots Rule,
  Rule of Two, Yee-Haa, Pair.
- Tiebreaker chain: named hand → most cards → highest Σ|cards| →
  highest |card| → highest positive card → suited → single-card draw.

Usage
-----
Run a quick simulation (10 000 games, 2 players, default bot modes)::

    python Corellian_Spike_Sabacc_Simulator.py

Key flags::

    -n N                    Number of games (default 10 000)
    --num-players N         2–8 players
    --modes "m1,m2,..."     Bot mode per seat: drawtozero, minimizer,
                            swaptozero, draw, stand, newmodes, hunter
    --hunter-target "p1:'Squadron'"  Named hand for the hunter bot
    --human-player N        Make seat N interactive
    --seed N                RNG seed for reproducibility
    --replay-file PATH      Write per-hand replay to JSONL or Parquet
    --compare "h1:5,-5;h2:3,-3,0"   Compare two hands and print the winner

Run ``python Corellian_Spike_Sabacc_Simulator.py --help`` for the full list.

Dependencies
------------
``sabacc_combos.py`` must be present in the same directory.  It contains a
precomputed lookup table of all valid zero-sum card combinations for every
named hand; the *hunter* bot uses it to decide which cards to chase.

Notes
-----
- Suits exist for non-zero cards (Circle, Square, Triangle).
  Zeros are Sylops (no suit).
- Card display shows only values; suits are used only for the optional
  suited tiebreaker (``--use-suits``).
- Bot policies mutate game state directly, then return an action string.
- For the human seat the engine performs the mutation after reading the
  chosen action.
"""

from __future__ import annotations

import random
import time
import argparse
from collections import Counter


RUN_START_TIME = None  # set in main() for elapsed-time printing


from sabacc_cards import (
    Card,
    build_full_deck,
    remove_values_from_deck,
)
from sabacc_hands import (
    NAMED_ORDERS,
    parse_starts_string,
    parse_fixed_starts_string,
)
from sabacc_policies import (
    mode_label_str,
)
from sabacc_engine import (
    pause_before_results,
    RunCounters,
    ReplayFileWriter,
    winner_and_reason,
    engine_recycle, engine_abort,
    play_one_game,
)

# =====================
# Output & Reporting
# =====================

from sabacc_output import (
    print_interactive_header,
    print_results_game_block,
    print_simulation_summary,
    print_replays,
)

# =====================
# CLI & Entry Point
# =====================

def parse_modes(args, num_players):
    # --all-modes | --modes | --p{i}-mode
    modes = ["drawtozero"]*num_players
    if args.all_modes:
        if args.all_modes not in ("draw","stand","drawtozero","minimizer","swaptozero","newmodes","hunter"):
            raise SystemExit("--all-modes invalid.")
        modes = [args.all_modes]*num_players
    if args.modes:
        lst = [s.strip() for s in args.modes.split(",") if s.strip()]
        while len(lst) < num_players:
            lst.append(lst[-1] if lst else "drawtozero")
        modes = lst[:num_players]
    # per-seat overrides
    for i in range(1, num_players+1):
        val = getattr(args, f"p{i}_mode", None)
        if val:
            modes[i-1] = val

    # hunter targets (per-seat)
    hunter_map = {}
    if args.hunter_target:
        parts = args.hunter_target.split(",")
        for part in parts:
            if ":" not in part:
                continue
            p, target_hand = part.split(":", 1)
            p = p.strip().lower()
            if not p.startswith("p"):
                continue
            try:
                idx = int(p[1:]) - 1
            except:
                continue
            target_hand = target_hand.strip().strip("'").strip('"')
            if target_hand:
                hunter_map[idx] = target_hand

    # apply hunter wrapper
    res = []
    for i,m in enumerate(modes):
        if i in hunter_map:
            res.append(("hunter", hunter_map[i]))
        else:
            res.append(m)
    return res

def main():
    parser = argparse.ArgumentParser(description="Corellian Spike Sabacc Simulator (V8)")
    global RUN_START_TIME
    RUN_START_TIME = time.perf_counter()
    parser.add_argument("-n","--n-games", type=int, default=10000,
                        help="Number of games to simulate (default 10000).")
    parser.add_argument("--num-players", type=int, default=2, choices=list(range(1,9)),
                        help="Number of players at the table (2–8, default 2).")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducible runs. Omit to use a random seed (printed in output).")
    parser.add_argument("--no-dice", action="store_true",
                        help="Disable the spike dice phase entirely. No doubles, no forced discards, no Spikes.")
    parser.add_argument("--dice-mode", choices=["classic","spike"], default="spike",
                        help="Dice variant. 'spike' (default): non-spike doubles trigger a forced discard+draw for each player; double-1 (Spike) wipes and redraws everyone's full hand. 'classic': any doubles wipe and redraw everyone's full hand.")
    parser.add_argument("--spike-cheat", action="store_true",
                    help="Interactive only. In spike mode, choose doubles (1-6) or no double (0) each dice phase.")
    parser.add_argument("--allow-discard-gain", action="store_true",
                        help="Allow taking the discard_top discard as an extra card (gain) during the card phase.")



    # starts
    parser.add_argument("--random-starts", action="store_true",
                        help="Deal each player a fresh random two-card starting hand once at the beginning of the run, then reuse the same starts for every game. Equivalent to the default behaviour; included for clarity when scripting.")
    parser.add_argument("--starts", type=str, default="",
                        help="Set starting hand values for specific seats. Syntax: 'p1:a,b;p3:c,d'. Unspecified seats get random starts. The same values are used for every game in the run.")
    parser.add_argument("--fixed-starts", type=str, default="",
                        help="Like --starts but named seats keep their fixed values every game while unspecified seats get a fresh random hand each game. Syntax: 'p1:a,b;p2:c,d'.")
    parser.add_argument("--randomize-all", action="store_true",
                        help="Deal completely fresh random starting hands to every player before each individual game. Overrides --starts and --fixed-starts.")

    # modes
    parser.add_argument("--all-modes", type=str, default=None,
                        help="Set every seat to the same bot mode. One of: draw, stand, drawtozero, minimizer, swaptozero, newmodes, hunter.")
    parser.add_argument("--modes", type=str, default=None,
                        help="Comma-separated bot mode per seat in order, e.g. 'drawtozero,minimizer,stand'. If fewer values than players are given, the last value is repeated. Default: drawtozero for all seats.")
    for i in range(1,9):
        parser.add_argument(
            f"--p{i}-mode",
            type=str,
            default=None,
            choices=["draw","stand","drawtozero","minimizer","swaptozero","newmodes"],
            help=f"Override the bot mode for seat {i} only. Takes precedence over --modes and --all-modes.",
        )

    parser.add_argument("--hunter-target", type=str, default="",
                        help="Map of seats to target named hand, e.g. p1:'Squadron',p3:'Rhylet'. Any seat listed here will run in hunter mode.")



    # rotation & human
    parser.add_argument("--rotate-first", action="store_true",
                        help="Rotate which seat acts first each game (seat 1 → seat 2 → ... → seat N → seat 1 → ...). By default seat 1 always goes first.")
    parser.add_argument("--human-player", type=int, default=None,
                        help="Make seat N (1-indexed) interactive: the simulator pauses each round and prompts you to choose an action (stand, draw, swap, or gain from discard). All other seats play as bots. Example: --human-player 1")

    # named system
    parser.add_argument("--named-order", type=str, default="default", choices=list(NAMED_ORDERS.keys()),
                        help="Named hand ranking table to use. 'default' follows corellian.spike.today; 'galedge' uses the Galaxy's Edge ordering; 'old' uses a legacy ordering. Affects which named hand beats which in a named-vs-named tie.")
    parser.add_argument("--named-low-wins", action="store_true",
                        help="When two players both hold named hands, the one with the lower rank key wins instead of the higher. This matches the Galaxy's Edge ruleset, where rarer hands lose to commoner ones in a named-vs-named tie. The default (higher key wins) follows the casino-ready corellian.spike.today rules.")

    # tiebreak controls
    parser.add_argument("--no-high-abs", action="store_true",
                        help="Disable the highest-|card| tiebreak step. By default, when two hands tie on Σ|cards|, the hand with the single highest absolute card value wins. Pass this flag to skip that step and proceed directly to highest positive card.")
    parser.add_argument("--no-suits", action="store_true",
                        help="Disable the suited tiebreak step. By default, when hands are still tied after all card-value tiebreakers, a hand whose non-zero cards all share the same suit (Circle, Square, or Triangle) wins. Pass this flag to skip that step and proceed directly to the single-card draw.")

    # developer helper
    parser.add_argument("--compare", type=str, default=None,
                        help="Compare two hands and print the winner. Syntax: 'h1:v1,v2;h2:v1,v2,v3'. Example: 'h1:5,-5;h2:3,-3,0'")

    # Memory/minimal-replay options
    parser.add_argument("--keep-games", type=int, default=8, help="Keep full replay data for the first K games in memory (default 8)")
    parser.add_argument("--no-replay", action="store_true", help="Disable ALL replay capture (memory + file); overrides other replay options")
    parser.add_argument("--replay-file", type=str, default=None, help="[EXPERIMENTAL] Path to write per-game replay data including suits (JSONL or Parquet). Schema may change between versions.")
    parser.add_argument("--replay-format", type=str, default="jsonl", choices=["jsonl","parquet"], help="[EXPERIMENTAL] Replay file format. 'parquet' requires pyarrow.")
    parser.add_argument("--replay-size", type=str, default=None, help="[EXPERIMENTAL] How many games to write to the replay file: N or 'all'. Defaults to --keep-games when not given.")
    
    args = parser.parse_args()

    # Seed
    if args.seed is None:
        seed = random.getrandbits(63)
        seed_given = False
    else:
        seed = int(args.seed)
        seed_given = True
    rng = random.Random(seed)

    # Compare helper
    if args.compare:
        # format: "h1:a,b;h2:c,d,e"
        h = {}
        for segment in args.compare.split(";"):
            segment = segment.strip()
            if not segment:
                continue
            if ":" not in segment:
                raise SystemExit(f"--compare: malformed segment '{segment}' (expected 'hN:v1,v2,...')")
            key, rest = segment.split(":", 1)
            key = key.strip().lower()
            try:
                vals = [int(x.strip()) for x in rest.split(",") if x.strip()]
            except ValueError:
                raise SystemExit(f"--compare: non-integer value in '{rest}'")
            h[key] = [Card(val, None) for val in vals]
        # build opts
        opts = {
            "named_order": args.named_order,
            "named_low_wins": args.named_low_wins,
            "high_abs": not args.no_high_abs,
            "use_suits": not args.no_suits,
        }
        counters = RunCounters()
        # Build a draw pile from the full deck minus the cards in the hands,
        # so the single-card draw tiebreaker has cards to work with.
        all_hands = [h.get("h1", []), h.get("h2", [])]
        used_vals = [c.v for hand in all_hands for c in hand]
        draw = build_full_deck()
        remove_values_from_deck(rng, draw, used_vals)
        rng.shuffle(draw)
        discard = [draw.pop()] if draw else []
        winner, reason, _ = winner_and_reason(all_hands, opts, rng, discard, draw, counters, None)
        print(f"Winner: h{winner+1} ({reason})")
        return

    # Validate starts options
    num_players = args.num_players
    fixed_starts = {}
    starts_run_vals = []
    random_starts = False

    if args.randomize_all:
        pass  # per-game
    elif args.fixed_starts:
        fixed_starts = parse_fixed_starts_string(args.fixed_starts)
        # validate seats
        for seat in fixed_starts.keys():
            if seat < 1 or seat > num_players:
                raise SystemExit(f"--fixed-starts seat out of range: p{seat}")
    elif args.starts:
        starts_map = parse_starts_string(args.starts)
        # Build starts list in seat order, others random (once per run if --random-starts else default random once)
        starts_run_vals = []
        for i in range(1, num_players+1):
            if i in starts_map:
                starts_run_vals.append(starts_map[i])
            else:
                starts_run_vals.append(None)
        random_starts = True
    else:
        # default random starts once per run
        random_starts = True
        starts_run_vals = [None]*num_players

    # If random_starts true: generate random starts ONCE per run
    if random_starts and not args.randomize_all and not fixed_starts:
        deck = build_full_deck()
        rng.shuffle(deck)
        # fill only the None entries; preserve any seats provided via --starts
        new_vals = []
        for i in range(num_players):
            if i < len(starts_run_vals) and starts_run_vals[i] is not None:
                new_vals.append(starts_run_vals[i])
            else:
                c1 = deck.pop(); pair_signs = deck.pop()
                new_vals.append([c1.v, pair_signs.v])
        starts_run_vals = new_vals

    # Modes
    modes_run = parse_modes(args, num_players)

    # Options dict
    
    opts = {
        "n_games": args.n_games,
        "num_players": num_players,
        "seed_given": seed_given,
        "no_dice": args.no_dice,
        "dice_mode": args.dice_mode,
        "spike_cheat": args.spike_cheat,
        "allow_discard_gain": args.allow_discard_gain,
        "random_starts": random_starts,
        "randomize_all": args.randomize_all,
        "rotate_first": args.rotate_first,
        "human_player": args.human_player,
        "named_order": args.named_order,
        "named_low_wins": args.named_low_wins,
        "high_abs": not args.no_high_abs,
        "use_suits": not args.no_suits,
        "keep_games": args.keep_games,
        "no_replay": args.no_replay,
        "replay_file": args.replay_file,
        "replay_format": args.replay_format,
        "replay_size": args.replay_size,
    }


    # State baseline
    state = {
        "rng": rng,
        "opts": opts,
        "counters": RunCounters(),
        "recycle": engine_recycle,
        "abort": engine_abort,
        "interactive": (args.human_player is not None),
        "known_in_hand_by_player": {i: [] for i in range(num_players)},
        "fixed_starts": fixed_starts,
        "starts_run_vals": starts_run_vals,
        "modes_run": modes_run,
        "start_seat_index": 0,
        "record_replay": False,  # set per run if n<9

    }
    # Build replay file writer (disabled by --no-replay)
    replay_writer = None
    if (not opts["no_replay"]) and opts.get("replay_file"):
        # Default replay-size to keep-games if not provided
        size_spec = opts.get("replay_size") if opts.get("replay_size") not in (None, "") else str(opts.get("keep_games", 8))
        replay_writer = ReplayFileWriter(opts["replay_file"], opts.get("replay_format","jsonl"), size_spec)
    state["replay_writer"] = replay_writer



    all_games = []
    per_player_stats = [{
        "wins": 0,
        "exact_zero": 0,
        "zero_wins": 0,
        "named_wins": Counter(),
        "named_totals": Counter(),
        "mode_label": mode_label_str(modes_run[i]),
        "end_sum_total": 0,
        "end_sum_count": 0,
    } for i in range(num_players)]

    # named aggregation
    named_agg = {
        "overall": Counter(),
        "wins": Counter(),
        "shared": Counter(),
        "constellations": Counter(),
        "single_games": Counter(),
    }
    # (record_replay is set per game using --keep-games / --no-replay)
    # Interactive header
    if state["interactive"]:
        print_interactive_header(opts, seed, modes_run)

    # Track starts behaviour without storing every game
    starts_first = None
    starts_same = None

    keep = int(opts.get("keep_games", 8) or 0)

    for game_index in range(1, opts["n_games"]+1):
        # rotate seat
        state["start_seat_index"] = (game_index-1) % opts["num_players"] if opts["rotate_first"] else 0

        # Set replay flags for this game
        state["record_replay"] = (not opts.get("no_replay", False)) and (game_index <= keep)
        wr = state.get("replay_writer")
        will_file = (wr is not None) and wr.will_write(game_index) and (not opts.get("no_replay", False))
        state["record_file_replay"] = will_file

        # Track total games
        state["counters"].total_games += 1

        # Run game
        state["game_num"] = game_index
        game_result = play_one_game(state)

        # Only keep full game objects for interactive runs or the first `keep` games
        if game_index <= keep:
            all_games.append(game_result)

        # Track starts for summary (without needing all_games)
        if starts_first is None:
            starts_first = game_result["starts"]
            starts_same = [True] * len(starts_first)
        else:
            s = game_result["starts"]
            for i in range(min(len(starts_first), len(s))):
                if starts_same[i] and s[i] != starts_first[i]:
                    starts_same[i] = False

        # Update dice counters
        d, sd = game_result["dice_stats"]
        state["counters"].doubles_hist[d] += 1
        state["counters"].spike_doubles_hist[sd] += 1

        # Update per-player stats
        winner_seat = game_result["winner"]
        per_player_stats[winner_seat]["wins"] += 1
        # Per-player stats and named agg across game
        names_in_game = []
        for i,(vals,s,name) in enumerate(game_result["finals"]):
            per_player_stats[i]["end_sum_total"] += abs(s)
            per_player_stats[i]["end_sum_count"] += 1
            if s == 0:
                per_player_stats[i]["exact_zero"] += 1
                if i == winner_seat:
                    per_player_stats[i]["zero_wins"] += 1
            if name:
                per_player_stats[i]["named_totals"][name] += 1
                if i == winner_seat:
                    per_player_stats[i]["named_wins"][name] += 1
                names_in_game.append(name)
                named_agg["overall"][name] += 1
                if i == winner_seat:
                    named_agg["wins"][name] += 1
            else:
                # Pseudo-named categories:
                #   Sabacc  = non-named zero-sum hand
                #   Nulrhek = non-zero non-named hand
                pseudo = "Sabacc" if s == 0 else "Nulrhek"
                named_agg["overall"][pseudo] += 1
                if i == winner_seat:
                    named_agg["wins"][pseudo] += 1

        # shared counts
        c = Counter(names_in_game)
        for named_hand, ct in c.items():
            if ct >= 2:
                named_agg["shared"][f"{named_hand} ({ct} players)"] += 1

        # Single named hand in this game (no other named present)
        if len(names_in_game) == 1:
            named_agg["single_games"][names_in_game[0]] += 1
        # constellations: sorted list with multiplicities (e.g., Pair×2 + Rule of Two)
        if len(names_in_game) >= 2:
            names_sorted = sorted(names_in_game)
            # Add multiplicity "xN" if >1
            const = []
            counted = Counter(names_sorted)
            for named_hand in sorted(set(names_sorted)):
                cnt = counted[named_hand]
                if cnt == 1:
                    const.append(named_hand)
                else:
                    const.append(f"{named_hand}×{cnt}")
            label_key = tuple(const)
            named_agg["constellations"][label_key] += 1

        # Interactive: after each interactive game, immediately show Results block per template
        if state["interactive"]:
            pause_before_results()
            print_results_game_block(game_index, game_result, opts)
            pause_before_results()
# After run, print summary
    print()
    print_simulation_summary(all_games, opts, seed, state["counters"], per_player_stats,
                             named_agg, starts_first, starts_same)

    # Print replays for the first --keep-games (unless --no-replay)
    if (not opts.get("no_replay", False)) and int(opts.get("keep_games",8) or 0) > 0:
        print_replays(all_games, opts)

    # Elapsed time footer
    if RUN_START_TIME is not None:
        elapsed = time.perf_counter() - RUN_START_TIME
        total_ms = int(round(elapsed * 1000))
        ms = total_ms % 1000
        total_s = total_ms // 1000
        s = total_s % 60
        total_m = total_s // 60
        m = total_m % 60
        h = total_m // 60
        if h:
            tstr = f"{h:d}:{m:02d}:{s:02d}.{ms:03d}"
        else:
            tstr = f"{m:02d}:{s:02d}.{ms:03d}"
        print(f"Elapsed time: {tstr}")
        print()


    # Close replay writer if any
    if state.get("replay_writer"):
        try:
            state["replay_writer"].close()
        except Exception:
            pass
if __name__ == "__main__":
    main()
