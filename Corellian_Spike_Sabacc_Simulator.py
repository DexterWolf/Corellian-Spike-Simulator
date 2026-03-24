
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
    --replay-file PATH      [EXPERIMENTAL] Write per-hand replay to JSONL or Parquet
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
  suited tiebreaker (disable with ``--no-suits``).
- Bot policies mutate game state directly, then return an action string.
- For the human seat the engine performs the mutation after reading the
  chosen action.
"""

from sabacc_sim import main

if __name__ == "__main__":
    main()
