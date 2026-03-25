# Corellian Spike Sabacc Simulator

A Monte Carlo simulator for **Corellian Spike Sabacc**, a card game from the
Star Wars universe. Implements the casino-ready ruleset described at
[corellian.spike.today](https://corellian.spike.today) by default. The Galaxy's
Edge variant is also supported via CLI flags — see below.

## Game basics

- 62-card deck: values −10 to +10 in three suits (Circle, Square, Triangle),
  three copies per value/sign combination; plus 2 zero-value Sylops (no suit).
- 2–8 players, each starting with 2 cards dealt face-down.
- 3 rounds. Each round: card phase (each player acts once) → spike dice phase.
- Goal: hand summing to zero (Sabacc), or as close as possible (Nulrhek).
- Spike dice: non-1 doubles → forced discard+draw for each player;
  double-1 (Spike) → full hand wipe and redraw.
- Named hands (ranked by rarity): Full Sabacc, Fleet, Rhylet, Wild Rhylet,
  Gee Whiz!, Full Straight, Sylop Straight Khyron, Five Card Squad, Squadron,
  Sylop Rule of Two, Banthas Wild, Pure Sabacc, Straight Khyron, Idiots Rule,
  Rule of Two, Yee-Haa, Pair.
- Tiebreaker chain: named hand → most cards → highest Σ|cards| →
  highest |card| → highest positive card → suited → single-card draw.

## What this simulator does (and doesn't do)

**Betting is intentionally not implemented.** The goal is to measure pure hand
frequencies — how often each hand type occurs, how often each player wins, how
the dice phase affects outcomes — independently of any betting strategy or pot
size. This gives clean probabilities that aren't skewed by how well or badly
players bet. Interactive betting may be added in the future.

The bots play purely to win hands, not to manage a bankroll.

## Simulating Galaxy's Edge rules

The default ruleset follows [corellian.spike.today](https://corellian.spike.today).
To approximate the Galaxy's Edge variant instead, combine these flags:

```bash
python sabacc_sim.py --dice-mode classic --no-high-abs --no-suits --named-low-wins --named-order galedge
```

- `--dice-mode classic` — any doubles wipes all hands (no separate Spike
  doubles mechanic)
- `--no-high-abs` — disables the highest-|card| tiebreaker step
- `--no-suits` — disables the suited tiebreaker step
- `--named-low-wins` — lower-index (key) if same named hand wins ties
- `--named-order galedge` — uses the Galaxy's Edge named hand ranking order

## Requirements

Python 3.8+. No third-party packages required for core use.

Optional: `pyarrow` for Parquet replay output (`--replay-format parquet`).

## Usage

```
python Corellian_Spike_Sabacc_Simulator.py
```

Or equivalently:

```
python sabacc_sim.py
```

Run 10 000 games with 2 players (default settings). Pass `--help` for the full
flag list.

### Common examples

```bash
# 4 players, 50 000 games, fixed seed
python sabacc_sim.py -n 50000 --num-players 4 --seed 42

# Play interactively as seat 1 against three bots
python sabacc_sim.py --human-player 1 --num-players 4

# Mix bot modes
python sabacc_sim.py --num-players 3 --modes "hunter,minimizer,drawtozero" --hunter-target "p1:Pair"

# Compare two hands directly
python sabacc_sim.py --compare "h1:5,-5;h2:3,-3,0"

# Write a replay file
python sabacc_sim.py -n 1000 --replay-file replay.jsonl
```

### Bot modes

| Mode | Behaviour |
|---|---|
| `drawtozero` | Swap/gain to zero if possible; otherwise draw. Default. |
| `minimizer` | Like drawtozero but also takes any swap/gain that reduces \|sum\|. Stands at \|sum\| ≤ 1. |
| `swaptozero` | Like minimizer but stands only at exactly zero. |
| `draw` | Always draws. |
| `stand` | Always stands. |
| `newmodes` | Randomly picks minimizer/swaptozero/drawtozero each game. |
| `hunter` | Chases a specific named hand (set via `--hunter-target`). |

### Key flags

```
-n N                    Number of games (default 10000)
--num-players N         2–8 players
--seed N                RNG seed
--no-dice               Disable dice phase
--dice-mode spike|classic
--allow-discard-gain    Enable gain-from-discard action
--random-starts         Random starting hands once per run
--randomize-all         New random starts every game
--starts "p1:a,b;p2:c,d"
--modes "m1,m2,..."     Bot mode per seat
--all-modes M           Set all seats to mode M
--hunter-target "p1:'Squadron',p2:'Pair'"
--named-order default|old|galedge
--human-player N        Make seat N interactive
--compare "h1:5,-5;h2:3,-3,0"
--keep-games N          Keep full replay for first N games (default 8)
--replay-file PATH      Write replay to file (JSONL or Parquet)
```

## File structure

| File | Contents |
|---|---|
| `Corellian_Spike_Sabacc_Simulator.py` | Entry point (backwards-compatible wrapper) |
| `sabacc_sim.py` | CLI parsing and `main()` |
| `sabacc_engine.py` | Game engine, tiebreakers, replay |
| `sabacc_policies.py` | Bot policies, information tracking |
| `sabacc_hands.py` | Named hand detection, starting hand parsing |
| `sabacc_output.py` | All print/reporting functions |
| `sabacc_cards.py` | `Card` dataclass, deck building, hand helpers |
| `sabacc_combos.py` | Precomputed named-hand combination lookup (used by hunter bot) |

## License

MIT — see [LICENSE](LICENSE).
