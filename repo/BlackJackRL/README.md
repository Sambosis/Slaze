```markdown
# BlackjackBetOptimizer

RL agent learns optimal bet sizing in blackjack using Hi-Lo card counting and bankroll management. All playing decisions (hit/stand/double/split) strictly follow a pre-defined basic strategy table. Pygame provides real-time visualization during training and demo mode.

## Features
- **Hi-Lo card counting**: Running/true count tracking, reshuffle at 75% penetration (configurable)
- **Tabular Q-learning**: State=(true_count_bin[-10..+10], bankroll_frac_bin[0..20 in 5% steps]); Actions=[1x,2x,4x,6x,8x,10x,12x] min_bet multipliers
- **Basic strategy**: Hard-coded table (dealer stands on soft 17, double any two cards, splits up to 3x, aces once)
- **Pygame visualization**: Felt table, text-based cards, real-time stats (bet, bankroll, counts, strategy decision), auto-advance
- **Training**: 100k+ hands (~10k/sec), epsilon-greedy (0.5→0.01), stats every 10 hands, viz every 50
- **Persistence**: Q-table (`model.pkl`), stats (`stats.json`), matplotlib bankroll plot
- **Bankroll management**: Starts at $10k, ruin if < min_bet, bets clamped ≤20% bankroll

## Setup
```bash
# Create virtual environment
uv venv  # or python -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -r requirements.txt  # or pip install -r requirements.txt
```

**requirements.txt:**
```
pygame
numpy
pandas
matplotlib
```

## Run
```bash
# Train RL agent (default: 100k hands, viz every 50, auto-plot)
python main.py --train

# Demo infinite hands with visualization (requires trained model)
python main.py --load

# Custom training
python main.py --train --num_training_hands 10000 --viz_every 10 --initial_bankroll 5000 --max_spread 8

# Frequent viz in demo
python main.py --load --viz_every 5
```

**Visualization controls:**
- **ESC**: Quit current hand
- **Auto-advance**: ~10s/hand during training viz
- **Ctrl+C**: Stop training/demo

## Files
| File       | Purpose                          |
|------------|----------------------------------|
| `main.py`  | Entry: argparse, train/demo loop |
| `game.py`  | Logic: `Shoe`, `Card`, `Hand`, `Game`, `BasicStrategy` |
| `agent.py` | `BettingAgent`: Q-learning       |
| `viz.py`   | `Visualizer`: Pygame rendering   |
| `config.py`| `CONFIG` dict + CLI overrides    |

**Saved files:**
- `model.pkl`: Q-table + agent params
- `stats.json`: Bankroll history, win rates, avg bets
- Final matplotlib plot: Bankroll evolution curve

## Configuration
All params in `config.py`, overridable via CLI (e.g., `--initial_bankroll 5000`):
```python
CONFIG = {
    'initial_bankroll': 10000.0,
    'min_bet': 10.0,
    'max_spread': 12,
    'num_decks': 6,
    'penetration': 0.75,
    'num_training_hands': 100000,
    'viz_every': 50,
    'stats_every': 10,
    'epsilon_start': 0.5,
    'epsilon_end': 0.01,
    'alpha': 0.1,
    'gamma': 0.95,
}
```

## Example Training Output
```
Hands: 10,000 | Bankroll: $10,850 | Win Rate: 43.2% | Avg Bet: $28 | RoR: 0.12%
Hands: 20,000 | Bankroll: $11,420 | Win Rate: 44.1% | Avg Bet: $31 | RoR: 0.09%
...
Training complete! Final bankroll: $14,230 | Win rate: 45.3%
```

## Performance & Architecture
- **Speed**: ~10k hands/sec (no viz), 30 FPS viz
- **State space**: 21 × 21 = 441 states × 7 actions (sparse Q-table)
```
State: (true_count_bin, bankroll_frac_bin)
   ↓ (ε-greedy)
Action → bet = min(multi * min_bet, 0.2 * bankroll, spread * min_bet)
   ↓ Basic strategy → payout
Reward → Q(s,a) ← α [r + γ max Q(s',a') - Q(s,a)]
```

## Blackjack Rules
- 4-8 decks (default 6)
- Dealer stands on soft 17
- BJ pays 3:2
- Double any two cards
- Split up to 3x (aces once)
- No insurance/surrender
- Reshuffle at 75% penetration

Ready to run! 🚀
```