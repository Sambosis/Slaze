# WarThink

WarThink is a turn-based strategy game on a 10x10 grid, designed for training an AI agent using reinforcement learning. Two players, each controlling a small army of 5 units, compete to be the last one standing. The project uses `stable-baselines3` with a PPO (Proximal Policy Optimization) model to learn the game, featuring asymmetric self-play and a policy pool for diverse opponents.

![WarThink Gameplay](https://i.imgur.com/example.png)  <!-- Replace with actual gameplay image -->

## Gameplay

The objective of the game is to eliminate all of the opponent's units.

- **Game Board:** A 10x10 grid.
- **Players:** Two players, Player 1 (Blue) and Player 2 (Red).
- **Turns:** The game is turn-based. It ends if one player has no units left.
- **Turn Limit & Tie-Breaker:** If the game reaches 200 turns, a tie-breaker is triggered. The player with more units remaining on the board is declared the winner. If both players have the same number of units, the game is a true draw.
- **Armies:** Each player starts with an army of 5 units in opposite corners of the board.

### Units

There are three types of units, each with unique stats:

| Unit Type | Health (HP) | Attack Damage | Attack Range | Move Range |
| :--- | :--- | :--- | :--- | :--- |
| **Warrior** | 3 | 1 | 1 | 1 |
| **Archer** | 3 | 1 | 3 | 1 |
| **Commander**| 5 | 2 | 1 | 2 |

Each army consists of:
- 2x Warriors
- 2x Archers
- 1x Commander

## Actions

On each turn, the active player issues one command to each of their 5 units simultaneously.

### Action Types

1.  **Stay (Action 0):** The unit holds its position.
2.  **Move (Actions 1-4):** Move North, South, East, or West.
    - Warriors and Archers move 1 square.
    - Commanders move 2 squares.
    - **Collision Rule:** If multiple friendly units attempt to move to the exact same square, none of them will move.
3.  **Attack (Actions 5-8):** Attack in one of the four cardinal directions (North, South, East, West).
    - The attack hits the **first** enemy unit encountered in a straight line in that direction, up to the unit's maximum range.

## Visuals & UI

The game is rendered in an isometric view.

- **Player Colors:** Player 1 is **Blue**, and Player 2 is **Red**.
- **Unit Appearance:**
    - Units are represented by diamond shapes with a surrounding "glow" of their player's color.
    - A health bar is displayed above each unit.
    - A small circle in the center of the unit indicates its type:
        - **Gold Circle:** Commander
        - **White Circle:** Warrior or Archer

- **UI Overlays:**
    - The top-left corner displays the current turn count and active player.
    - During evaluation phases, training statistics like the current episode, average reward, and P1's win rate are shown.
    - At the end of a match, the winner (`Player 1 WINS!`, `Player 2 WINS!`, or `DRAW!`) is displayed prominently in the center of the screen.

## Reinforcement Learning Environment

The AI agent is trained using the Gymnasium API.

### Observation Space

The agent perceives the game state as a `10x10x5` tensor:
- **Channel 0:** Turn Indicator (1.0 for the current player).
- **Channel 1:** Player 1's unit types (normalized).
- **Channel 2:** Player 1's unit HP (normalized).
- **Channel 3:** Player 2's unit types (normalized).
- **Channel 4:** A combination of Player 2's unit HP and a mask indicating valid move/attack locations for the current player.

### Action Space

The agent outputs an action for each of its 5 units, represented by a `MultiDiscrete([9] * 5)` space. Each of the 5 actions is an integer from 0 to 8, mapping to either a "Stay/Move" or "Attack" command.

### Reward System

The agent is incentivized to learn through a system of rewards and penalties designed to be symmetric for both players.

- **Dense Rewards (during gameplay):**
    - `+0.1` for each point of damage dealt to an enemy unit.
    - `-0.1` for each point of damage taken from an enemy unit on their previous turn.
    - `-0.001` small penalty for each turn taken (to encourage efficiency).

- **Terminal Rewards (at the end of the game):**
    - `+10.0` for winning the game.
    - `-10.0` for losing the game.
    - `0.0` for a draw.

## How to Run

1.  **Installation:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Training:**
    To start the training process, run:
    ```bash
    python main.py
    ```
    - The agent will train and periodically save model checkpoints to the `models/` directory.
    - Every 100 episodes, an evaluation game will be rendered on-screen. You can close the window or press `ESC` to resume training.

3.  **Watching a Pre-trained Model:**
    (Instructions to be added on how to load and watch a specific model)
