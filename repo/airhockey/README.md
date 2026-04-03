# AirHockey RL

Self-play reinforcement learning project where **two DQN agents** learn to play 2D air hockey against each other.

## Overview

This project trains two independent agents in the same environment:

- **Environment**: 2D air hockey simulation with puck/paddle physics and goal detection
- **Agents**: Double DQN-style training with replay buffers, target networks, epsilon-greedy exploration
- **Training loop**: Episode-based self-play with checkpointing and CSV logs
- **Visualization**: Optional live Pygame rendering
- **Dashboard**: Rich TUI metrics during training

## Project Structure

- `main.py` — CLI entrypoint; loads `config.json`, parses args, and starts training
- `train.py` — Core training orchestration and logging/checkpoint behavior
- `environment.py` — Air hockey physics, state/action spaces, rewards
- `agent.py` — DQN agent implementation and optimization logic
- `visualizer.py` — Pygame initialization and drawing helpers
- `dashboard.py` — Terminal live dashboard using `rich`
- `config.json` — Tuned training/environment/reward defaults

## Requirements

- Python **3.12+**
- Dependencies:
  - `numpy`
  - `pygame`
  - `rich`
  - `torch`
- `imageio`

## Setup

### Option A: using `uv` (recommended)

```bash
uv sync
```

### Option B: using `venv` + `pip`

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -U numpy pygame rich torch imageio
```

## Run

From the project root (`repo/airhockey`):

```bash
python main.py
```

If you installed with `uv`, you can also run:

```bash
uv run python main.py
```

## Common CLI Examples

Train for fewer episodes:

```bash
python main.py --num_episodes 2000 --visualize_every 200
```

Resume from final checkpoints:

```bash
python main.py --resume
```

Resume and override saved LR/epsilon:

```bash
python main.py --resume --override_lr 0.0001 --override_epsilon 0.2
```

Show all options:

```bash
python main.py --help
```

## Default Configuration

`config.json` is used as the primary defaults source. Current key defaults include:

- Training: `num_episodes=10000`, `visualize_every=1000`, `max_steps=2000`
- DQN: `learning_rate=0.0003`, `gamma=0.995`, `batch_size=512`, `memory_size=500000`
- Exploration: `epsilon_start=1.0`, `epsilon_end=0.01`, `epsilon_decay=0.9997`
- Rewards: `goal=+10`, `concede=-5`, `hit_puck=+2`, plus shaping terms

CLI flags override these values at runtime.

## Outputs and Artifacts

Training produces:

- Checkpoints like `agent1_episode_1000.pth`, `agent2_episode_1000.pth`
- Final models: `agent1_final.pth`, `agent2_final.pth`
- CSV logs in `logs/` (timestamped)
- Optional recorded MP4 videos in `videos/` by default

## Recording Visualized Episodes

You can record each visualized episode to a timestamped MP4 file.

Record every visualized episode:

```bash
python main.py --visualize_every 1 --record
```

Record and write videos into a custom directory:

```bash
python main.py --visualize_every 1 --record --record_dir videos
```

Record with a custom frame skip to reduce file size:

```bash
python main.py --visualize_every 1 --record --record_frame_skip 1
```

Each saved file uses a timestamped per-episode name like:

```text
videos/episode_000100_YYYYmmdd_HHMMSS.mp4
```

## Remote Access via GitHub

If you want each completed visualization to be accessible remotely through GitHub while training is running elsewhere, enable video pushing:

```bash
python main.py --visualize_every 1 --record --push_videos
```

You can also specify the Git remote or branch:

```bash
python main.py --visualize_every 100 --record --record_dir videos --push_videos --git_remote origin --git_branch main
```

With `--push_videos` enabled, the training loop will:

- create one timestamped MP4 per visualized episode,
- close the file when that visualization ends,
- commit that MP4,
- and push it to the configured Git remote.

This is convenient for remote review, but MP4 files are binary artifacts and can grow the repository quickly over time.

## Notes

- Press **`v`** during training to force visualization on the next episode.
- Press **Esc** or close the Pygame window to stop visualization/training loops cleanly.
