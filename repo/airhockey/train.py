#!/usr/bin/env python3
"""
Training Module for Air Hockey RL Application

This module contains the training loop and coordination between environment and agents.
It handles periodic visualization using Pygame.
"""

import csv
import os
import subprocess
from collections import deque
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pygame

# Import internal modules
from environment import AirHockeyEnv
from agent import DQNAgent, DEVICE
from recorder import Recorder
from visualizer import init_pygame, draw_game
from dashboard import TrainingDashboard


def _build_episode_video_path(record_dir: str, episode: int) -> str:
    """Build a timestamped output path for a visualized episode recording."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"episode_{episode:06d}_{timestamp}.mp4"
    return os.path.join(record_dir, filename)


def _push_video_artifact(
    video_path: str,
    episode: int,
    remote: str = "origin",
    branch: Optional[str] = None,
) -> tuple[bool, str]:
    """Commit and push a recorded episode video to Git without crashing training."""
    repo_dir = os.getcwd()
    abs_video_path = os.path.abspath(video_path)

    if not os.path.exists(abs_video_path):
        return False, f"Video file does not exist: {abs_video_path}"

    try:
        relative_video_path = os.path.relpath(abs_video_path, repo_dir)
    except ValueError:
        relative_video_path = abs_video_path

    add_result = subprocess.run(
        ["git", "add", relative_video_path],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if add_result.returncode != 0:
        message = (add_result.stderr or add_result.stdout).strip() or "git add failed"
        return False, f"Could not stage video: {message}"

    commit_result = subprocess.run(
        ["git", "commit", "-m", f"Add visualization video for episode {episode}"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if commit_result.returncode != 0:
        commit_output = (commit_result.stderr or commit_result.stdout).strip()
        if "nothing to commit" in commit_output.lower():
            return True, f"No new video changes to commit for episode {episode}."
        return False, f"Could not commit video: {commit_output or 'git commit failed'}"

    push_command = ["git", "push", remote]
    if branch:
        push_command.append(branch)

    push_result = subprocess.run(
        push_command,
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if push_result.returncode != 0:
        message = (push_result.stderr or push_result.stdout).strip() or "git push failed"
        return False, f"Committed video but push failed: {message}"

    return True, f"Pushed video for episode {episode} to {remote}{'/' + branch if branch else ''}."


def train(
    num_episodes: int = 10000,
    visualize_every: int = 5,
    learning_rate: float = 0.001,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    batch_size: int = 64,
    memory_size: int = 10000,
    target_update_freq: int = 10,
    max_steps: int = 1000,
    learn_every: int = 4,
    resume: bool = False,
    resume_interrupted: bool = False,
    env_config: Optional[Dict[str, Any]] = None,
    reward_config: Optional[Dict[str, float]] = None,
    tau: float = 0.005,
    lr_min: float = 1e-6,
    lr_T0: int = 1000,
    lr_T_mult: int = 2,
    override_lr: Optional[float] = None,
    override_epsilon: Optional[float] = None,
    record: bool = False,
    record_dir: str = "videos",
    record_fps: Optional[int] = None,
    record_frame_skip: int = 0,
    push_videos: bool = False,
    git_remote: str = "origin",
    git_branch: Optional[str] = None,
) -> None:
    """
    Main training function for the air hockey RL agents.

    Args:
        num_episodes: Total number of training episodes
        visualize_every: Visualize every nth episode using Pygame
        learning_rate: Learning rate for DQN agents
        gamma: Discount factor for future rewards
        epsilon_start: Starting epsilon for exploration
        epsilon_end: Minimum epsilon value
        epsilon_decay: Epsilon decay rate per episode
        batch_size: Batch size for training
        memory_size: Replay memory buffer size
        target_update_freq: Update target network every n episodes
        max_steps: Maximum steps per episode before auto-reset
        learn_every: Number of steps between agent learning updates
        resume: Whether to resume training from saved checkpoint
        resume_interrupted: Whether to resume from manually interrupted checkpoint
        env_config: Configuration dictionary for the environment
        reward_config: Configuration dictionary for rewards
        tau: Soft update interpolation factor (Polyak averaging)
        lr_min: Minimum learning rate for cosine annealing
        lr_T0: Episodes in first cosine annealing cycle
        lr_T_mult: Cycle length multiplier after each restart
        override_lr: Optional learning rate to override the saved optimizer schedule
        override_epsilon: Optional epsilon to override the saved starting epsilon
        record: Whether to record visualized episodes to MP4
        record_dir: Directory where per-episode videos are written
        record_fps: Optional FPS override for the recorder
        record_frame_skip: Save every N+1th visualized frame
        push_videos: Whether to commit and push each completed video to Git
        git_remote: Git remote to push videos to
        git_branch: Optional Git branch to push videos to
    """
    print("\n" + "=" * 50)
    print("Starting Air Hockey RL Training")
    print("=" * 50)

    # Initialize the live TUI dashboard
    dash = TrainingDashboard(num_episodes=num_episodes)

    # Initialize environment
    env_config = env_config or {}
    reward_config = reward_config or {}
    env = AirHockeyEnv(**env_config, reward_config=reward_config)
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Device: {DEVICE}")

    # Initialize two DQN agents
    agent1 = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        memory_size=memory_size,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        learn_every=learn_every,
        tau=tau,
        lr_min=lr_min,
        lr_T0=lr_T0,
        lr_T_mult=lr_T_mult,
    )

    agent2 = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        memory_size=memory_size,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        learn_every=learn_every,
        tau=tau,
        lr_min=lr_min,
        lr_T0=lr_T0,
        lr_T_mult=lr_T_mult,
    )

    # Resume from checkpoint if requested
    start_episode = 1
    if resume or resume_interrupted:
        suffix = "interrupted" if resume_interrupted else "final"
        chk1 = f"agent1_{suffix}.pth"
        chk2 = f"agent2_{suffix}.pth"
        
        if os.path.exists(chk1) and os.path.exists(chk2):
            try:
                agent1.load_model(chk1, override_lr=override_lr, override_epsilon=override_epsilon)
                agent2.load_model(chk2, override_lr=override_lr, override_epsilon=override_epsilon)
                print(f"Resumed training from {suffix} checkpoints.")
                if override_lr is not None:
                    print(f"Overrode learning rate to: {override_lr}")
                if override_epsilon is not None:
                    print(f"Overrode starting epsilon to: {override_epsilon}")
                print(f"Loaded epsilon: Agent1: {agent1.epsilon:.4f}, Agent2: {agent2.epsilon:.4f}")
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                print("Starting training from scratch.")
        else:
            print("No checkpoint files found. Starting training from scratch.")

    # Initialize Pygame for visualization immediately so the window is open
    screen, clock = init_pygame(env.width, env.height)
    pygame_initialized = True
    dash.log("[blue]Pygame visualizer started.[/blue]")

    # Training metrics
    episode_rewards1 = []
    episode_rewards2 = []
    episode_lengths = []
    losses1 = []
    losses2 = []
    wins1 = 0
    wins2 = 0
    ties = 0

    # Best reward tracking
    max_reward1 = float("-inf")
    max_reward2 = float("-inf")
    best_ep1 = 0
    best_ep2 = 0

    # Moving averages for logging – cached running sums avoid np.mean() per episode
    avg_window = 100
    recent_rewards1: deque = deque(maxlen=avg_window)
    recent_rewards2: deque = deque(maxlen=avg_window)
    recent_steps: deque = deque(maxlen=avg_window)
    recent_loss1: deque = deque(maxlen=avg_window)
    recent_loss2: deque = deque(maxlen=avg_window)
    recent_breakdown1: deque = deque(maxlen=avg_window)
    recent_breakdown2: deque = deque(maxlen=avg_window)
    _sum1 = 0.0
    _sum2 = 0.0
    _sum_steps = 0
    _sum_loss1 = 0.0
    _sum_loss2 = 0.0

    # Setup CSV logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(log_dir, f"training_log_{timestamp}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "episode",
            "reward1",
            "reward2",
            "length",
            "loss1",
            "loss2",
            "epsilon1",
            "epsilon2",
            "winner",
            "avg_reward1_100",
            "avg_reward2_100",
            "win_rate1",
            "win_rate2",
            "score1",
            "score2",
        ]
    )

    print(f"Logging training metrics to: {csv_path}")
    print("Starting dashboard…")

    dash.start()
    dash.log(f"[cyan]CSV log:[/cyan] {csv_path}")
    if resume:
        dash.log("[yellow]Resumed from checkpoint[/yellow]")

    episode_recorder: Optional[Recorder] = None
    episode_video_path: Optional[str] = None

    try:
        # Main training loop
        force_visualize_next = False
        for episode in range(start_episode, num_episodes + 1):
            # Determine if this episode should be visualized
            visualize = (episode % visualize_every == 0) or force_visualize_next
            if visualize:
                force_visualize_next = False

            # Reset environment
            state = env.reset()
            initial_score1 = env.score1
            initial_score2 = env.score2

            # Initialize episode variables
            total_reward1 = 0.0
            total_reward2 = 0.0
            episode_loss1 = 0.0
            episode_loss2 = 0.0
            ep_breakdown1 = {k: 0.0 for k in env.reward_config}
            ep_breakdown2 = {k: 0.0 for k in env.reward_config}
            step = 0
            done = False
            episode_recorder = None
            episode_video_path = None

            if visualize and record:
                episode_video_path = _build_episode_video_path(record_dir, episode)
                episode_recorder = Recorder(
                    path=episode_video_path,
                    fps=record_fps,
                    frame_skip=record_frame_skip,
                )
                dash.log(f"[cyan]Recording episode {episode:,} to[/cyan] {episode_video_path}")

            try:
                # Run episode
                while not done and step < max_steps:
                    # Check for Pygame quit and keep window responsive
                    if pygame_initialized:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                print("\nTraining interrupted by user.")
                                return
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_ESCAPE:
                                    print("\nTraining interrupted by user.")
                                    return
                                if event.key == pygame.K_v:
                                    force_visualize_next = True
                                    dash.log("[blue]Will visualize next episode due to 'v' key press.[/blue]")

                    # Agents select actions
                    action1 = agent1.select_action(state)
                    action2 = agent2.select_action(state)

                    # Execute actions in environment
                    next_state, reward1, reward2, done, info = env.step(action1, action2)

                    b1 = info.get('reward_breakdown1', {})
                    b2 = info.get('reward_breakdown2', {})
                    for k, v in b1.items():
                        ep_breakdown1[k] = ep_breakdown1.get(k, 0.0) + v
                    for k, v in b2.items():
                        ep_breakdown2[k] = ep_breakdown2.get(k, 0.0) + v

                    # Store experiences in replay buffers
                    agent1.remember(state, action1, reward1, next_state, done)
                    agent2.remember(state, action2, reward2, next_state, done)

                    # Perform learning step for both agents
                    loss1 = agent1.learn()
                    loss2 = agent2.learn()

                    # Accumulate metrics
                    total_reward1 += reward1
                    total_reward2 += reward2
                    episode_loss1 += loss1 if loss1 is not None else 0.0
                    episode_loss2 += loss2 if loss2 is not None else 0.0

                    # Update state
                    state = next_state
                    step += 1

                    # Render if visualizing
                    if visualize:
                        draw_game(
                            screen=screen,
                            env=env,
                            episode=episode,
                            step=step,
                            score1=env.score1,
                            score2=env.score2,
                            agent1_epsilon=agent1.epsilon,
                            agent2_epsilon=agent2.epsilon,
                        )

                        if episode_recorder is not None:
                            episode_recorder.capture(screen)

                        clock.tick(60)
            finally:
                if episode_recorder is not None:
                    episode_recorder.close()
                    dash.log(f"[green]Saved video:[/green] {episode_video_path}")
                    if push_videos and episode_video_path:
                        ok, message = _push_video_artifact(
                            episode_video_path,
                            episode,
                            git_remote,
                            git_branch,
                        )
                        color = "green" if ok else "red"
                        dash.log(f"[{color}]{message}[/{color}]")
                    episode_recorder = None

            # Episode completed - decay epsilon and step LR scheduler
            agent1.update_epsilon()
            agent2.update_epsilon()
            agent1.step_lr_scheduler()
            agent2.step_lr_scheduler()

            # Determine winner
            if env.score1 > initial_score1:
                wins1 += 1
                winner = "Agent1"
            elif env.score2 > initial_score2:
                wins2 += 1
                winner = "Agent2"
            else:
                ties += 1
                winner = "Tie"

            # Store episode metrics
            episode_rewards1.append(total_reward1)
            episode_rewards2.append(total_reward2)
            episode_lengths.append(step)

            # Update best rewards
            if total_reward1 > max_reward1:
                max_reward1 = total_reward1
                best_ep1 = episode
            if total_reward2 > max_reward2:
                max_reward2 = total_reward2
                best_ep2 = episode

            avg_loss1 = episode_loss1 / step if step > 0 else 0.0
            avg_loss2 = episode_loss2 / step if step > 0 else 0.0
            losses1.append(avg_loss1)
            losses2.append(avg_loss2)

            # Update running sums (pop evicted value when window is full)
            recent_breakdown1.append(ep_breakdown1)
            recent_breakdown2.append(ep_breakdown2)
            if len(recent_rewards1) == avg_window:
                _sum1 -= recent_rewards1[0]
                _sum2 -= recent_rewards2[0]
                _sum_steps -= recent_steps[0]
                _sum_loss1 -= recent_loss1[0]
                _sum_loss2 -= recent_loss2[0]
            recent_rewards1.append(total_reward1)
            recent_rewards2.append(total_reward2)
            recent_steps.append(step)
            recent_loss1.append(avg_loss1)
            recent_loss2.append(avg_loss2)
            _sum1 += total_reward1
            _sum2 += total_reward2
            _sum_steps += step
            _sum_loss1 += avg_loss1
            _sum_loss2 += avg_loss2

            # Calculate moving averages from cached sums
            n = len(recent_rewards1)
            avg_reward1 = _sum1 / n if n > 0 else 0.0
            avg_reward2 = _sum2 / n if n > 0 else 0.0
            avg_loss1_window = _sum_loss1 / n if n > 0 else 0.0
            avg_loss2_window = _sum_loss2 / n if n > 0 else 0.0
            win_rate1 = wins1 / episode if episode > 0 else 0.0
            win_rate2 = wins2 / episode if episode > 0 else 0.0

            avg_bd1 = {k: sum(d.get(k, 0) for d in recent_breakdown1) / n for k in env.reward_config} if n > 0 else {}
            avg_bd2 = {k: sum(d.get(k, 0) for d in recent_breakdown2) / n for k in env.reward_config} if n > 0 else {}

            # Write to CSV
            csv_writer.writerow(
                [
                    episode,
                    total_reward1,
                    total_reward2,
                    step,
                    avg_loss1,
                    avg_loss2,
                    agent1.epsilon,
                    agent2.epsilon,
                    winner,
                    avg_reward1,
                    avg_reward2,
                    win_rate1,
                    win_rate2,
                    env.score1,
                    env.score2,
                ]
            )
            if episode % 100 == 0:
                csv_file.flush()  # Periodic flush (not every episode)

            # Update live dashboard every episode
            dash.update(
                {
                    "episode": episode,
                    "steps": step,
                    "avg_steps": _sum_steps / n if n > 0 else 0.0,
                    "reward1": total_reward1,
                    "reward2": total_reward2,
                    "avg_reward1": avg_reward1,
                    "avg_reward2": avg_reward2,
                    "avg_breakdown1": avg_bd1,
                    "avg_breakdown2": avg_bd2,
                    "loss1": avg_loss1,
                    "loss2": avg_loss2,
                    "avg_loss1": avg_loss1_window,
                    "avg_loss2": avg_loss2_window,
                    "epsilon1": agent1.epsilon,
                    "epsilon2": agent2.epsilon,
                    "mem1": len(agent1.memory),
                    "mem2": len(agent2.memory),
                    "wins1": wins1,
                    "wins2": wins2,
                    "ties": ties,
                    "score1": env.score1,
                    "score2": env.score2,
                    "best_reward1": max_reward1,
                    "best_reward2": max_reward2,
                    "best_ep1": best_ep1,
                    "best_ep2": best_ep2,
                    "lr": agent1.get_current_lr(),
                    "gamma": gamma,
                    "batch_size": batch_size,
                    "target_update_freq": target_update_freq,
                }
            )

            # Log milestone events
            if episode % 500 == 0:
                dash.log(
                    f"[white]ep {episode:,}[/] | "
                    f"wr1 [magenta]{win_rate1:.1%}[/] "
                    f"wr2 [green]{win_rate2:.1%}[/] "
                    f"avg-r1 [magenta]{avg_reward1:+.2f}[/] "
                    f"avg-r2 [green]{avg_reward2:+.2f}[/]"
                )

            # Save models periodically (every 1000 episodes)
            if episode % 1000 == 0:
                try:
                    agent1.save_model(f"agent1_episode_{episode}.pth")
                    agent2.save_model(f"agent2_episode_{episode}.pth")
                    dash.log(f"[green]✔ Models saved at episode {episode:,}[/green]")
                except Exception as e:
                    dash.log(f"[red]⚠ Could not save models: {e}[/red]")

        # Training completed
        dash.log("[bold green]🏁 Training complete![/bold green]")
        dash.stop()
        print("\n" + "=" * 50)
        print("Training Completed!")
        print("=" * 50)
        print(f"Total episodes: {num_episodes}")
        print(
            f"Final Win Rate: Agent1: {wins1 / num_episodes:.3f}, "
            f"Agent2: {wins2 / num_episodes:.3f}, Ties: {ties / num_episodes:.3f}"
        )
        print(f"Final epsilon: Agent1: {agent1.epsilon:.4f}, Agent2: {agent2.epsilon:.4f}")

        # Save final models
        try:
            agent1.save_model("agent1_final.pth")
            agent2.save_model("agent2_final.pth")
            print("Final models saved.")
        except Exception as e:
            print(f"Warning: Could not save final models: {e}")

        # Print training summary
        if episode_rewards1 and episode_rewards2:
            print("\nTraining Summary:")
            print(
                f"  Agent1 - Avg Reward: {np.mean(episode_rewards1):.3f}, "
                f"Max Reward: {np.max(episode_rewards1):.3f}"
            )
            print(
                f"  Agent2 - Avg Reward: {np.mean(episode_rewards2):.3f}, "
                f"Max Reward: {np.max(episode_rewards2):.3f}"
            )
            print(f"  Avg Episode Length: {np.mean(episode_lengths):.1f} steps")

    except KeyboardInterrupt:
        dash.log("[yellow]⚠ Training interrupted by user[/yellow]")
        dash.stop()
        print("\n\nTraining interrupted by user.")
        # Save models on interrupt
        try:
            agent1.save_model("agent1_interrupted.pth")
            agent2.save_model("agent2_interrupted.pth")
            print("Models saved on interrupt.")
        except Exception as e:
            print(f"Warning: Could not save models on interrupt: {e}")

    except Exception as e:
        dash.log(f"[bold red]✖ Error: {e}[/bold red]")
        dash.stop()
        print(f"\n\nAn error occurred during training: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Ensure dashboard is stopped (idempotent)
        dash.stop()
        # Close any still-open episode recorder
        if episode_recorder is not None:
            episode_recorder.close()
        # Close CSV file
        csv_file.close()
        print(f"Training log saved to: {csv_path}")

        # Clean up Pygame
        if pygame_initialized:
            pygame.quit()
            print("Pygame closed.")

        print("\nTraining process finished.")


def visualize_episode(
    env: AirHockeyEnv,
    agent1: DQNAgent,
    agent2: DQNAgent,
    screen: pygame.Surface,
    clock: pygame.time.Clock,
    episode: int,
    max_steps: int = 1000,
    recorder: Optional[Recorder] = None,
) -> Tuple[float, float, int]:
    """
    Visualize a single episode without training.

    Args:
        env: Air hockey environment
        agent1: First DQN agent
        agent2: Second DQN agent
        screen: Pygame screen surface
        clock: Pygame clock
        episode: Current episode number
        max_steps: Maximum steps per episode
        recorder: Optional recorder for captured frames

    Returns:
        Tuple of (total_reward1, total_reward2, steps)
    """
    state = env.reset()
    total_reward1 = 0.0
    total_reward2 = 0.0
    step = 0
    done = False

    while not done and step < max_steps:
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return total_reward1, total_reward2, step
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return total_reward1, total_reward2, step

        # Select actions (no exploration during visualization)
        action1 = agent1.select_action(state, explore=False)
        action2 = agent2.select_action(state, explore=False)

        # Execute actions
        next_state, reward1, reward2, done, info = env.step(action1, action2)

        # Update metrics
        total_reward1 += reward1
        total_reward2 += reward2
        state = next_state
        step += 1

        # Render
        draw_game(
            screen=screen,
            env=env,
            episode=episode,
            step=step,
            score1=env.score1,
            score2=env.score2,
            agent1_epsilon=agent1.epsilon,
            agent2_epsilon=agent2.epsilon,
        )

        if recorder is not None:
            recorder.capture(screen)

        # Cap frame rate
        clock.tick(60)

    return total_reward1, total_reward2, step


if __name__ == "__main__":
    # This allows running train.py directly for testing
    # Use default parameters
    train(
        num_episodes=1000,
        visualize_every=10,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        memory_size=10000,
        target_update_freq=10,
        max_steps=500,
        learn_every=4,
    )
