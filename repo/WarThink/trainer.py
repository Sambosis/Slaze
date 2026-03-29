import os
import re
import pygame
import numpy as np
from typing import List, Optional, Tuple
from rich import print as rr

from env import WarGameEnv
from rl import PPOSelfPlayAgent
from renderer import Renderer
from recorder import record_pygame

class Trainer:
    def __init__(self, load_model_path: Optional[str] = None):
        self.env = WarGameEnv()
        self.agent = PPOSelfPlayAgent(self.env)
        self.renderer = Renderer()
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_winners = []
        os.makedirs('models', exist_ok=True)

        if load_model_path:
            if os.path.exists(load_model_path):
                print(f"Loading model from: {load_model_path}")
                self.agent.load(load_model_path)
                
                # Try to parse episode count from filename
                match = re.search(r'_(\d+)\.zip', load_model_path)
                if match:
                    self.episode_count = int(match.group(1))
                    print(f"Resuming from episode {self.episode_count}")
            else:
                print(f"Warning: Model path not found, starting new training: {load_model_path}")

    def get_recent_stats(self, window: int = 100):
        import numpy as np
        if len(self.episode_rewards) < window:
            return 0.0, 0.0
        recent_rewards = np.array(self.episode_rewards[-window:])
        recent_winners = self.episode_winners[-window:]
        avg_reward = np.mean(recent_rewards)
        p1_win_rate = np.mean([w == 1 for w in recent_winners])
        return avg_reward, p1_win_rate

    def play_episode(self):
        obs, _ = self.env.reset()
        self.agent.start_episode()
        p1_total_reward = 0.0
        p2_total_reward = 0.0
        done = False
        while not done:
            player = self.env.state.current_player
            actions = self.agent.act(obs, player)
            obs, reward, terminated, truncated, _ = self.env.step(actions)
            done = terminated or truncated
            if player == 1:
                p1_total_reward += reward
            else:
                p2_total_reward += reward
        # --- End of episode loop ---

        # Add terminal rewards symmetrically based on win condition
        winner = self.env.state.winner
        win_condition = self.env.state.win_condition

        if winner in [1, 2]:
            # Annihilation wins get a larger bonus
            bonus = 50.0 if win_condition == 'annihilation' else 10.0
            rr(f"[bold yellow]Applying terminal reward bonus player {winner} for win condition: {win_condition}[/bold yellow]")
            if winner == 1:
                p1_total_reward += bonus
                p2_total_reward -= bonus
            else: # winner == 2
                p2_total_reward += bonus
                p1_total_reward -= bonus
        
        # Log the final total reward (sum of dense + terminal rewards)
        final_total_reward = p1_total_reward + p2_total_reward
        self.episode_rewards.append(final_total_reward)
        
        self.episode_winners.append(winner)
        self.agent.update_pool([p1_total_reward, p2_total_reward])

    def eval_render(self):
        print("Starting evaluation render at 10 FPS (close/ESC to resume training)...")
        obs, _ = self.env.reset()
        self.agent.start_episode()
        done = False
        with record_pygame(f"videos/WarWatch_{self.episode_count}.mp4", fps=10):
            while not done:
                if not self.renderer.handle_events():
                    print("Evaluation interrupted by user.")
                    return
                player = self.env.state.current_player
                actions = self.agent.act(obs, player)
                obs, reward, terminated, truncated, _ = self.env.step(actions)
                done = terminated or truncated
                self.renderer.render(self.env.state)
                avg_reward, win_rate = self.get_recent_stats(100)
                self.renderer.draw_stats(
                    self.episode_count, avg_reward, win_rate,
                    self.env.state.turn_count, self.env.state.winner
                )
                self.renderer.clock.tick(10)
        print("Evaluation complete.")

    def train(self, max_episodes=float('inf')):
        print('Starting asymmetric self-play PPO training...')
        print(f"Device: {self.agent.model.device}")
        try:
            while self.episode_count < max_episodes:
                self.play_episode()
                self.episode_count += 1
                if self.episode_count % 10 == 0:
                    avg_reward, p1_win_rate = self.get_recent_stats(100)
                    print(f'Episode {self.episode_count:4d} | Avg reward (100): {avg_reward:6.2f} | P1 win rate (100): {p1_win_rate:.1%}')
                if self.episode_count % 5 == 0:
                    print(f"Quick learn at ep {self.episode_count}...")
                    self.agent.learn(total_timesteps=1024)
                if self.episode_count % 100 == 0:
                    print(f"\n--- Episode {self.episode_count} ---")
                    print("Evolving policy pool...")
                    self.agent.evolve_pool()
                    print("Performing intensive PPO learning...")
                    self.agent.learn(total_timesteps=10000)
                    print("Rendering evaluation game...")
                    self.eval_render()
                if self.episode_count % 500 == 0:
                    checkpoint_path = f"models/checkpoint_{self.episode_count}.zip"
                    self.agent.save(checkpoint_path)
                    print(f"Checkpoint saved: {checkpoint_path}")
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            if self.episode_count > 0:
                checkpoint_path = f"models/interrupt_checkpoint_{self.episode_count}.zip"
                self.agent.save(checkpoint_path)
                print(f"Final checkpoint saved: {checkpoint_path}")
def eval_render(self) -> None:
    """
    Render a full self-play evaluation episode at 10 FPS using current agent/pool.
    Pauses until window closed or ESC pressed.
    """
    print("Starting evaluation render (close window or press ESC to continue)...")
    eval_env = WarGameEnv()
    obs, _ = eval_env.reset()
    self.agent.start_episode()
    avg_reward, win_rate = self.get_recent_stats()
    running = True
    terminated = truncated = False
    while running and not (terminated or truncated):
        running = self.renderer.handle_events()
        if not running:
            break
        self.renderer.render(eval_env.state)
        self.renderer.draw_stats(
            episode=self.episode_count,
            avg_reward=avg_reward,
            win_rate=win_rate,
            turn_count=eval_env.state.turn_count,
            winner=eval_env.state.winner
        )
        self.renderer.clock.tick(2)
        player = eval_env.state.current_player
        actions = self.agent.act(obs, player)
        obs, reward, terminated, truncated, info = eval_env.step(actions)
    print("Evaluation render complete.")

    def save_checkpoint(self, suffix: str = '') -> None:
        path = f'models/checkpoint_ep{self.episode_count}{suffix}.zip'
        self.agent.save(path)
        print(f'Checkpoint saved: {path}')