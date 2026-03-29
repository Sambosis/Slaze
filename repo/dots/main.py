import gymnasium.wrappers
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
import torch
import numpy as np
import sys
import pygame
import time
import random
from envs.dots_and_boxes import DotsAndBoxesEnv

CONFIG = {
    'grid_size': 3,
    'render_delay': 0.2,
    'seed': 42,
    'total_timesteps': 100000,
    'n_envs': 4,
    'render_every': 100,
}

def make_env():
    """Create a wrapped DotsAndBoxesEnv for training."""
    env = DotsAndBoxesEnv(grid_size=CONFIG['grid_size'], render_mode=None)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

def make_render_env():
    """Create a DotsAndBoxesEnv for live rendering."""
    return DotsAndBoxesEnv(grid_size=CONFIG['grid_size'], render_mode='human')

class CustomCallback(BaseCallback):
    """
    Custom callback to render a live gameplay episode every N episodes.
    """
    def __init__(self, model, render_every: int = 100, verbose: int = 0):
        super(CustomCallback, self).__init__(verbose)
        self.model = model
        self.render_every = render_every
        self.episode_count = 0
    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        new_episodes = [info for info in infos if 'episode' in info]
        self.episode_count += len(new_episodes)
        
        if self.episode_count > 0 and self.episode_count % self.render_every == 0:
            print(f"\n[Callback] Rendering live game after {self.episode_count} episodes...")
            render_env = make_render_env()
            play_live_game(self.model, render_env)
        
        return True
def play_live_game(model: PPO, env: DotsAndBoxesEnv) -> None:
    """
    Play a full live episode with the trained model (player 0) vs random opponent (player 1),
    rendering step-by-step with delay.
    """
    try:
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        
        print("\n=== Live Gameplay Started ===")
        env.render()
        
        while not (done or truncated):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
            
            valid_actions = env.get_valid_actions()
            if len(valid_actions) == 0:
                print("No valid actions left. Ending game.")
                break
            
            action, _ = model.predict(obs, deterministic=True)
            action = int(np.clip(action, 0, 31))
            print(f"Agent action: {action}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            print(f"Reward: {reward:.3f} (total: {total_reward:.3f}) | Scores: {info['scores']}")
            time.sleep(CONFIG['render_delay'])
            
            done = terminated or truncated
        
        print(f"=== Live Gameplay Ended | Final Reward: {total_reward:.3f} ===")
    except Exception as e:
        print(f"Pygame or other error during live game: {e}")
    finally:
        if hasattr(env, 'close'):
            env.close()

def train_agent(vec_env: SubprocVecEnv, config: dict) -> PPO:
    """
    Initialize PPO model, set up callbacks, train the agent, and save the final model.
    """
    model = PPO(
        "MlpPolicy",
        vec_env,
        seed=config['seed'],
        verbose=1,
        device='auto',
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./',
        name_prefix='model',
    )
    
    custom_callback = CustomCallback(
        model=model,
        render_every=config['render_every'],
    )
    
    callbacks = CallbackList([checkpoint_callback, custom_callback])
    
    print("Starting training...")
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=callbacks,
        progress_bar=True,
    )
    
    model.save('model.zip')
    print("Training completed. Final model saved as 'model.zip'")
    
    return model

if __name__ == "__main__":
    # Set seeds for reproducibility
    seed = CONFIG['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Starting Dots and Boxes RL training with seed {seed}...")
    print(f"Config: {CONFIG}")
    
    # Create vectorized environment
    vec_env = SubprocVecEnv([make_env for _ in range(CONFIG['n_envs'])])
    
    # Train the agent
    model = train_agent(vec_env, CONFIG)
    
    # Cleanup
    vec_env.close()
    
    print("All done!")