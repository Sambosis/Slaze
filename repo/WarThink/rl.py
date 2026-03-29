import numpy as np
import torch
from stable_baselines3 import PPO
from copy import deepcopy
from typing import List, Optional

from env import WarGameEnv


class PPOSelfPlayAgent:
    """
    Handles PPO model with asymmetric self-play, population pool, and noise injection.
    
    pool[0]: main clean policy for P1
    pool[1:]: noisy variants for P2 (fixed idx per episode)
    Evolves every 100 episodes: top-2 selected, others mutated, best -> pool[0]
    """

    def __init__(self, env: WarGameEnv, pool_size: int = 5, noise_std: float = 0.1):
        self.env = env
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Base PPO model with standard params
        base_model = PPO(
            'MlpPolicy',
            env,
            verbose=0,
            device=self.device,
            n_steps=512,
            batch_size=256,
            learning_rate=3e-4,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2
        )

        # Policy pool
        ppo_params = {
            'policy': 'MlpPolicy',
            'env': env,
            'verbose': 0,
            'device': self.device,
            'n_steps': 512,
            'batch_size': 256,
            'learning_rate': 3e-4,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2
        }
        base_model = PPO(**ppo_params)
        self.policy_pool: List[PPO] = [base_model]
        for _ in range(1, pool_size):
            new_model = PPO(**ppo_params)
            new_model.set_parameters(base_model.get_parameters())
            self.policy_pool.append(new_model)
        self.model = self.policy_pool[0]  # Main model reference
        del base_model

        # Tracking stats
        self.episode_rewards: List[float] = []
        self.episode_count: int = 0
        self.pool_performances: List[float] = [0.0] * pool_size
        self.pool_usage_count: List[int] = [0] * pool_size
        self.noise_std = noise_std
        self.current_p2_idx: Optional[int] = None

    def start_episode(self):
        """Select fixed P2 policy index for entire episode."""
        self.current_p2_idx = np.random.randint(1, len(self.policy_pool))

    def act(self, obs: np.ndarray, player: int) -> np.ndarray:
        """
        Args:
            obs: np.ndarray (10,10,5)
            player: 1 or 2
            
        Returns:
            actions: np.ndarray (5,) Discrete(9)
        """
        if player == 1:
            model = self.policy_pool[0]
        else:
            if self.current_p2_idx is None:
                raise ValueError("start_episode() must be called before act() for P2")
            model = self.policy_pool[self.current_p2_idx]
        
        # Stochastic prediction
        actions, _ = model.predict(obs, deterministic=False)
        actions = actions.astype(np.int32)
        
        # Action noise for P2 only
        if player == 2 and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=(5,))
            actions = np.clip(actions + noise, 0, 8).astype(np.int32)
        
        return actions

    def update_pool(self, rewards: List[float]):
        """
        Credit episode rewards to used policies, increment usage.
        
        Args:
            rewards: [p1_reward, p2_reward]
        """
        if len(rewards) != 2 or self.current_p2_idx is None:
            print("Warning: update_pool called without start_episode or invalid rewards.")
            return
        
        p1_reward, p2_reward = rewards
        p1_idx = 0
        p2_idx = self.current_p2_idx
        
        self.episode_rewards.extend(rewards)
        
        self.pool_performances[p1_idx] += p1_reward
        self.pool_performances[p2_idx] += p2_reward
        
        self.pool_usage_count[p1_idx] += 1
        self.pool_usage_count[p2_idx] += 1
        
        self.episode_count += 1
        self.current_p2_idx = None

    def evolve_pool(self):
        """Evolve if episode_count % 100 == 0: normalize perf/usage, top-2 elite, mutate others, promote best to [0]."""
        if self.episode_count % 100 != 0:
            return
        
        recent_rewards = self.episode_rewards[-200:] if len(self.episode_rewards) >= 200 else self.episode_rewards
        recent_avg = np.mean(recent_rewards)
        print(f"Pool evolution triggered at episode {self.episode_count}. Recent avg reward: {recent_avg:.3f}")
        
        # Normalize performances by usage
        normalized_performances = np.array([
            perf / max(usage, 1.0)
            for perf, usage in zip(self.pool_performances, self.pool_usage_count)
        ])
        
        top_indices = np.argsort(normalized_performances)[-2:]
        top_scores = normalized_performances[top_indices]
        scores_str = ", ".join([f"{s:.3f}" for s in top_scores])
        print(f"Top performers: {top_indices} (scores: {scores_str})")
        
        # Mutate non-elites
        for i in range(len(self.policy_pool)):
            if i not in top_indices:
                self.mutate_policy(self.policy_pool[i])
                print(f"Mutated policy {i}")
        
        # Promote best to main
        best_idx = int(top_indices[-1])
        if best_idx != 0:
            best_params = self.policy_pool[best_idx].get_parameters()
            self.policy_pool[0].set_parameters(best_params)
        self.model = self.policy_pool[0]
        print(f"Best policy {best_idx} promoted to main (slot 0)")
        
        # Reset stats
        self.pool_performances = [0.0] * len(self.policy_pool)
        self.pool_usage_count = [0] * len(self.policy_pool)

    def mutate_policy(self, model: PPO):
        """Apply Gaussian noise to policy parameters."""
        with torch.no_grad():
            for param in model.policy.parameters():
                if param.requires_grad:
                    noise = torch.normal(0, self.noise_std, param.shape, device=self.device)
                    param.add_(noise)

    def learn(self, total_timesteps: int):
        """PPO learning: full on main, reduced on pool variants."""
        if total_timesteps <= 0:
            return
        # Main policy
        self.model.learn(total_timesteps=total_timesteps)
        # Variants
        n_variants = len(self.policy_pool) - 1
        if n_variants > 0:
            variant_timesteps = max(1, total_timesteps // n_variants)
            for model in self.policy_pool[1:]:
                model.learn(total_timesteps=variant_timesteps)

    def save(self, path: str):
        """Save main model (pool[0])."""
        self.model.save(path)

    def load(self, path: str):
        """Load main model and deepcopy to pool."""
        self.model = PPO.load(path, env=self.env)
        self.policy_pool[0] = self.model
        # Load params into other models
        base_params = self.model.get_parameters()
        for i in range(1, len(self.policy_pool)):
            self.policy_pool[i].set_parameters(base_params)