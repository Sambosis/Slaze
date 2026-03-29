import os
import numpy as np
import torch as th
import torch.nn as nn
from typing import Callable, Dict, List, Optional
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from env import WarGameEnv


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_chans = observation_space.shape[-1]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_chans, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )
        self.n_flatten = 64 * 12 * 12
        self.linear = nn.Sequential(
            nn.Linear(self.n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = observations.float()
        x = observations.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        x = self.cnn(x)
        x = self.linear(x)
        return x


def load_policy(path: str, env: gym.Env) -> PPO:
    return PPO.load(path, env=env, device='auto')


class SelfPlayCallback(BaseCallback):
    def __init__(self, trainer):
        super().__init__(verbose=0)
        self.trainer = trainer

    def _on_step(self) -> bool:
        if self.num_timesteps % 50000 == 0 and self.num_timesteps > 0:
            winrates = self.trainer._eval_vs_opponents()
            self.trainer._eval_and_update_league(winrates)
        return True


class LeagueTrainer:
    def __init__(self,
                 env_fn: Callable[[], WarGameEnv],
                 league_size: int = 5,
                 learning_rate: float = 3e-4,
                 clip_range: float = 0.2,
                 n_steps: int = 512,
                 n_envs: int = 4,
                 ent_coef: float = 0.01):
        self.env_fn = env_fn
        self.league_size = league_size
        self.learning_rate = learning_rate
        self.clip_range = clip_range
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.ent_coef = ent_coef
        self.league: List[str] = []
        self.league_winrates: Dict[str, float] = {}
        self.league_elos: Dict[str, float] = {}
        self.opponent_models: List[PPO] = []
        self.model: Optional[PPO] = None
        self.policy_kwargs = {
            "features_extractor_class": CustomCNN,
            "net_arch": {"pi": [256], "vf": [256]}
        }
        self.dummy_env = env_fn()
        self._init_league()

    def _init_league(self):
        os.makedirs('models', exist_ok=True)
        for i in range(self.league_size):
            model = PPO(
                'CnnPolicy',
                self.dummy_env,
                policy_kwargs=self.policy_kwargs,
                verbose=0,
                device='auto',
                learning_rate=self.learning_rate,
                clip_range=self.clip_range,
                n_steps=self.n_steps,
                ent_coef=self.ent_coef
            )
            if i > 0:
                for param in model.policy.parameters():
                    param.data += th.randn_like(param.data) * 0.05
            path = f'models/initial_{i}.zip'
            model.save(path)
            self.league.append(path)
            self.league_winrates[path] = 0.5
            self.league_elos[path] = 1200.0
        self._update_opponent_models()

    def _update_opponent_models(self):
        self.opponent_models = [load_policy(p, self.dummy_env) for p in self.league]

    def _make_vec_env(self) -> DummyVecEnv:
        vec_env = DummyVecEnv([self.env_fn for _ in range(self.n_envs)])
        for i, env in enumerate(vec_env.envs):
            opp_model = self.opponent_models[i % len(self.opponent_models)]
            env.set_opponent_policy(opp_model)
        return vec_env
    def _eval_vs_opponents(self) -> Dict[str, float]:
        winrates = {}
        for i, opp_path in enumerate(self.league):
            opp_model = self.opponent_models[i]
            eval_env = self.env_fn()
            eval_env.set_opponent_policy(opp_model)
            wins = 0
            n_ep = 20
            for _ in range(n_ep):
                obs, _ = eval_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, r, term, trunc, info = eval_env.step(action)
                    done = term or trunc
                if info.get('winner') == 'p1_win':
                    wins += 1
            winrate = wins / n_ep
            winrates[opp_path] = winrate
            eval_env.close()
        return winrates

    def _eval_and_update_league(self, winrates: Dict[str, float]):
        avg_winrate = np.mean(list(winrates.values()))
        if not hasattr(self, 'league_elos') or not self.league_elos:
            self.league_elos = {p:1200.0 for p in self.league}
        my_elo = 1200.0
        for opp_path, winrate in winrates.items():
            opp_elo = self.league_elos[opp_path]
            exp_score = 1 / (1 + 10**((opp_elo - my_elo)/400))
            my_elo += 32 * (winrate - exp_score)
        avg_league_elo = np.mean(list(self.league_elos.values()))
        elo_prob = 1 / (1 + 10**((avg_league_elo - my_elo)/400))
        print(f"Avg winrate {avg_winrate:.2f}, My Elo {my_elo:.0f}, Elo prob {elo_prob:.2f}")
        if avg_winrate > 0.6 and elo_prob > 0.6:
            path = f'models/learner_{int(self.model.num_timesteps)}.zip'
            self.model.save(path)
            self.league.append(path)
            self.league_winrates[path] = avg_winrate
            self.league_elos[path] = my_elo
            if len(self.league) > self.league_size:
                worst = min(self.league_winrates, key=self.league_winrates.get)
                self.league.remove(worst)
                del self.league_winrates[worst]
                del self.league_elos[worst]
            self._update_opponent_models()
            vec_env = self.model.get_env()
            if vec_env is not None:
                self._assign_opponents(vec_env)
            print(f"Added to league, winrate {avg_winrate:.2f}")
        else:
            print("Not added: winrate or elo_prob <0.6")

    def _assign_opponents(self, vec_env: DummyVecEnv):
        for i, env in enumerate(vec_env.envs):
            opp_model = self.opponent_models[i % len(self.opponent_models)]
            env.set_opponent_policy(opp_model)

    def train(self, total_timesteps: int):
        vec_env = self._make_vec_env()
        if self.model is None:
            self.model = PPO(
                'CnnPolicy',
                vec_env,
                policy_kwargs=self.policy_kwargs,
                verbose=1,
                device='auto',
                learning_rate=self.learning_rate,
                clip_range=self.clip_range,
                n_steps=self.n_steps,
                ent_coef=self.ent_coef
            )
        else:
            self.model.set_env(vec_env)
        callback = SelfPlayCallback(self)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False
        )

    def eval_game(self, policy_path: str = None) -> Dict[str, float]:
        if not self.opponent_models:
            return {'avg_reward': 0.0, 'winrate': 0.0}
        opp_model = np.random.choice(self.opponent_models)
        learner = self.model if policy_path is None else load_policy(policy_path, self.dummy_env)
        eval_env = self.env_fn()
        eval_env.set_opponent_policy(opp_model)
        total_reward = 0.0
        wins = 0
        n_ep = 20
        for _ in range(n_ep):
            obs, _ = eval_env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action, _ = learner.predict(obs, deterministic=True)
                obs, r, term, trunc, info = eval_env.step(action)
                ep_reward += r
                done = term or trunc
            total_reward += ep_reward
            if info.get('winner') == 'p1_win':
                wins += 1
        eval_env.close()
        avg_reward = total_reward / n_ep
        winrate = wins / n_ep
        return {'avg_reward': avg_reward, 'winrate': winrate}

    def get_current_policy(self):
        return self.model
