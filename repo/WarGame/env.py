
from collections import defaultdict
import numpy as np
from gymnasium.spaces import Box, MultiDiscrete
from pettingzoo import ParallelEnv
from game import GameState
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv


class WarGameEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv wrapper for the war game, handling reset/step/observations/rewards
    for two agents ('red_agent', 'blue_agent').
    """
    def __init__(self):
        self.agents = ['red_agent', 'blue_agent']
        self._obs_space = Box(low=0, high=2, shape=(1351,), dtype=np.float32)
        self._act_space = MultiDiscrete([13] * 5)
        self.current_state = None
        self.np_random = np.random.RandomState(42)

    def observation_space(self, agent: str) -> Box:
        assert agent in self.agents
        return self._obs_space

    def action_space(self, agent: str) -> MultiDiscrete:
        assert agent in self.agents
        return self._act_space

    def seed(self, seed=None):
        if seed is not None:
            self.np_random.seed(seed)

    def reset(self, seed=None, options=None):
        self.seed(seed)
        self.agents = ['red_agent', 'blue_agent']
        self.current_state = GameState()
        obs = {
            a: self.current_state.get_observation(a.replace('_agent', ''))
            for a in self.agents
        }
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        assert len(actions) == 2 and all(len(a) == 5 for a in actions.values())
        red_acts = list(map(int, actions['red_agent']))
        blue_acts = list(map(int, actions['blue_agent']))
        rews, done_turn = self.current_state.step(red_acts, blue_acts)
        if not done_turn:
            obs = {
                a: self.current_state.get_observation(a.replace('_agent', ''))
                for a in self.agents
            }
        else:
            obs = {
                a: np.zeros(self.observation_space(a).shape, dtype=np.float32)
                for a in self.agents
            }
        rews_dict = {'red_agent': rews['red'], 'blue_agent': rews['blue']}
        terms = {a: done_turn for a in self.agents}
        truncs = terms
        infos = {a: {} for a in self.agents}
        if done_turn:
            self.agents = []
        return obs, rews_dict, terms, truncs, infos

    def close(self):
        pass

def env_creator(config=None):
    return ParallelPettingZooEnv(WarGameEnv())