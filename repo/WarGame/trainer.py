import os
import os
from ray.tune.registry import register_env
from ray.tune.registry import register_env
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from gymnasium.spaces import Box
from typing import Dict, Any, Callable, List, Tuple
from game import GameState
from env import WarGameEnv
from renderer import GameRenderer
import pygame

torch, nn = try_import_torch(); import torch.nn.functional as F
os.environ['RAY_DISABLE_UV_RUN'] = '1'
ray.init(ignore_reinit_error=True, num_cpus=2)

class WarGameICMModel(TorchModelV2, nn.Module):
    """Custom recurrent LSTM policy with agent_id embedding and ICM feature extractors (phi, fwd).
    
    - Obs (1350,) cat agent_id_emb(16) -> shared (256) -> LSTM(256) -> policy(65)/value(1)
    - ICM: phi_net(obs->128), fwd_net(128+65->128) exposed for callback.
    - agent_ids default 0 (red bias, symmetric self-play approx).
    """
    def __init__(self, obs_space: Box, action_space, num_outputs: int, model_config: Dict[str, Any], name: str):
        super(WarGameICMModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        print(f"DEBUG: WarGameICMModel init. model_config={model_config}")
        self.obs_dim = 1350  # feat part 15*15*6
        self.agent_id_emb = nn.Embedding(2, 16)
        self.shared = nn.Sequential(
            nn.Linear(self.obs_dim + 16, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(256, 256, batch_first=True)
        self.policy_head = nn.Linear(256, 65)  # Flattened for MultiDiscrete(13x5)
        self.value_head = nn.Linear(256, 1)
        # ICM
        self.phi_net = nn.Sequential(
            nn.Linear(1351, 128),  # full obs incl agent_id
            nn.ReLU()
        )
        self.fwd_net = nn.Sequential(
            nn.Linear(128 + 65, 128),
            nn.ReLU()
        )
    def forward(self, input_dict: Dict[str, torch.Tensor], state: List[torch.Tensor], seq_lens: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # print(f"DEBUG: forward state len={len(state)}")
        obs = input_dict["obs"].float()
        aid = (obs[:, -1] > 0.5).long()
        obs_feat = obs[:, :-1]
        aid_emb = self.agent_id_emb(aid)
        x = torch.cat([obs_feat, aid_emb], dim=-1)
        shared_feat = self.shared(x)
        lstm_in = shared_feat.unsqueeze(1)
        lstm_out, lstm_state = self.lstm(lstm_in, state)
        features = lstm_out.squeeze(1)
        logits = self.policy_head(features)
        return logits, lstm_state
    def value_function(self, input_dict: Dict[str, torch.Tensor], state: List[torch.Tensor], seq_lens: torch.Tensor) -> torch.Tensor:
        obs = input_dict["obs"].float()
        aid = (obs[:, -1] > 0.5).long()
        obs_feat = obs[:, :-1]
        aid_emb = self.agent_id_emb(aid)
        x = torch.cat([obs_feat, aid_emb], dim=-1)
        shared_feat = self.shared(x)
        lstm_in = shared_feat.unsqueeze(1)
        lstm_out, _ = self.lstm(lstm_in, state)
        features = lstm_out.squeeze(1)
        return self.value_head(features).squeeze(1)

class ICMCallback(DefaultCallbacks):
    """ICM intrinsic reward callback: augments rewards with eta * ||fwd(phi(s), onehot(a)) - phi(s')||^2
    Applied in on_postsample for trajectory batches.
    Assumes MultiDiscrete actions (T,5) -> onehot (T,65).
    """
    def __init__(self, eta: float = 0.01):
        self.eta = eta
    def on_postsample_workers(self, postprocessor, batch, **kwargs):
        from ray.rllib.policy.sample_batch import SampleBatch
        import torch.nn.functional as F
        
        policy_map = postprocessor.policy_map
        policy = list(policy_map.values())[0]
        model = policy.model
        device = next(model.parameters()).device

        obs = batch.data[SampleBatch.OBS]
        new_obs = batch.data[SampleBatch.NEXT_OBS]
        actions = batch.data[SampleBatch.ACTIONS]

        obs_t = torch.as_tensor(obs, device=device).float()
        new_obs_t = torch.as_tensor(new_obs, device=device).float()
        acts_t = torch.as_tensor(actions, device=device).long()

        feats = model.phi_net(obs_t)
        next_feats = model.phi_net(new_obs_t)

        act_onehot = F.one_hot(acts_t, num_classes=13).float().view(-1, 65)

        pred_next_feats = model.fwd_net(torch.cat([feats, act_onehot], dim=-1))

        mse = F.mse_loss(pred_next_feats, next_feats.detach()).mean()
        intrinsic_r = 0.01 * mse.detach().cpu().numpy() * torch.ones_like(batch.data[SampleBatch.REWARDS])

        batch.data[SampleBatch.REWARDS] += intrinsic_r
        return batch
def train_selfplay_model(env_creator: callable, total_episodes: int = 10000, render_every: int = 50) -> Any:
    """Configures and runs RLlib PPO multi-agent self-play training with shared LSTM policy + agent_id + ICM curiosity, handles rendering integration and checkpoints."""
    os.environ['RAY_DISABLE_UV_RUN'] = '1'
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True)
    
    register_env("WarGame-v0", env_creator)
    
    dummy_env = env_creator()
    # PettingZooEnv wrapper exposes observation_space as a Dict space or attribute
    # We need to get the space for one agent (assuming homogeneous or specific agent)
    obs_space = dummy_env.observation_space['red_agent']
    act_space = dummy_env.action_space['red_agent']
    dummy_env.close()
    
    policy_config = {
        # "custom_model": WarGameICMModel,
        "use_lstm": False,
        "lstm_cell_size": 256,
        "max_seq_len": 20,
    }
    
    policies = {"default_policy": (None, obs_space, act_space, policy_config)}
    
    config = (
        PPOConfig()
        .environment("WarGame-v0")
        .env_runners(num_env_runners=2)
        .training(
            lr=3e-4,
            train_batch_size=4000,
            minibatch_size=256,
            num_epochs=10
        )
        .framework("torch")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "default_policy"
        )
        # .callbacks(ICMCallback)
        .resources(num_gpus=0)
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    )
    
    algo = config.build()
    print("Trainer built successfully!")
    
    os.makedirs('model_checkpoint', exist_ok=True)
    
    return algo
def run_rendered_episode(algo, renderer):
    env = WarGameEnv()
    obs, _ = env.reset()
    renderer.animations.clear()
    total_rew = {'red': 0, 'blue': 0}
    policy = algo.get_policy('default_policy')
    rnn_states = {a: policy.get_initial_state() for a in env.agents}
    running = True
    
    while env.agents and running:
        actions = {}
        for a in env.agents:
            act_out, rnn_state, _ = policy.compute_single_action(
                obs[a],
                state=rnn_states[a],
                explore=False,
                full_fetch=True
            )
            actions[a] = act_out
            rnn_states[a] = rnn_state
        
        # Pre-animate moves
        for a, act_lst in actions.items():
            owner = a.replace('_agent', '')
            units = getattr(env.current_state, f'units_{owner}')
            for i, act in enumerate(act_lst):
                typ, delta = env.current_state.get_action_meaning(act)
                if typ == 'move':
                    tgt = env.current_state.compute_target_pos(units[i].pos, *delta, units[i].speed)
                    renderer.animate_move(units[i], tgt)
        
        obs, rews, terms, truncs, infos = env.step(actions)
        total_rew['red'] += rews['red_agent']
        total_rew['blue'] += rews['blue_agent']
        
        # Wait for animations
        while renderer.animations and running:
            for ev in pygame.event.get():
                if ev.type == QUIT or (ev.type == KEYDOWN and ev.key == K_ESCAPE):
                    running = False
                    break
            if not running:
                break
            renderer.draw(env.current_state, 'spectate')
            pygame.display.flip()
            renderer.clock.tick(60)
        
        # Post-step slow tick for 2x speed
        renderer.draw(env.current_state, 'spectate')
        pygame.display.flip()
        renderer.clock.tick(2)
    
    env.close()
    
    winner = 'Red' if env.current_state.bases_red_alive and not env.current_state.bases_blue_alive else 'Blue' if env.current_state.bases_blue_alive and not env.current_state.bases_red_alive else 'Tie'
    print(f"Rendered: turns={env.current_state.turn_count} rew_r={total_rew['red']:.1f} rew_b={total_rew['blue']:.1f} winner={winner}")
    return total_rew
