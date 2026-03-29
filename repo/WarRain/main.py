import gymnasium as gym
import torch
import numpy as np
import os
import random
from war_env import WarEnv
from agent import RainbowDQN
def render_episode(agent: RainbowDQN, max_steps: int = 5000) -> float:
    env = WarEnv(render_mode='human')
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    episode_step = 0
    while not done and episode_step < max_steps:
        action = agent.act(obs, training=False)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        env.render()
        done = terminated or truncated
        episode_step += 1
    env.render()
    env.close()
    return total_reward


def train(
    env: gym.Env,
    agent: RainbowDQN,
    max_steps: int = 10_000_000,
    eval_freq: int = 500
) -> None:
    """
    Main training loop for Rainbow DQN.
    Collects experiences, updates agent periodically, evaluates every eval_freq episodes,
    saves checkpoints every 100k steps, logs to console and training_log.txt.
    """
    total_reward_log = []
    episode = 0
    global_step = 0
    log_file = 'training_log.txt'
    
    # Header for log file
    with open(log_file, 'w') as f:
        f.write("Episode\\tGlobalStep\\tEpisodeReward\\tAvg100Reward\\tEpsilon\\n")
    
    replay_capacity = agent.replay_buffer.capacity
    warmup_steps = replay_capacity // 10  # Approx 100k steps of random actions
    
    print("Starting training...")
    print(f"Warmup steps (random actions): {warmup_steps}")
    print(f"Replay capacity: {replay_capacity}")
    
    while global_step < max_steps:
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done and global_step < max_steps:
            # Select action
            if global_step < warmup_steps:
                action = int(env.action_space.sample())
            else:
                action = agent.act(obs, training=True)

            # Step environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_reward += reward
            global_step += 1

            # Agent updates
            if global_step % 4 == 0 and len(agent.replay_buffer) > 32 * 10:
                agent.update()
            if global_step % 10000 == 0:
                agent.target_update()
                print(f"Target network updated at step {global_step}")

            # Save checkpoint
            if global_step % 100000 == 0:
                torch.save(
                    agent.q_net.state_dict(),
                    f'models/checkpoint_{global_step}.pth'
                )
                print(f"Checkpoint saved at step {global_step}")

        # Episode finished
        episode += 1
        total_reward_log.append(episode_reward)
        avg_100 = np.mean(total_reward_log[-100:]) if len(total_reward_log) >= 100 else np.mean(total_reward_log)
        print_str = (
            f"Episode {episode} | "
            f"Step {global_step} | "
            f"EpRew {episode_reward:.2f} | "
            f"Avg100 {avg_100:.2f} | "
            f"Epsilon {agent.epsilon:.3f}"
        )
        print(print_str)

        # Log to file
        with open(log_file, 'a') as f:
            f.write(
                f"{episode}\\t{global_step}\\t{episode_reward:.2f}\\t"
                f"{avg_100:.2f}\\t{agent.epsilon:.3f}\\n"
            )

        # Evaluation every eval_freq episodes
        if episode % eval_freq == 0:
            eval_reward = render_episode(agent)
            eval_print = f"Eval Episode {episode}: {eval_reward:.2f}"
            print(eval_print)
            with open(log_file, 'a') as f:
                f.write(f"EVAL {episode}\\t{eval_reward:.2f}\\n")
    
    print("Training completed!")


if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env = WarEnv()
    agent = RainbowDQN(env.observation_space.shape, env.action_space.n)
    train(env, agent)
    env.close()