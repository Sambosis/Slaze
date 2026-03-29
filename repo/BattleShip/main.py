import numpy as np
from collections import deque
import pygame
from game import *
from agents import *
from display import PygameDisplay


def train_agents(
    game: BattleshipGame,
    agent_a: DQNAgent,
    agent_b: PolicyGradientAgent,
    display: PygameDisplay,
    total_episodes: int = 100000,
) -> None:
    """
    Main training loop: reset game, play episode (fast or live), collect/store/update, log metrics.
    Runs indefinitely or up to total_episodes, stops on pygame quit event handled by display.
    """
    episodes = 0
    wins_a = wins_b = draws = 0
    rewards_a = deque(maxlen=100)
    rewards_b = deque(maxlen=100)
    trajectory_b = []

    while display.running and episodes < total_episodes:
        game.reset()
        trajectory_b = []
        sum_r_a = 0.0
        sum_r_b = 0.0
        live = (episodes % 100 == 0)
        done = False

        while not done:
            player = game.current_player
            state = game.get_state(player)

            if player == 'A':
                action = agent_a.act(state, training=True)
                row, col = divmod(action, 10)
                reward, _, _ = game.shoot('A', row, col)
                next_state = game.get_state('A')
                done, winner = game.is_terminal()
                agent_a.replay.push(state, action, reward, next_state, done)
                sum_r_a += reward
            else:
                action = agent_b.act(state, training=True)
                row, col = divmod(action, 10)
                reward, _, _ = game.shoot('B', row, col)
                trajectory_b.append({
                    'state': state.copy(),
                    'action': action,
                    'reward': reward
                })
                sum_r_a += 0  # No reward added for A here
                sum_r_b += reward
                done, winner = game.is_terminal()

            if live:
                display.draw_boards(game, live=True)
                display.update()
                pygame.time.wait(500)

        # Post-episode updates
        rewards_a.append(sum_r_a)
        rewards_b.append(sum_r_b)

        if winner == 'A':
            wins_a += 1
        elif winner == 'B':
            wins_b += 1
        elif winner == 'draw':
            draws += 1

        # Update agents
        agent_a.update()
        agent_b.update(trajectory_b)

        episodes += 1

        # Metrics and display update every episode
        display.draw_boards(game, live=False)
        if episodes % 100 == 0:
            avg_r_a = np.mean(rewards_a)
            avg_r_b = np.mean(rewards_b)
            loss_a = agent_a.losses[-1] if hasattr(agent_a, 'losses') and agent_a.losses else 0.0
            loss_b = agent_b.episode_losses[-1] if hasattr(agent_b, 'episode_losses') and agent_b.episode_losses else 0.0
            display.draw_metrics(
                episodes, wins_a, wins_b, draws, avg_r_a, avg_r_b, loss_a, loss_b
            )
        display.update()


if __name__ == "__main__":
    pygame.init()
    game = BattleshipGame()
    agent_a = DQNAgent(state_dim=200, action_dim=100)
    agent_b = PolicyGradientAgent(state_dim=200, action_dim=100)
    display = PygameDisplay(screen_width=1200, screen_height=800)
    train_agents(game, agent_a, agent_b, display)
    pygame.quit()