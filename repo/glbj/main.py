"""
Main entry point for BlackjackBetOptimizer application.
Orchestrates initialization of game components, RL training loop,
visualization updates, event handling, and demo mode.
"""

import pygame
import pickle
import os
import collections
from typing import Dict, Any

from config import CONFIG
from game import BlackjackGame, BasicStrategy, CardCounter
from rl import QLearner
from viz import Visualizer
from utils import compute_stats


def main() -> None:
    """
    Main training and demo loop.
    Initializes components, runs epsilon-greedy training for configurable hands,
    interleaves with Pygame visualization updates and animations.
    Handles user controls: Space (pause/step), s (save), l (load), q (quit).
    Transitions to greedy demo mode post-training or on bankruptcy.
    """
    print(f'Starting training with bankroll={bankroll}, target hands={CONFIG["training_hands"]}', flush=True)
    
    while True:
        action = viz.run_event_loop()
        if action == 'quit':
            break
        if action == 'save':
            qlearner.save(qtable_file)
            print('Q-table saved to', qtable_file, flush=True)
            continue
        if action == 'load':
            try:
                qlearner.load(qtable_file)
                print('Q-table reloaded from', qtable_file, flush=True)
            except (FileNotFoundError, pickle.UnpicklingError, KeyError):
                print('Load failed: No valid Q-table found', flush=True)
            continue
        if viz.paused:
            viz.clock.tick(10)
            continue

        if hands_played >= CONFIG['training_hands'] or bankroll < CONFIG['table_min_bet']:
            demo_mode = True
            training = False
            print(f'Training complete / low bankroll. Switching to demo mode. Hands: {hands_played}, Bankroll: {bankroll:.0f}', flush=True)

        decks_rem = len(game.shoe) / (52 * CONFIG['num_decks'])
        true_count = counter.get_true_count(decks_rem)
        bankroll_frac = bankroll / CONFIG['initial_bankroll']
        recent_avg_units = (
            sum(recent_bets) / len(recent_bets) / CONFIG['unit_size']
            if recent_bets else 1.0
        )
        state = qlearner.get_state(bankroll_frac, true_count, recent_avg_units)

        multiplier = qlearner.get_action(state, training=training)
        bet = multiplier * CONFIG['unit_size']

        bet = max(CONFIG['table_min_bet'], min(CONFIG['table_max_bet'], bet))
        bet = min(bet, bankroll * 0.1)
        if bet < CONFIG['table_min_bet']:
            print('Bankrupt: Insufficient bankroll to place min bet', flush=True)
            break

        delta, player_hands, dealer_cards = game.play_hand(bet, strategy, counter)

        bankroll += delta
        bankroll_history.append(bankroll)
        bets.append(bet)
        recent_bets.append(bet)
        if delta > 0:
            wins += 1
        elif delta < 0:
            losses += 1
        hands_played += 1

        decks_rem_next = len(game.shoe) / (52 * CONFIG['num_decks'])
        next_true_count = counter.get_true_count(decks_rem_next)
        next_bank_frac = bankroll / CONFIG['initial_bankroll']
        next_recent_avg_units = (
            sum(recent_bets) / len(recent_bets) / CONFIG['unit_size']
        )
        next_state = qlearner.get_state(next_bank_frac, next_true_count, next_recent_avg_units)

        qlearner.update(state, multiplier, delta, next_state)

        if training and hands_played % 1000 == 0:
            qlearner.decay_params()

        stats_dict: Dict[str, Any] = {
            'epsilon': qlearner.epsilon,
            'alpha': qlearner.alpha,
        }
        full_stats = compute_stats(bankroll_history, hands_played, wins, losses, bets)
        stats_dict.update(full_stats)

        decks_rem = len(game.shoe) / (52 * CONFIG['num_decks'])
        stats_dict['decks_remaining'] = round(decks_rem, 1)
        stats_dict['running_count'] = counter.running_count
        stats_dict['true_count'] = counter.get_true_count(decks_rem)

        if hands_played % 10 == 0:
            viz.update(bankroll, bet, counter, stats_dict, bankroll_history, game=game)
        if demo_mode or (hands_played % 50 == 0):
            viz.animate_hand(game, bet, strategy, counter)

        if hands_played % 10 == 0 or demo_mode:
            print(
                f'Hands: {hands_played}, Bankroll: {bankroll:.0f}, '
                f'Bet: {bet:.1f}, Delta: {delta:.1f}, '
                f'TC: {true_count:.1f}, '
                f'Avg Return: {full_stats["avg_return"]:.4f}',
                flush=True
            )

        if demo_mode:
            viz.clock.tick(2)


if __name__ == "__main__":
    main()