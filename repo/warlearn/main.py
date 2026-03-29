import sys
import numpy as np
import pygame
import os
from env import WarGameEnv, Board
from renderer import Renderer
from trainer import LeagueTrainer, load_policy

SEED = 42
np.random.seed(SEED)

import sys
import numpy as np
import pygame
import os
import argparse
from env import WarGameEnv, Board
from renderer import Renderer
from trainer import LeagueTrainer, load_policy

SEED = 42
np.random.seed(SEED)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--n_steps', type=int, default=512)
    parser.add_argument('--total_timesteps', type=int, default=50000)
    parser.add_argument('--render_freq', type=int, default=1)
    parser.add_argument('--no-render', action='store_true', default=False)
    parser.add_argument('--league_size', type=int, default=5)
    parser.add_argument('--n_envs', type=int, default=4)
    parser.add_argument('--ent_coef', type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs('models', exist_ok=True)
    env_fn = lambda: WarGameEnv(np.random.randint(0, 2**31 - 1))
    trainer = LeagueTrainer(
        env_fn,
        league_size=args.league_size,
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        n_steps=args.n_steps,
        n_envs=args.n_envs,
        ent_coef=args.ent_coef
    )
    renderer = Renderer()
    clock = pygame.time.Clock()
    training_loops = 0
    running = True

    while running:
        print(f'Training {args.total_timesteps} steps...')
        trainer.train(args.total_timesteps)
        training_loops += 1
        if training_loops % 25 == 0 and training_loops > 0:
            trainer.model.save('models/latest.zip')
            print(f'Auto-saved model after {training_loops} training loops.')
        metrics = trainer.eval_game()
        print(f'Metrics: {metrics}')

        if not args.no_render and training_loops % args.render_freq == 0:
            print('Rendering eval game vs random league opponent...')
            dummy_env = env_fn()
            learner_policy = trainer.get_current_policy()
            if trainer.league:
                opp_idx = np.random.randint(0, len(trainer.league))
                opp_path = trainer.league[opp_idx]
                opp_policy = load_policy(opp_path, dummy_env)
            else:
                opp_policy = None
            board = Board(seed=np.random.randint(0, 2**31 - 1))
            done_flag, winner = board.is_done()
            while running and not done_flag:
                stats = {
                    'training_loops': training_loops,
                    'avg_reward': metrics['avg_reward'],
                    'winrate': metrics['winrate']
                }
                if not renderer.render(board, clock, 0.02, stats):
                    running = False
                    break
                obs_p1 = board.get_obs(True)
                a1, _ = learner_policy.predict(obs_p1, deterministic=True)
                a1 = a1.tolist()
                obs_p2 = board.get_obs(False)
                if opp_policy is not None:
                    a2, _ = opp_policy.predict(obs_p2, deterministic=True)
                    a2 = a2.tolist()
                else:
                    a2 = [4] * len(board.p2_units)
                board.resolve_turn(a1, a2)
                done_flag, winner = board.is_done()
                if done_flag:
                    stats['winner'] = winner
                    renderer.render(board, clock, 0.02, stats)
                    pygame.time.wait(300)
                    break
            dummy_env.close()
        if renderer.save_requested:
            trainer.model.save('models/latest.zip')
            renderer.save_requested = False
            print('Saved latest model.')

if __name__ == '__main__':
    main()
    pygame.quit()
    sys.exit()