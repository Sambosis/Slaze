import os
import pygame
from pygame.locals import *
import os
from game import GameState
from env import WarGameEnv, env_creator
from renderer import GameRenderer
from trainer import train_selfplay_model, run_rendered_episode
os.environ['RAY_DISABLE_UV_RUN'] = '1'

def main():
    pygame.init()
    screen = pygame.display.set_mode((1024, 768))
    pygame.display.set_caption('WarGame')
    clock = pygame.time.Clock()
    renderer = GameRenderer(screen, clock)
    dummy_state = GameState()
    trainer = None
    episodes = 0
    paused = False
    running = True
    last_result = None
    os.makedirs('model_checkpoint', exist_ok=True)
    font = pygame.font.SysFont('arial', 24, bold=True)
    small_font = pygame.font.SysFont('arial', 18)
    print("Initializing trainer...")
    
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_SPACE:
                    paused = not paused
                elif event.key == K_s:
                    if trainer is not None:
                        trainer.save('model_checkpoint/')
                        print("Manual checkpoint saved.")
        if not paused:
            if trainer is None:
                trainer = train_selfplay_model(env_creator)
            else:
                result = trainer.train()
                last_result = result
                episodes_this_iter = result.get('episodes_this_iter', 0)
                episodes += episodes_this_iter
                if episodes % 50 == 0 and episodes > 0:
                    print(f"Rendering episode at {episodes} episodes...")
                    run_rendered_episode(trainer, renderer)
                if episodes % 500 == 0 and episodes > 0:
                    trainer.save('model_checkpoint/')
                    print(f"Auto-saved checkpoint at {episodes} episodes.")
        renderer.draw(dummy_state, 'spectate')
        ui_x = 620
        ui_y = 10
        lines = [
            f"Episodes: {episodes:,}",
            f"Paused: {'Yes' if paused else 'No'}",
        ]
        if last_result:
            avg_reward = last_result.get('episode_reward_mean', 0.0)
            loss = last_result.get('loss', 0.0)
            lines += [
                f"Avg Reward: {avg_reward:.2f}",
                f"Loss: {loss:.4f}",
            ]
        else:
            lines += ["Avg Reward: --", "Loss: --"]
        for line in lines:
            text_surf = font.render(line, True, (255, 255, 255))
            screen.blit(text_surf, (ui_x, ui_y))
            ui_y += 30
        instr_text = small_font.render(
            "SPACE=pause | S=save | ESC=quit (auto-save)", True, (200, 200, 200)
        )
        screen.blit(instr_text, (ui_x, 720 - 20))
        pygame.display.flip()
        clock.tick(60)
    if trainer is not None:
        trainer.save('model_checkpoint/')
        print("Final auto-save on quit.")
    pygame.quit()

if __name__ == '__main__':
    main()