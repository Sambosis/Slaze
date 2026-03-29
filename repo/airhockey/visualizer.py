#!/usr/bin/env python3
"""
Air Hockey Visualization Module

This module provides helper functions for initializing Pygame and drawing
the air hockey game elements during visualization episodes.
"""

import pygame
import sys
from typing import Tuple, Optional


def init_pygame(width: int, height: int, caption: str = "Air Hockey RL Training") -> tuple[pygame.Surface, pygame.time.Clock]:
    """
    Initialize Pygame and create the display window.
    
    Args:
        width: Width of the game window in pixels
        height: Height of the game window in pixels
        caption: Title for the Pygame window
        
    Returns:
        Tuple of (Pygame screen surface, Pygame clock)
        
    Raises:
        RuntimeError: If Pygame initialization fails
    """
    try:
        # Initialize all imported pygame modules
        pygame.init()

        # Create the display surface
        screen = pygame.display.set_mode((width, height))

        # Set window caption
        pygame.display.set_caption(caption)

        # Initialize font for text rendering
        pygame.font.init()

        # Create clock for frame rate control
        clock = pygame.time.Clock()

        # Disable sound (as per requirements)
        # Note: No pygame.mixer.init() call

        return screen, clock
    
    except Exception as e:
        print(f"Error initializing Pygame: {e}")
        pygame.quit()
        sys.exit(1)
def draw_game(screen: pygame.Surface, env, episode: int, step: int,
              score1: int, score2: int, agent1_epsilon: float,
              agent2_epsilon: float, clock: pygame.time.Clock = None) -> bool:
    """
    Draw the current game state on the provided Pygame surface.
    
    This function renders the air hockey table, paddles, puck, scores,
    and episode information. It also handles Pygame events and maintains
    the specified frame rate.
    
    Args:
        screen: Pygame surface to draw on
        env: The AirHockeyEnv instance containing game state
        episode: Current episode number
        step: Current step within the episode
        score1: Score for player 1
        score2: Score for player 2
        agent1_epsilon: Epsilon value for agent 1
        agent2_epsilon: Epsilon value for agent 2
        clock: Pygame clock for frame rate control (optional)
        
    Returns:
        bool: True if the visualization should continue, False if user wants to quit
    """
    # Clear the screen (background will be drawn by env.render)
    screen.fill((0, 0, 0))
    
    # Draw the game state using the environment's render method
    env.render(screen)
    
    # Draw UI elements (scores, episode info)
    _draw_ui(screen, score1, score2, episode, step, agent1_epsilon, agent2_epsilon, 
             env.width, env.height)
    
    # Update the display
    pygame.display.flip()
    
    # Maintain frame rate if clock is provided
    if clock is not None:
        clock.tick(60)
    
    return True
def _draw_ui(screen: pygame.Surface, score1: int, score2: int, 
             episode: int, step: int, agent1_epsilon: float, agent2_epsilon: float,
             width: int, height: int) -> None:
    """
    Draw a modern, premium HUD overlay.
    
    Args:
        screen: Pygame surface to draw on
        score1: Score for player 1 (left side)
        score2: Score for player 2 (right side)
        episode: Current episode number
        step: Current step within the episode
        agent1_epsilon: Epsilon value for agent 1
        agent2_epsilon: Epsilon value for agent 2
        width: Width of the game area
        height: Height of the game area
    """
    try:
        # Initialize fonts
        font_score = pygame.font.SysFont('Arial', 44, bold=True)
        font_medium = pygame.font.SysFont('Arial', 20, bold=True)
        font_small = pygame.font.SysFont('Arial', 16)
        font_badge = pygame.font.SysFont('Arial', 14, bold=True)
        
        # Colors
        white = (255, 255, 255)
        red_bright = (255, 80, 100)
        red_glow = (255, 60, 80)
        blue_bright = (80, 140, 255)
        blue_glow = (50, 120, 255)
        cyan = (60, 180, 220)
        panel_bg = (10, 15, 25)
        
        # ── Top panel ──
        panel_height = 60
        panel_surf = pygame.Surface((width, panel_height), pygame.SRCALPHA)
        # Gradient-ish panel: dark at top, fades out at bottom
        for y in range(panel_height):
            alpha = int(200 * (1 - y / panel_height))
            pygame.draw.line(panel_surf, (*panel_bg, alpha), (0, y), (width, y))
        # Accent line at bottom of panel
        pygame.draw.line(panel_surf, (*cyan, 80), (0, panel_height - 1),
                         (width, panel_height - 1), 1)
        screen.blit(panel_surf, (0, 0))
        
        # ── Scores with glow ──
        # Player 1 score (left quarter)
        score1_text = font_score.render(f"{score1}", True, red_bright)
        s1_x = width // 4 - score1_text.get_width() // 2
        s1_y = 8
        # Glow layer
        score1_glow = font_score.render(f"{score1}", True, red_glow)
        glow_surf1 = pygame.Surface(score1_glow.get_size(), pygame.SRCALPHA)
        glow_surf1.blit(score1_glow, (0, 0))
        glow_surf1.set_alpha(60)
        screen.blit(glow_surf1, (s1_x - 2, s1_y - 2))
        screen.blit(glow_surf1, (s1_x + 2, s1_y + 2))
        # Main score
        screen.blit(score1_text, (s1_x, s1_y))
        
        # Player 2 score (right quarter)
        score2_text = font_score.render(f"{score2}", True, blue_bright)
        s2_x = 3 * width // 4 - score2_text.get_width() // 2
        s2_y = 8
        # Glow layer
        score2_glow = font_score.render(f"{score2}", True, blue_glow)
        glow_surf2 = pygame.Surface(score2_glow.get_size(), pygame.SRCALPHA)
        glow_surf2.blit(score2_glow, (0, 0))
        glow_surf2.set_alpha(60)
        screen.blit(glow_surf2, (s2_x - 2, s2_y - 2))
        screen.blit(glow_surf2, (s2_x + 2, s2_y + 2))
        # Main score
        screen.blit(score2_text, (s2_x, s2_y))
        
        # ── VS divider ──
        vs_text = font_medium.render("VS", True, (*cyan,))
        screen.blit(vs_text, (width // 2 - vs_text.get_width() // 2, 18))
        
        # ── Episode / Step info (centered, below scores) ──
        info_text = font_small.render(
            f"Episode {episode}  ·  Step {step}", True, (180, 190, 210)
        )
        screen.blit(info_text, (width // 2 - info_text.get_width() // 2, 42))

        # ── Instruction tip ──
        tip_text = font_small.render(
            "Press 'v' to visualize next episode", True, (120, 130, 150)
        )
        screen.blit(tip_text, (width // 2 - tip_text.get_width() // 2, 60))
        
        # ── Epsilon values (under each score) ──
        eps1_text = font_small.render(f"ε {agent1_epsilon:.3f}", True, (200, 120, 130))
        screen.blit(eps1_text, (width // 4 - eps1_text.get_width() // 2,
                                s1_y + score1_text.get_height() + 2))
        
        eps2_text = font_small.render(f"ε {agent2_epsilon:.3f}", True, (120, 150, 220))
        screen.blit(eps2_text, (3 * width // 4 - eps2_text.get_width() // 2,
                                s2_y + score2_text.get_height() + 2))
        
        # ── Bottom bar ──
        bar_h = 28
        bar_surf = pygame.Surface((width, bar_h), pygame.SRCALPHA)
        for y in range(bar_h):
            alpha = int(180 * (y / bar_h))
            pygame.draw.line(bar_surf, (*panel_bg, alpha), (0, y), (width, y))
        # Accent line at top
        pygame.draw.line(bar_surf, (*cyan, 50), (0, 0), (width, 0), 1)
        screen.blit(bar_surf, (0, height - bar_h))
        
        # Player labels
        label1 = font_small.render("Player 1", True, red_bright)
        screen.blit(label1, (14, height - bar_h + 6))
        
        label2 = font_small.render("Player 2", True, blue_bright)
        screen.blit(label2, (width - label2.get_width() - 14, height - bar_h + 6))
        
        # VISUALIZATION badge (centered)
        badge_text = font_badge.render(" VISUALIZATION ", True, (10, 15, 25))
        badge_w = badge_text.get_width() + 12
        badge_h = badge_text.get_height() + 4
        badge_x = width // 2 - badge_w // 2
        badge_y = height - bar_h + (bar_h - badge_h) // 2
        # Badge background
        badge_surf = pygame.Surface((badge_w, badge_h), pygame.SRCALPHA)
        pygame.draw.rect(badge_surf, (*cyan, 200), (0, 0, badge_w, badge_h),
                         border_radius=4)
        screen.blit(badge_surf, (badge_x, badge_y))
        screen.blit(badge_text, (badge_x + 6, badge_y + 2))
        
    except Exception as e:
        # Fallback if font initialization fails
        print(f"Warning: Could not render UI text: {e}")


def cleanup_pygame() -> None:
    """
    Clean up Pygame resources.
    Should be called when the application exits.
    """
    try:
        pygame.quit()
    except:
        pass


# Example usage (for testing purposes)
if __name__ == "__main__":
    from environment import AirHockeyEnv

    env = AirHockeyEnv()
    _ = env.reset()

    screen, clock = init_pygame(env.width, env.height, "Air Hockey Visualizer Smoke Test")

    running = True
    episode = 1
    step = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        action1 = 0  # Stay
        action2 = 0  # Stay

        _, _, _, done, _ = env.step(action1, action2)
        step += 1

        draw_game(
            screen=screen,
            env=env,
            episode=episode,
            step=step,
            score1=env.score1,
            score2=env.score2,
            agent1_epsilon=0.0,
            agent2_epsilon=0.0,
            clock=clock,
        )

        if done:
            env.reset()
            episode += 1
            step = 0

    cleanup_pygame()