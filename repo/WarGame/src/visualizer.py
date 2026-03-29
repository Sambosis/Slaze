import os
import sys
import pygame
import torch
import numpy as np
from typing import Dict, Tuple, List, Optional

import config
from src.environment import WarGameEnv, Unit
from src.model import Agent

def render_game(env: WarGameEnv, agent_a: Agent, agent_b: Agent) -> None:
    """
    Runs a Pygame loop to visually display a match between two agents.
    
    This function takes control of the environment loop, manually inferring actions
    for both agents (overriding the environment's internal random opponent logic) 
    to ensure a true Head-to-Head visualization between the specific provided agents.
    
    Args:
        env: The game environment instance.
        agent_a: The primary learning agent (Player A - Blue).
        agent_b: The opponent agent (Player B - Red).
    """
    # --- Configuration & Setup ---
    cfg = config.CFG
    
    # Initialize Pygame (Headless/Server compatibility)
    # Set dummy audio driver to prevent ALSA errors on servers
    try:
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
        pygame.mixer.quit()
    except Exception:
        pass

    if not pygame.get_init():
        pygame.init()
        
    pygame.display.set_caption("Operation Neural Storm | Spectator Mode")
    
    # Window Setup
    screen_width = cfg.view.SCREEN_WIDTH
    screen_height = cfg.view.SCREEN_HEIGHT
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    
    # Load Icon
    icon_path = os.path.join(cfg.paths.ASSETS_DIR, "icon.png")
    if os.path.exists(icon_path):
        try:
            icon = pygame.image.load(icon_path)
            pygame.display.set_icon(icon)
        except pygame.error:
            pass

    # Timing and Fonts
    clock = pygame.time.Clock()
    font_ui = pygame.font.SysFont("Consolas", 14)
    font_hud = pygame.font.SysFont("Consolas", 12, bold=True)
    font_large = pygame.font.SysFont("Arial", 48, bold=True)
    font_header = pygame.font.SysFont("Arial", 20, bold=True)

    # Grid Layout Calculations (Center the board)
    cell_size = cfg.game.CELL_SIZE
    grid_w_px = cfg.game.GRID_WIDTH * cell_size
    grid_h_px = cfg.game.GRID_HEIGHT * cell_size
    
    grid_offset_x = (screen_width - grid_w_px) // 2
    grid_offset_y = (screen_height - grid_h_px) // 2 + 20

    # --- Helper Functions ---

    def get_inference(agent: Agent, state_tensor: torch.Tensor) -> Tuple[Dict[str, int], float]:
        """
        Infers greedy actions (argmax) and critic value from the agent.
        Returns: (Action Dictionary, Win Probability)
        """
        agent.eval()
        device = next(agent.parameters()).device
        state = state_tensor.to(device)
        
        # Ensure batch dimension [1, C, H, W]
        if state.dim() == 3:
            state = state.unsqueeze(0)

        with torch.no_grad():
            embedding = agent.calculate_embedding(state)
            
            # Actor: Get logits, reshape to [1, Num_Units, Action_Dim]
            logits = agent.actor(embedding)
            logits = logits.view(-1, agent.num_units, agent.action_dim)
            
            # Greedy Action Selection
            actions = torch.argmax(logits, dim=-1).cpu().numpy()[0]
            
            # Critic: Get value, convert to probability via sigmoid
            value = agent.critic(embedding).item()
            win_prob = torch.sigmoid(torch.tensor(value)).item()

        # Map to unit_id format expected by environment
        action_dict = {f"unit_{i}": int(actions[i]) for i in range(len(actions))}
        return action_dict, win_prob

    def draw_grid() -> None:
        """Draws the background, grid lines, and terrain obstacles."""
        screen.fill(cfg.view.COLOR_BG)
        
        # Draw Terrain/Obstacles
        for y in range(cfg.game.GRID_HEIGHT):
            for x in range(cfg.game.GRID_WIDTH):
                rect = pygame.Rect(
                    grid_offset_x + x * cell_size,
                    grid_offset_y + y * cell_size,
                    cell_size, cell_size
                )
                
                # Grid Lines
                pygame.draw.rect(screen, cfg.view.COLOR_GRID, rect, 1)
                
                # Obstacles
                if env.terrain[y, x] > 0.5:
                    # Draw slightly smaller rect for obstacle to show grid lines
                    obs_rect = rect.inflate(-2, -2)
                    pygame.draw.rect(screen, cfg.view.COLOR_OBSTACLE, obs_rect)

    def draw_units(units: List[Unit], color: Tuple[int, int, int]) -> None:
        """Draws units with specific shapes for Tanks vs Artillery."""
        for unit in units:
            if not unit.is_alive():
                continue
            
            # Coordinates
            cx = grid_offset_x + unit.x * cell_size + cell_size // 2
            cy = grid_offset_y + unit.y * cell_size + cell_size // 2
            size = cell_size - 8
            
            # Draw Body
            if unit.unit_type == 'tank':
                # Tank: Heavy rectangle with turret
                rect = pygame.Rect(cx - size//2, cy - size//2, size, size)
                pygame.draw.rect(screen, color, rect)
                # Turret accent (white)
                pygame.draw.rect(screen, (255, 255, 255), 
                               (cx - 3, cy - size//2 + 2, 6, size//2))
            else:
                # Artillery: Triangle pointing up
                points = [
                    (cx, cy - size//2),
                    (cx - size//2, cy + size//2),
                    (cx + size//2, cy + size//2)
                ]
                pygame.draw.polygon(screen, color, points)
                # Barrel (white line)
                pygame.draw.line(screen, (255, 255, 255), 
                               (cx, cy - size//2), (cx, cy - size), 3)

            # Health Bar
            hp_ratio = unit.hp_normalized
            bar_width = cell_size - 4
            bar_height = 4
            bar_x = cx - bar_width // 2
            bar_y = cy + size // 2 + 2
            
            # Background (Dark Gray)
            pygame.draw.rect(screen, (50, 50, 50), 
                           (bar_x, bar_y, bar_width, bar_height))
            
            # Fill (Green/Red based on health)
            hp_color = cfg.view.COLOR_HP_FULL if hp_ratio > 0.4 else cfg.view.COLOR_HP_LOW
            fill_width = int(bar_width * hp_ratio)
            if fill_width > 0:
                pygame.draw.rect(screen, hp_color, 
                               (bar_x, bar_y, fill_width, bar_height))

    def draw_ui(win_prob_a: float, win_prob_b: float, step: int) -> None:
        """Draws HUD with win probabilities, step count, and unit stats."""
        # Header Info
        header_text = f"Episode Step: {step} / {cfg.game.MAX_STEPS}"
        surf_header = font_header.render(header_text, True, cfg.view.COLOR_TEXT)
        screen.blit(surf_header, (screen_width // 2 - surf_header.get_width() // 2, 10))

        # Stats Panel Background
        panel_y = 40
        
        # Agent A Stats (Left - Blue)
        a_alive = sum(1 for u in env.units_player_a if u.is_alive())
        a_hp = sum(u.hp for u in env.units_player_a if u.is_alive())
        
        text_a_1 = font_hud.render(f"AGENT A (BLUE)", True, cfg.view.COLOR_AGENT_A)
        text_a_2 = font_ui.render(f"Units: