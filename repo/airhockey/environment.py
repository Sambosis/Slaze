"""
Air Hockey Environment Module

This module implements the AirHockeyEnv class for simulating an air hockey game.
It handles all game physics, state management, and rendering for reinforcement learning.
"""

import numpy as np
import math
import pygame
from math import hypot
from typing import Tuple, Dict, Any, Optional

class AirHockeyEnv:
    """
    Air Hockey Environment for reinforcement learning.
    
    Simulates a 2D air hockey game with two paddles and a puck.
    Provides standard RL interface with reset() and step() methods.
    """
    
    def __init__(self, 
                 width: int = 800, 
                 height: int = 500,
                 paddle_radius: int = 30,
                 puck_radius: int = 20,
                 max_speed: float = 15.0,
                 friction: float = 0.99,
                 restitution: float = 0.95,
                 reward_config: Optional[Dict[str, float]] = None):
        """
        Initialize the air hockey environment.
        
        Args:
            width: Width of the table in pixels
            height: Height of the table in pixels
            paddle_radius: Radius of paddles in pixels
            puck_radius: Radius of puck in pixels
            max_speed: Maximum speed for puck and paddles
            friction: Friction coefficient for puck movement
            restitution: Coefficient of restitution for collisions
        """
        self.width = width
        self.height = height
        self.paddle_radius = paddle_radius
        self.puck_radius = puck_radius
        self.max_speed = max_speed
        self.friction = friction
        self.restitution = restitution
        
        # Reward configuration
        self.reward_config = {
            "goal": 1.0,
            "concede": -1.0,
            "hit_puck": 0.1,
            "step_penalty": 0.0,
            "boundary_penalty": 0.0,
            "puck_dir_bonus": 0.02,
            "defense_bonus": 0.01
        }
        if reward_config:
            self.reward_config.update(reward_config)
        
        # Goal area dimensions
        self.goal_width = 20
        self.goal_height = 120
        
        # Paddle movement speed
        self.paddle_speed = 8.0
        
        # Game state
        self.state = None
        self.done = False
        self.score1 = 0
        self.score2 = 0
        
        # Action mapping: 0=stay, 1=up, 2=down, 3=left, 4=right,
        #                 5=up-left, 6=up-right, 7=down-left, 8=down-right
        _d = np.sqrt(2) / 2  # ~0.7071, normalizes diagonal speed to match cardinal
        self.action_map = {
            0: np.array([0, 0]),     # stay
            1: np.array([0, -1]),    # up
            2: np.array([0, 1]),     # down
            3: np.array([-1, 0]),    # left
            4: np.array([1, 0]),     # right
            5: np.array([-_d, -_d]), # up-left
            6: np.array([_d, -_d]),  # up-right
            7: np.array([-_d, _d]),  # down-left
            8: np.array([_d, _d])    # down-right
        }
        
        # State dimension: 22 values
        # Normalized:
        # [paddle1_x, paddle1_y, paddle1_vx, paddle1_vy,
        #  paddle2_x, paddle2_y, paddle2_vx, paddle2_vy,
        #  puck_x, puck_y, puck_vx, puck_vy,
        #  rel_puck1_x, rel_puck1_y, rel_puck2_x, rel_puck2_y,
        #  dist_puck_left_goal, dist_puck_right_goal,
        #  puck_speed, paddle1_speed, paddle2_speed,
        #  puck_heading_angle]
        self.state_dim = 22
        
        # Colors for rendering (premium palette)
        self.table_color_dark = (15, 22, 36)      # Deep navy
        self.table_color_mid = (22, 33, 52)        # Slightly lighter
        self.border_color = (60, 180, 220)         # Cyan accent
        self.line_color = (60, 180, 220, 100)      # Translucent cyan
        self.paddle1_color = (255, 60, 80)         # Vivid red
        self.paddle1_color_dark = (180, 30, 45)    # Darker red for edge
        self.paddle2_color = (50, 120, 255)        # Vivid blue
        self.paddle2_color_dark = (25, 70, 180)    # Darker blue for edge
        self.puck_color = (220, 225, 235)          # Near-white
        self.puck_color_dark = (160, 165, 175)     # Slightly darker
        self.goal_glow_color = (255, 50, 50)       # Red glow for goals
        self.shadow_color = (0, 0, 0, 80)          # Translucent black
        
        # Pre-computed collision constants (avoid recomputing every step)
        self._combined_radius = self.puck_radius + self.paddle_radius
        self._combined_radius_sq = self._combined_radius ** 2
        self._max_speed_sq = self.max_speed ** 2
        
        # Initialize pygame for rendering (only if needed)
        self.pygame_initialized = False
        
        # Puck trail for motion blur effect
        self._puck_trail = []
        self._max_trail_length = 12
        
        # Pre-built surfaces cache (built on first render)
        self._render_cache = {}

    def _get_obs(self) -> np.ndarray:
        """Calculate normalized observation from current internal state (22 features)."""
        p1x, p1y, p1vx, p1vy = self.state[0:4]
        p2x, p2y, p2vx, p2vy = self.state[4:8]
        px, py, pvx, pvy = self.state[8:12]
        
        w, h = float(self.width), float(self.height)
        ms = float(self.max_speed)
        
        # Distance from puck to each goal (normalized)
        dist_puck_left_goal = px / w
        dist_puck_right_goal = (w - px) / w
        
        # Scalar speeds (normalized)
        puck_speed = math.sqrt(pvx * pvx + pvy * pvy) / ms
        paddle1_speed = math.sqrt(p1vx * p1vx + p1vy * p1vy) / ms
        paddle2_speed = math.sqrt(p2vx * p2vx + p2vy * p2vy) / ms
        
        # Puck heading angle (−1 to 1, normalized by π)
        puck_heading = math.atan2(pvy, pvx) / math.pi if (pvx != 0 or pvy != 0) else 0.0
        
        return np.array([
            p1x / w, p1y / h, p1vx / ms, p1vy / ms,
            p2x / w, p2y / h, p2vx / ms, p2vy / ms,
            px / w, py / h, pvx / ms, pvy / ms,
            (px - p1x) / w, (py - p1y) / h,
            (px - p2x) / w, (py - p2y) / h,
            dist_puck_left_goal, dist_puck_right_goal,
            puck_speed, paddle1_speed, paddle2_speed,
            puck_heading
        ], dtype=np.float32)

    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state observation as numpy array
        """
        # Reset scores
        self.score1 = 0
        self.score2 = 0
        self.done = False
        
        # Clear puck trail on reset
        self._puck_trail.clear()
        
        # Initialize positions
        # Paddle 1 (left side)
        paddle1_x = 100
        paddle1_y = self.height // 2
        paddle1_vx = 0
        paddle1_vy = 0
        
        # Paddle 2 (right side)
        paddle2_x = self.width - 100
        paddle2_y = self.height // 2
        paddle2_vx = 0
        paddle2_vy = 0
        
        # Puck (center)
        puck_x = self.width // 2
        puck_y = self.height // 2
        puck_vx = np.random.uniform(-8, 8)
        puck_vy = np.random.uniform(-8, 8)
        
        # Create internal state vector
        self.state = np.array([
            paddle1_x, paddle1_y, paddle1_vx, paddle1_vy,
            paddle2_x, paddle2_y, paddle2_vx, paddle2_vy,
            puck_x, puck_y, puck_vx, puck_vy
        ], dtype=np.float32)
        
        return self._get_obs()
    
    def step(self, action1: int, action2: int) -> Tuple[np.ndarray, float, float, bool, Dict[str, Any]]:
        """
        Execute one time step for both agents.
        
        Args:
            action1: Action for agent 1 (0=stay, 1=up, 2=down)
            action2: Action for agent 2 (0=stay, 1=up, 2=down)
            
        Returns:
            Tuple containing:
                - next_state: Updated state observation
                - reward1: Reward for agent 1
                - reward2: Reward for agent 2
                - done: Whether the episode is finished
                - info: Additional information dictionary
        """
        if self.done:
            raise ValueError("Episode is already finished. Call reset() first.")
        
        # Extract current state components
        paddle1_x, paddle1_y, paddle1_vx, paddle1_vy = self.state[0:4]
        paddle2_x, paddle2_y, paddle2_vx, paddle2_vy = self.state[4:8]
        puck_x, puck_y, puck_vx, puck_vy = self.state[8:12]
        
        # Proximity-weighted step penalty: penalize proportionally to distance from the puck.
        # Agents near the puck incur little penalty; agents ignoring the puck incur the full penalty.
        _max_dist = math.sqrt(self.width ** 2 + self.height ** 2)
        _dist1 = math.sqrt((paddle1_x - puck_x) ** 2 + (paddle1_y - puck_y) ** 2)
        _dist2 = math.sqrt((paddle2_x - puck_x) ** 2 + (paddle2_y - puck_y) ** 2)
        
        breakdown1 = {k: 0.0 for k in self.reward_config}
        breakdown2 = {k: 0.0 for k in self.reward_config}
        
        reward1 = self.reward_config["step_penalty"] * (_dist1 / _max_dist)
        reward2 = self.reward_config["step_penalty"] * (_dist2 / _max_dist)
        breakdown1["step_penalty"] = reward1
        breakdown2["step_penalty"] = reward2
        
        # Apply actions to paddles
        # Map actions to velocity changes
        action1_vec = self.action_map.get(action1, np.array([0, 0]))
        action2_vec = self.action_map.get(action2, np.array([0, 0]))
        
        # Update paddle velocities (with some damping)
        paddle1_vx = 0.8 * paddle1_vx + self.paddle_speed * action1_vec[0]
        paddle1_vy = 0.8 * paddle1_vy + self.paddle_speed * action1_vec[1]
        paddle2_vx = 0.8 * paddle2_vx + self.paddle_speed * action2_vec[0]
        paddle2_vy = 0.8 * paddle2_vy + self.paddle_speed * action2_vec[1]
        
        # Update paddle positions
        paddle1_x += paddle1_vx
        paddle1_y += paddle1_vy
        paddle2_x += paddle2_vx
        paddle2_y += paddle2_vy
        
        # Constrain paddles to their halves and within table bounds
        # Apply boundary penalty AND zero the velocity component into the wall
        p1_hit_boundary = False
        if paddle1_x < self.paddle_radius:
            paddle1_x = self.paddle_radius
            paddle1_vx = max(0.0, paddle1_vx)
            p1_hit_boundary = True
        elif paddle1_x > self.width // 2 - self.paddle_radius:
            paddle1_x = self.width // 2 - self.paddle_radius
            paddle1_vx = min(0.0, paddle1_vx)
            p1_hit_boundary = True
        if paddle1_y < self.paddle_radius:
            paddle1_y = self.paddle_radius
            paddle1_vy = max(0.0, paddle1_vy)
            p1_hit_boundary = True
        elif paddle1_y > self.height - self.paddle_radius:
            paddle1_y = self.height - self.paddle_radius
            paddle1_vy = min(0.0, paddle1_vy)
            p1_hit_boundary = True
        if p1_hit_boundary:
            reward1 += self.reward_config["boundary_penalty"]
            breakdown1["boundary_penalty"] += self.reward_config["boundary_penalty"]

        p2_hit_boundary = False
        if paddle2_x < self.width // 2 + self.paddle_radius:
            paddle2_x = self.width // 2 + self.paddle_radius
            paddle2_vx = max(0.0, paddle2_vx)
            p2_hit_boundary = True
        elif paddle2_x > self.width - self.paddle_radius:
            paddle2_x = self.width - self.paddle_radius
            paddle2_vx = min(0.0, paddle2_vx)
            p2_hit_boundary = True
        if paddle2_y < self.paddle_radius:
            paddle2_y = self.paddle_radius
            paddle2_vy = max(0.0, paddle2_vy)
            p2_hit_boundary = True
        elif paddle2_y > self.height - self.paddle_radius:
            paddle2_y = self.height - self.paddle_radius
            paddle2_vy = min(0.0, paddle2_vy)
            p2_hit_boundary = True
        if p2_hit_boundary:
            reward2 += self.reward_config["boundary_penalty"]
            breakdown2["boundary_penalty"] += self.reward_config["boundary_penalty"]
        
        # Update puck position
        puck_x += puck_vx
        puck_y += puck_vy
        
        # Apply friction to puck
        puck_vx *= self.friction
        puck_vy *= self.friction
        
        # Cap puck speed (squared comparison avoids sqrt in common case)
        puck_speed_sq = puck_vx * puck_vx + puck_vy * puck_vy
        if puck_speed_sq > self._max_speed_sq:
            scale = self.max_speed / math.sqrt(puck_speed_sq)
            puck_vx *= scale
            puck_vy *= scale
        
        # Check collisions with walls
        # Top and bottom walls
        if puck_y - self.puck_radius <= 0:
            puck_y = self.puck_radius
            puck_vy = -puck_vy * self.restitution
        elif puck_y + self.puck_radius >= self.height:
            puck_y = self.height - self.puck_radius
            puck_vy = -puck_vy * self.restitution
        
        # Left and right walls (excluding goal areas)
        if puck_x - self.puck_radius <= 0:
            # Check if in goal area
            if (self.height // 2 - self.goal_height // 2 <= puck_y <= 
                self.height // 2 + self.goal_height // 2):
                # Goal for player 2
                self.done = True
                reward1 += self.reward_config["concede"]
                reward2 += self.reward_config["goal"]
                breakdown1["concede"] += self.reward_config["concede"]
                breakdown2["goal"] += self.reward_config["goal"]
                self.score2 += 1
            else:
                puck_x = self.puck_radius
                puck_vx = -puck_vx * self.restitution
        elif puck_x + self.puck_radius >= self.width:
            # Check if in goal area
            if (self.height // 2 - self.goal_height // 2 <= puck_y <= 
                self.height // 2 + self.goal_height // 2):
                # Goal for player 1
                self.done = True
                reward1 += self.reward_config["goal"]
                reward2 += self.reward_config["concede"]
                breakdown1["goal"] += self.reward_config["goal"]
                breakdown2["concede"] += self.reward_config["concede"]
                self.score1 += 1
            else:
                puck_x = self.width - self.puck_radius
                puck_vx = -puck_vx * self.restitution
        
        # Check collisions with paddles
        # Paddle 1 collision (squared-distance to skip sqrt when no collision)
        dx1 = puck_x - paddle1_x
        dy1 = puck_y - paddle1_y
        dist1_sq = dx1 * dx1 + dy1 * dy1
        if dist1_sq < self._combined_radius_sq:
            dist1 = math.sqrt(dist1_sq)
            # Calculate collision response
            nx = dx1 / dist1
            ny = dy1 / dist1
            
            # Separate puck from paddle
            overlap = self._combined_radius - dist1
            puck_x += nx * overlap * 0.5
            puck_y += ny * overlap * 0.5
            
            # Calculate relative velocity
            rel_vx = puck_vx - paddle1_vx
            rel_vy = puck_vy - paddle1_vy
            rel_vn = rel_vx * nx + rel_vy * ny
            
            # Apply impulse (elastic collision)
            if rel_vn < 0:
                impulse = -(1 + self.restitution) * rel_vn
                puck_vx += impulse * nx
                puck_vy += impulse * ny
                
                # Small reward for hitting the puck
                reward1 += self.reward_config["hit_puck"]
                breakdown1["hit_puck"] += self.reward_config["hit_puck"]
        
        # Paddle 2 collision (squared-distance to skip sqrt when no collision)
        dx2 = puck_x - paddle2_x
        dy2 = puck_y - paddle2_y
        dist2_sq = dx2 * dx2 + dy2 * dy2
        if dist2_sq < self._combined_radius_sq:
            dist2 = math.sqrt(dist2_sq)
            # Calculate collision response
            nx = dx2 / dist2
            ny = dy2 / dist2
            
            # Separate puck from paddle
            overlap = self._combined_radius - dist2
            puck_x += nx * overlap * 0.5
            puck_y += ny * overlap * 0.5
            
            # Calculate relative velocity
            rel_vx = puck_vx - paddle2_vx
            rel_vy = puck_vy - paddle2_vy
            rel_vn = rel_vx * nx + rel_vy * ny
            
            # Apply impulse (elastic collision)
            if rel_vn < 0:
                impulse = -(1 + self.restitution) * rel_vn
                puck_vx += impulse * nx
                puck_vy += impulse * ny
                
                # Small reward for hitting the puck
                reward2 += self.reward_config["hit_puck"]
                breakdown2["hit_puck"] += self.reward_config["hit_puck"]
        
        # ── Reward Shaping ─────────────────────────────────────────────
        # These small dense rewards guide learning without overwhelming
        # the sparse goal/concede signal.
        
        if not self.done:
            center_y = self.height / 2.0
            
            # 1) Puck-toward-opponent-goal bonus:
            #    Reward when the puck is moving toward the opponent's goal.
            #    Agent1 wants puck moving right (+vx), Agent2 wants it left (-vx).
            puck_speed_val = math.sqrt(puck_vx * puck_vx + puck_vy * puck_vy)
            if puck_speed_val > 0.5:  # only when puck is actually moving
                b1 = self.reward_config["puck_dir_bonus"] * (puck_vx / self.max_speed)
                b2 = self.reward_config["puck_dir_bonus"] * (-puck_vx / self.max_speed)
                reward1 += b1   # positive when moving right
                reward2 += b2  # positive when moving left
                breakdown1["puck_dir_bonus"] += b1
                breakdown2["puck_dir_bonus"] += b2
            
            # 2) Defensive positioning bonus:
            #    Reward when paddle moves toward the puck on its own side.
            #    Encourages interception rather than just sitting still.
            half_w = self.width / 2.0
            # Agent 1 (left side): puck is on its side when px < half_w
            if puck_x < half_w:
                dist_to_puck = math.sqrt((paddle1_x - puck_x)**2 + (paddle1_y - puck_y)**2)
                max_dist = math.sqrt(half_w**2 + self.height**2)
                d1 = self.reward_config["defense_bonus"] * (1.0 - dist_to_puck / max_dist)
                reward1 += d1
                breakdown1["defense_bonus"] += d1
            # Agent 2 (right side): puck is on its side when px >= half_w
            if puck_x >= half_w:
                dist_to_puck = math.sqrt((paddle2_x - puck_x)**2 + (paddle2_y - puck_y)**2)
                max_dist = math.sqrt(half_w**2 + self.height**2)
                d2 = self.reward_config["defense_bonus"] * (1.0 - dist_to_puck / max_dist)
                reward2 += d2
                breakdown2["defense_bonus"] += d2
        
        # Update state
        self.state = np.array([
            paddle1_x, paddle1_y, paddle1_vx, paddle1_vy,
            paddle2_x, paddle2_y, paddle2_vx, paddle2_vy,
            puck_x, puck_y, puck_vx, puck_vy
        ], dtype=np.float32)
        
        # Info dictionary
        info = {
            'score1': self.score1,
            'score2': self.score2,
            'paddle1_pos': (paddle1_x, paddle1_y),
            'paddle2_pos': (paddle2_x, paddle2_y),
            'puck_pos': (puck_x, puck_y),
            'reward_breakdown1': breakdown1,
            'reward_breakdown2': breakdown2
        }
        
        return self._get_obs(), reward1, reward2, self.done, info
    
    # ── Rendering helpers ─────────────────────────────────────────────
    
    def _build_render_cache(self) -> None:
        """Pre-build expensive surfaces that don't change between frames."""
        # ── Table background with subtle vertical gradient ──
        bg = pygame.Surface((self.width, self.height))
        for y in range(self.height):
            t = y / self.height
            r = int(self.table_color_dark[0] * (1 - t) + self.table_color_mid[0] * t)
            g = int(self.table_color_dark[1] * (1 - t) + self.table_color_mid[1] * t)
            b = int(self.table_color_dark[2] * (1 - t) + self.table_color_mid[2] * t)
            pygame.draw.line(bg, (r, g, b), (0, y), (self.width, y))
        self._render_cache['bg'] = bg
        
        # ── Goal glow surfaces ──
        goal_glow_w = self.goal_width + 30
        goal_glow_h = self.goal_height + 30
        for side in ('left', 'right'):
            glow_surf = pygame.Surface((goal_glow_w, goal_glow_h), pygame.SRCALPHA)
            for i in range(15, 0, -1):
                alpha = int(25 * (i / 15))
                color = (255, 50, 50, alpha)
                inflate = i * 2
                pygame.draw.rect(glow_surf, color,
                                 (15 - i, 15 - i,
                                  self.goal_width + inflate,
                                  self.goal_height + inflate),
                                 border_radius=4)
            self._render_cache[f'goal_glow_{side}'] = glow_surf
        
        # ── Paddle surfaces (radial gradient look) ──
        for name, color_bright, color_dark in [
            ('paddle1', self.paddle1_color, self.paddle1_color_dark),
            ('paddle2', self.paddle2_color, self.paddle2_color_dark),
        ]:
            r = self.paddle_radius
            size = r * 2 + 4
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            cx, cy = size // 2, size // 2
            # Draw concentric circles from outside→in with color gradient
            for i in range(r, 0, -1):
                t = i / r  # 1.0 at edge → 0.0 at center
                cr = int(color_dark[0] * t + color_bright[0] * (1 - t))
                cg = int(color_dark[1] * t + color_bright[1] * (1 - t))
                cb = int(color_dark[2] * t + color_bright[2] * (1 - t))
                pygame.draw.circle(surf, (cr, cg, cb, 255), (cx, cy), i)
            # White highlight crescent (top-left)
            highlight = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.circle(highlight, (255, 255, 255, 70),
                               (cx - r // 4, cy - r // 4), r // 2)
            surf.blit(highlight, (0, 0))
            # Outer ring
            pygame.draw.circle(surf, (255, 255, 255, 120), (cx, cy), r, 2)
            # Inner knob circle
            pygame.draw.circle(surf, (255, 255, 255, 50), (cx, cy), r // 3, 1)
            self._render_cache[name] = surf
        
        # ── Puck surface ──
        pr = self.puck_radius
        psize = pr * 2 + 4
        puck_surf = pygame.Surface((psize, psize), pygame.SRCALPHA)
        pcx, pcy = psize // 2, psize // 2
        # Concentric gradient
        for i in range(pr, 0, -1):
            t = i / pr
            cr = int(self.puck_color_dark[0] * t + self.puck_color[0] * (1 - t))
            cg = int(self.puck_color_dark[1] * t + self.puck_color[1] * (1 - t))
            cb = int(self.puck_color_dark[2] * t + self.puck_color[2] * (1 - t))
            pygame.draw.circle(puck_surf, (cr, cg, cb, 255), (pcx, pcy), i)
        # Highlight dot
        pygame.draw.circle(puck_surf, (255, 255, 255, 120),
                           (pcx - pr // 4, pcy - pr // 4), pr // 3)
        # Edge ring
        pygame.draw.circle(puck_surf, (200, 205, 215, 180), (pcx, pcy), pr, 2)
        self._render_cache['puck'] = puck_surf
        
        # ── Shadow surface (reusable ellipse) ──
        shadow = pygame.Surface((pr * 2 + 8, pr + 8), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 50),
                            (0, 0, pr * 2 + 8, pr + 8))
        self._render_cache['puck_shadow'] = shadow
        
        paddle_shadow = pygame.Surface((self.paddle_radius * 2 + 8,
                                        self.paddle_radius + 8), pygame.SRCALPHA)
        pygame.draw.ellipse(paddle_shadow, (0, 0, 0, 45),
                            (0, 0, self.paddle_radius * 2 + 8,
                             self.paddle_radius + 8))
        self._render_cache['paddle_shadow'] = paddle_shadow
    
    def render(self, screen: pygame.Surface) -> None:
        """
        Draw the current game state with premium visuals.
        
        Args:
            screen: Pygame surface to draw on
        """
        # Initialize pygame if not already done
        if not self.pygame_initialized:
            pygame.init()
            self.pygame_initialized = True
        
        # Build cached surfaces on first render call
        if not self._render_cache:
            self._build_render_cache()
        
        # ── 1. Background gradient ──
        screen.blit(self._render_cache['bg'], (0, 0))
        
        # ── 2. Table border glow ──
        border_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        # Outer glow layers
        for i in range(4, 0, -1):
            alpha = int(30 * (i / 4))
            pygame.draw.rect(border_surf, (*self.border_color, alpha),
                             (0, 0, self.width, self.height), i + 1, border_radius=8)
        # Crisp inner border
        pygame.draw.rect(border_surf, (*self.border_color, 180),
                         (0, 0, self.width, self.height), 2, border_radius=8)
        screen.blit(border_surf, (0, 0))
        
        # ── 3. Surface markings ──
        marking_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        mark_color = (60, 180, 220, 60)
        
        # Center line (dashed)
        cx = self.width // 2
        dash_len = 12
        gap = 8
        y = 10
        while y < self.height - 10:
            end_y = min(y + dash_len, self.height - 10)
            pygame.draw.line(marking_surf, mark_color, (cx, y), (cx, end_y), 2)
            y += dash_len + gap
        
        # Center circle
        cy = self.height // 2
        pygame.draw.circle(marking_surf, mark_color, (cx, cy), 70, 2)
        # Center dot
        pygame.draw.circle(marking_surf, (60, 180, 220, 90), (cx, cy), 6)
        
        # Defensive zone arcs
        arc_radius = 90
        # Left arc
        pygame.draw.arc(marking_surf, mark_color,
                        (self.goal_width - arc_radius // 2,
                         cy - arc_radius,
                         arc_radius, arc_radius * 2),
                        -math.pi / 2, math.pi / 2, 2)
        # Right arc
        pygame.draw.arc(marking_surf, mark_color,
                        (self.width - self.goal_width - arc_radius // 2,
                         cy - arc_radius,
                         arc_radius, arc_radius * 2),
                        math.pi / 2, 3 * math.pi / 2, 2)
        
        # Corner arcs
        corner_r = 30
        corners = [
            (0, 0, 0, math.pi / 2),                                      # top-left
            (self.width - corner_r * 2, 0, -math.pi / 2, 0),             # top-right
            (0, self.height - corner_r * 2, math.pi / 2, math.pi),       # bottom-left
            (self.width - corner_r * 2, self.height - corner_r * 2,
             math.pi, 3 * math.pi / 2),                                  # bottom-right
        ]
        for rx, ry, a1, a2 in corners:
            pygame.draw.arc(marking_surf, mark_color,
                            (rx, ry, corner_r * 2, corner_r * 2), a1, a2, 2)
        
        screen.blit(marking_surf, (0, 0))
        
        # ── 4. Goal areas with glow ──
        goal_y = self.height // 2 - self.goal_height // 2
        # Left goal glow
        glow_l = self._render_cache['goal_glow_left']
        screen.blit(glow_l, (-15, goal_y - 15))
        # Left goal solid
        pygame.draw.rect(screen, (180, 40, 40),
                         (0, goal_y, self.goal_width, self.goal_height),
                         border_radius=3)
        pygame.draw.rect(screen, (255, 80, 80),
                         (0, goal_y, self.goal_width, self.goal_height), 2,
                         border_radius=3)
        
        # Right goal glow
        glow_r = self._render_cache['goal_glow_right']
        screen.blit(glow_r, (self.width - self.goal_width - 15, goal_y - 15))
        # Right goal solid
        pygame.draw.rect(screen, (180, 40, 40),
                         (self.width - self.goal_width, goal_y,
                          self.goal_width, self.goal_height),
                         border_radius=3)
        pygame.draw.rect(screen, (255, 80, 80),
                         (self.width - self.goal_width, goal_y,
                          self.goal_width, self.goal_height), 2,
                         border_radius=3)
        
        # ── 5. Game objects ──
        if self.state is not None:
            paddle1_x, paddle1_y = self.state[0], self.state[1]
            paddle2_x, paddle2_y = self.state[4], self.state[5]
            puck_x, puck_y = self.state[8], self.state[9]
            puck_vx, puck_vy = self.state[10], self.state[11]
            
            # Update puck trail
            speed_sq = puck_vx * puck_vx + puck_vy * puck_vy
            self._puck_trail.append((int(puck_x), int(puck_y), speed_sq))
            if len(self._puck_trail) > self._max_trail_length:
                self._puck_trail.pop(0)
            
            # Draw puck trail (only when moving fast enough)
            if speed_sq > 4.0:
                trail_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                n = len(self._puck_trail)
                for i, (tx, ty, _) in enumerate(self._puck_trail[:-1]):
                    t = (i + 1) / n
                    alpha = int(40 * t)
                    radius = max(2, int(self.puck_radius * t * 0.6))
                    pygame.draw.circle(trail_surf, (60, 180, 220, alpha),
                                       (tx, ty), radius)
                screen.blit(trail_surf, (0, 0))
            
            # ── Shadows ──
            p_shadow = self._render_cache['paddle_shadow']
            sw, sh = p_shadow.get_size()
            screen.blit(p_shadow, (int(paddle1_x) - sw // 2 + 3,
                                   int(paddle1_y) - sh // 2 + 6))
            screen.blit(p_shadow, (int(paddle2_x) - sw // 2 + 3,
                                   int(paddle2_y) - sh // 2 + 6))
            
            puck_sh = self._render_cache['puck_shadow']
            psw, psh = puck_sh.get_size()
            screen.blit(puck_sh, (int(puck_x) - psw // 2 + 2,
                                  int(puck_y) - psh // 2 + 5))
            
            # ── Paddles (pre-built gradient surfaces) ──
            p1 = self._render_cache['paddle1']
            p2 = self._render_cache['paddle2']
            ps = p1.get_size()
            screen.blit(p1, (int(paddle1_x) - ps[0] // 2,
                             int(paddle1_y) - ps[1] // 2))
            screen.blit(p2, (int(paddle2_x) - ps[0] // 2,
                             int(paddle2_y) - ps[1] // 2))
            
            # ── Puck ──
            puck_s = self._render_cache['puck']
            pks = puck_s.get_size()
            screen.blit(puck_s, (int(puck_x) - pks[0] // 2,
                                 int(puck_y) - pks[1] // 2))
    
    def get_state_dim(self) -> int:
        """Return the dimension of the state vector."""
        return self.state_dim
    
    def get_action_dim(self) -> int:
        """Return the number of possible actions."""
        return len(self.action_map)