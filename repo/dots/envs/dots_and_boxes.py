import gymnasium.spaces as spaces
from typing import Tuple, Dict, Any
import gymnasium
from gymnasium import spaces
import numpy as np
import pygame
from typing import Tuple, Dict, Any

class DotsAndBoxesEnv(gymnasium.Env):
    """
    Custom Gymnasium environment for Dots and Boxes on a 3x3 grid of boxes (4x4 dots).
    Observation: 41-float array [16 h_lines (4x4), 16 v_lines (4x4), 9 boxes (3x3)].
    Actions: Discrete(32), 0-15: h_lines[r//4][r%4], 16-31: v_lines[(a-16)//4][(a-16)%4].
    Invalid actions incur -0.5 reward, no board change.
    Agent (player 0, value 1.0) vs random opponent (player 1, value -1.0).
    Rewards: +1 per agent box, +0.5 bonus if >=2 in one move, -0.5 invalid, -0.1 per opp box.
    Terminal reward: agent_boxes - opp_boxes.
    Supports render_mode='human' with Pygame visualization.
    """
    metadata = {'render_modes': ['human', None], 'render_fps': 5}
    def __init__(self, grid_size: int = 3, render_mode: str | None = None):
        super().__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(32)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(41,), dtype=np.float32
        )
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.np_random = None
        self.render_delay = 0.2
        self.dot_radius = 8
        self.dot_spacing = 150
        self.offset_x = 50
        self.offset_y = 50
        self.line_width = 6
        self.box_border_width = 4
        # Board state
        self.h_lines = None
        self.v_lines = None
        self.boxes = None
        self.current_player = None
    def reset(
            self, seed: int | None = None, options: Dict[str, Any] | None = None
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_player = 0
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()
        self.h_lines = np.zeros((self.grid_size + 1, self.grid_size + 1), dtype=np.float32)
        self.v_lines = np.zeros((self.grid_size + 1, self.grid_size + 1), dtype=np.float32)
        self.boxes = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        obs = self._get_obs()
        info = {'scores': {'agent': 0, 'opp': 0}}
        return obs, info


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # Agent's move (player 0)
        agent_capt = 0
        if self._is_valid_action(action):
            agent_capt = self._draw_line(action, player=0)
            reward += agent_capt * 1.0
            if agent_capt >= 2:
                reward += 0.5
        else:
            reward -= 0.5

        # Check for terminal after agent's move
        if self._is_terminal():
            agent_score = np.sum(self.boxes == 1)
            opp_score = np.sum(self.boxes == -1)
            reward += agent_score - opp_score
            terminated = True
        else:
            # Opponent's full turn if agent didn't capture
            if agent_capt == 0:
                while not self._is_terminal():
                    valid_actions = self.get_valid_actions()
                    if len(valid_actions) == 0:
                        break
                    opp_action = self.np_random.choice(valid_actions)
                    opp_capt = self._draw_line(opp_action, player=1)
                    reward -= 0.1 * opp_capt
                    if opp_capt == 0:
                        break

        # Final terminal check after opponent turn
        if self._is_terminal():
            agent_score = np.sum(self.boxes == 1)
            opp_score = np.sum(self.boxes == -1)
            reward += agent_score - opp_score
            terminated = True

        obs = self._get_obs()
        agent_score = int(np.sum(self.boxes == 1))
        opp_score = int(np.sum(self.boxes == -1))
        info = {'scores': {'agent': agent_score, 'opp': opp_score}}

        if terminated:
            print(f"Game over. Agent boxes: {agent_score}, Opp: {opp_score}, Reward: {reward:.2f}")

        return obs, reward, terminated, truncated, info

    def _draw_line(self, action: int, player: int) -> int:
        captured = 0
        if action < 16:
            # Horizontal line
            r = action // 4
            c = action % 4
            if c >= self.grid_size or self.h_lines[r, c] != 0:
                return 0
            self.h_lines[r, c] = 1.0
            # Check box above
            if r > 0 and self._is_box_complete(r - 1, c):
                self.boxes[r - 1, c] = 1.0 if player == 0 else -1.0
                captured += 1
            # Check box below
            if r < self.grid_size and self._is_box_complete(r, c):
                self.boxes[r, c] = 1.0 if player == 0 else -1.0
                captured += 1
        else:
            # Vertical line
            r = (action - 16) // 4
            c = (action - 16) % 4
            if r >= self.grid_size or self.v_lines[r, c] != 0:
                return 0
            self.v_lines[r, c] = 1.0
            # Check box left
            if c > 0 and self._is_box_complete(r, c - 1):
                self.boxes[r, c - 1] = 1.0 if player == 0 else -1.0
                captured += 1
            # Check box right
            if c < self.grid_size and self._is_box_complete(r, c):
                self.boxes[r, c] = 1.0 if player == 0 else -1.0
                captured += 1

        # Render if human mode
        if self.render_mode == 'human':
            self.render()
            pygame.display.flip()
            pygame.time.wait(int(self.render_delay * 1000))
        return captured

    def _is_box_complete(self, bi: int, bj: int) -> bool:
        return (
            self.h_lines[bi, bj] == 1.0 and
            self.h_lines[bi + 1, bj] == 1.0 and
            self.v_lines[bi, bj] == 1.0 and
            self.v_lines[bi, bj + 1] == 1.0 and
            self.boxes[bi, bj] == 0
        )

    def _is_valid_action(self, action: int) -> bool:
        if action < 16:
            r = action // 4
            c = action % 4
            return c < self.grid_size and self.h_lines[r, c] == 0
        else:
            r = (action - 16) // 4
            c = (action - 16) % 4
            return r < self.grid_size and self.v_lines[r, c] == 0

    def get_valid_actions(self) -> np.ndarray:
        valid = np.array([self._is_valid_action(a) for a in range(32)])
        return np.where(valid)[0]

    def _is_terminal(self) -> bool:
        return np.sum(self.boxes == 0) == 0

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([
            self.h_lines.ravel(),
            self.v_lines.ravel(),
            self.boxes.ravel()
        ]).astype(np.float32)

    def render(self):
        if self.render_mode is None:
            return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("Dots and Boxes")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        # Draw dots
        for i in range(self.grid_size + 1):
            for j in range(self.grid_size + 1):
                x = self.offset_x + i * self.dot_spacing
                y = self.offset_y + j * self.dot_spacing
                pygame.draw.circle(self.screen, (0, 0, 0), (int(x), int(y)), self.dot_radius)

        # Draw lines
        for r in range(self.grid_size + 1):
            for c in range(self.grid_size + 1):
                # Horizontal lines
                if self.h_lines[r, c] == 1.0:
                    x1 = self.offset_x + c * self.dot_spacing
                    y1 = self.offset_y + r * self.dot_spacing
                    x2 = self.offset_x + (c + 1) * self.dot_spacing
                    y2 = y1
                    pygame.draw.line(self.screen, (0, 0, 0), (int(x1), int(y1)), (int(x2), int(y2)), self.line_width)
                # Vertical lines
                if self.v_lines[r, c] == 1.0:
                    x1 = self.offset_x + c * self.dot_spacing
                    y1 = self.offset_y + r * self.dot_spacing
                    x2 = x1
                    y2 = self.offset_y + (r + 1) * self.dot_spacing
                    pygame.draw.line(self.screen, (0, 0, 0), (int(x1), int(y1)), (int(x2), int(y2)), self.line_width)

        # Draw owned boxes
        for bi in range(self.grid_size):
            for bj in range(self.grid_size):
                if self.boxes[bi, bj] != 0:
                    x1 = self.offset_x + bj * self.dot_spacing
                    y1 = self.offset_y + bi * self.dot_spacing
                    x2 = self.offset_x + (bj + 1) * self.dot_spacing
                    y2 = self.offset_y + (bi + 1) * self.dot_spacing
                    if self.boxes[bi, bj] == 1.0:
                        fill_color = (200, 255, 200)  # light green
                        border_color = (0, 255, 0)    # green
                    else:
                        fill_color = (255, 200, 200)  # light red
                        border_color = (255, 0, 0)    # red
                    # Fill
                    pygame.draw.rect(self.screen, fill_color, (int(x1 + 2), int(y1 + 2),
                                                              int(x2 - x1 - 4), int(y2 - y1 - 4)))
                    # Border
                    pygame.draw.rect(self.screen, border_color, (int(x1), int(y1),
                                                                 int(x2 - x1), int(y2 - y1)), self.box_border_width)

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
