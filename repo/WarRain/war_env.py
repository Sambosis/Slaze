import gymnasium as gym
import gymnasium
from gymnasium import spaces
import pygame
import numpy as np
from collections import deque
import cv2
import math

def preprocess_frame(surface: pygame.Surface) -> np.ndarray:
    """
    Converts a Pygame surface to an 84x84 grayscale frame normalized to [0,1].
    """
    arr = pygame.surfarray.array3d(surface)
    # Transpose to (height, width, 3) for OpenCV
    arr = np.transpose(arr, (1, 0, 2))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0

class WarEnv(gymnasium.Env):
    """
    Custom Gymnasium environment for a 2D top-down tank battle game.
    Agent controls a blue tank, fights up to 5 red enemy tanks.
    Observation: 84x84x4 grayscale frame stack.
    Actions repeat for 4 frames Atari-style.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None):
        super().__init__()
        pygame.init()
        assert render_mode in [None, 'human', 'rgb_array']
        self.render_mode = render_mode
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(84, 84, 4), dtype=np.float32
        )
        self.action_space = spaces.Discrete(6)
        self.screen = None
        self.clock = pygame.time.Clock()
        self.dt = 1.0 / 60.0
        self.action_repeat = 4
        self.tank_radius = 15
        self.bullet_radius = 3
        self.agent_speed = 150.0
        self.bullet_speed = 400.0
        self.max_steps = 5000
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()

        self.agent_pos = np.array([200.0, 200.0])
        self.agent_vel = np.zeros(2)
        self.agent_dir = np.array([0.0, -1.0])
        self.agent_cooldown = 0
        self.enemies = []
        self.agent_bullets = []
        self.enemy_bullets = []
        self.score = 0
        self.frame_count = 0
        self.steps = 0
        self.recent_obs = deque(maxlen=4)

        if self.screen is None:
            if self.render_mode == 'human':
                pygame.init()
                self.screen = pygame.display.set_mode((400, 400))
            else:
                self.screen = pygame.Surface((400, 400))

        # Render initial frames
        for _ in range(4):
            self._render_frame()
            frame = preprocess_frame(self.screen)
            self.recent_obs.append(frame)

        obs = np.stack(list(self.recent_obs), axis=-1)
        info = {}
        return obs, info

    def step(self, action):
        reward = 0.0
        terminal = False
        truncated = False
        
        for _ in range(self.action_repeat):
            # Update cooldowns
            self._update_cooldowns()
            # Apply action
            self._apply_action(action)
            # Update enemy velocities
            self._update_enemy_velocities()
            # Update positions
            self._update_positions()
            # Enemy shooting
            self._handle_enemy_shooting()
            # Remove offscreen bullets
            self._remove_offscreen_bullets()
            # Check collisions
            collision_reward, dead = self._check_collisions()
            reward += collision_reward
            if dead:
                terminal = True
                break
            # Spawn enemies
            if self.frame_count % 60 == 0 and len(self.enemies) < 5:
                self._spawn_enemy()
            self.frame_count += 1
            # Render frame for obs
            self._render_frame()
            frame_obs = preprocess_frame(self.screen)
            self.recent_obs.append(frame_obs)
        
        # Penalize per env step
        reward -= 1.0
        # Ensure last render if terminated early
        self._render_frame()
        obs = np.stack(list(self.recent_obs), axis=-1)
        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True
        reward = np.clip(reward, -1.0, 1.0)
        info = {'score': self.score}
        return obs, reward, terminal, truncated, info

    def _update_cooldowns(self):
        self.agent_cooldown = max(0, self.agent_cooldown - 1)
        for enemy in self.enemies:
            enemy['cooldown'] = max(0, enemy['cooldown'] - 1)

    def _apply_action(self, action):
        if action == 1:  # up
            self.agent_vel = np.array([0.0, -self.agent_speed])
            self.agent_dir = np.array([0.0, -1.0])
        elif action == 2:  # down
            self.agent_vel = np.array([0.0, self.agent_speed])
            self.agent_dir = np.array([0.0, 1.0])
        elif action == 3:  # left
            self.agent_vel = np.array([-self.agent_speed, 0.0])
            self.agent_dir = np.array([-1.0, 0.0])
        elif action == 4:  # right
            self.agent_vel = np.array([self.agent_speed, 0.0])
            self.agent_dir = np.array([1.0, 0.0])
        else:  # no-op or shoot
            self.agent_vel = np.zeros(2)
        if action == 5 and self.agent_cooldown == 0:
            b_pos = self.agent_pos + self.tank_radius * self.agent_dir
            b_vel = self.agent_dir * self.bullet_speed
            self.agent_bullets.append({'pos': b_pos.copy(), 'vel': b_vel.copy()})
            self.agent_cooldown = 30

    def _update_enemy_velocities(self):
        for enemy in self.enemies:
            dir_to_agent = self.agent_pos - enemy['pos']
            dist = np.linalg.norm(dir_to_agent)
            if dist > 1e-2:
                unit_dir = dir_to_agent / dist
                enemy['vel'] = unit_dir * enemy['speed']
            else:
                enemy['vel'] = np.zeros(2)

    def _update_positions(self):
        self.agent_pos += self.agent_vel * self.dt
        self.agent_pos = np.clip(self.agent_pos, self.tank_radius, 400 - self.tank_radius)
        for enemy in self.enemies:
            enemy['pos'] += enemy['vel'] * self.dt
            enemy['pos'] = np.clip(enemy['pos'], self.tank_radius, 400 - self.tank_radius)
        for bullet in self.agent_bullets:
            bullet['pos'] += bullet['vel'] * self.dt
        for bullet in self.enemy_bullets:
            bullet['pos'] += bullet['vel'] * self.dt

    def _handle_enemy_shooting(self):
        for enemy in self.enemies:
            if enemy['cooldown'] == 0:
                vel_norm = np.linalg.norm(enemy['vel'])
                if vel_norm > 1e-6:
                    unit_dir = enemy['vel'] / vel_norm
                else:
                    unit_dir = np.array([1.0, 0.0])
                eb_pos = enemy['pos'] + self.tank_radius * unit_dir
                eb_vel = unit_dir * self.bullet_speed
                self.enemy_bullets.append({'pos': eb_pos.copy(), 'vel': eb_vel.copy()})
                enemy['cooldown'] = self.np_random.randint(90, 151)

    def _remove_offscreen_bullets(self):
        self.agent_bullets = [
            b for b in self.agent_bullets
            if 0 <= b['pos'][0] <= 400 and 0 <= b['pos'][1] <= 400
        ]
        self.enemy_bullets = [
            b for b in self.enemy_bullets
            if 0 <= b['pos'][0] <= 400 and 0 <= b['pos'][1] <= 400
        ]

    def _check_collisions(self):
        dead = False
        kill_reward = 0

        # Check agent death: enemy bullets
        for eb in self.enemy_bullets[:]:
            if np.linalg.norm(eb['pos'] - self.agent_pos) < self.tank_radius + self.bullet_radius:
                self.enemy_bullets.remove(eb)
                dead = True
                break
        if not dead:
            # Check agent-enemy collision
            for enemy in self.enemies[:]:
                if np.linalg.norm(enemy['pos'] - self.agent_pos) < 2 * self.tank_radius:
                    dead = True
                    break

        # Check agent bullets hit enemies
        i = 0
        while i < len(self.agent_bullets):
            ab = self.agent_bullets[i]
            j = 0
            while j < len(self.enemies):
                enemy = self.enemies[j]
                if np.linalg.norm(ab['pos'] - enemy['pos']) < self.tank_radius + self.bullet_radius:
                    del self.agent_bullets[i]
                    del self.enemies[j]
                    kill_reward += 20
                    self.score += 20
                    break
                j += 1
            else:
                i += 1

        death_reward = -5.0 if dead else 0.0
        return kill_reward + death_reward, dead

    def _spawn_enemy(self):
        side = self.np_random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            pos = np.array([self.np_random.uniform(15, 385), 15])
        elif side == 'bottom':
            pos = np.array([self.np_random.uniform(15, 385), 385])
        elif side == 'left':
            pos = np.array([15, self.np_random.uniform(15, 385)])
        else:  # right
            pos = np.array([385, self.np_random.uniform(15, 385)])
        speed = self.np_random.uniform(100, 200)
        cooldown = self.np_random.randint(60, 180)
        self.enemies.append({
            'pos': pos,
            'vel': np.zeros(2),
            'cooldown': cooldown,
            'speed': speed
        })

    def _render_frame(self):
        self.screen.fill((0, 0, 0))
        # Draw grid
        for coord in range(0, 401, 50):
            pygame.draw.line(self.screen, (255, 255, 255), (coord, 0), (coord, 400), 1)
            pygame.draw.line(self.screen, (255, 255, 255), (0, coord), (400, coord), 1)
        # Draw agent tank
        self._draw_tank(self.agent_pos, self.agent_dir, (0, 0, 255))
        # Draw enemy tanks
        for enemy in self.enemies:
            vel_norm = np.linalg.norm(enemy['vel'])
            if vel_norm > 1e-6:
                e_dir = enemy['vel'] / vel_norm
            else:
                e_dir = np.array([1.0, 0.0])
            self._draw_tank(enemy['pos'], e_dir, (255, 0, 0), inverted=True)
        # Draw bullets
        for bullet in self.agent_bullets:
            pos_int = bullet['pos'].astype(int)
            pygame.draw.circle(self.screen, (255, 255, 0), pos_int, self.bullet_radius)
        for bullet in self.enemy_bullets:
            pos_int = bullet['pos'].astype(int)
            pygame.draw.circle(self.screen, (255, 165, 0), pos_int, self.bullet_radius)
        # Draw info text
        font = pygame.font.Font(None, 24)
        fps = self.clock.get_fps()
        text = font.render(f"Score: {self.score} FPS: {fps:.1f}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))

    def _draw_tank(self, pos, dir_vec, color, inverted=False):
        dir_norm = dir_vec / np.linalg.norm(dir_vec)
        perp = np.array([-dir_norm[1], dir_norm[0]])
        if inverted:
            dir_norm = -dir_norm
        p1 = pos + 12 * dir_norm
        p2 = pos - 6 * dir_norm + 6 * perp
        p3 = pos - 6 * dir_norm - 6 * perp
        points = [p1.astype(int), p2.astype(int), p3.astype(int)]
        pygame.draw.polygon(self.screen, color, points)

    def render(self):
        if self.render_mode is None:
            return None
        self._render_frame()
        if self.render_mode == 'human':
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
        elif self.render_mode == 'rgb_array':
            arr = pygame.surfarray.array3d(self.screen)
            return np.transpose(arr, (1, 0, 2))

    def close(self):
        if self.screen is not None:
            if self.render_mode == 'human':
                pygame.display.quit()
            pygame.quit()
            self.screen = None