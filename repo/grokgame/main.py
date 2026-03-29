import pygame
import sys
import math
import random
import numpy as np

def reflect_velocity(vel: tuple[float, float], normal: tuple[float, float]) -> tuple[float, float]:
    """Computes realistic velocity reflection off a surface normal."""
    vx, vy = vel
    dot = vx * normal[0] + vy * normal[1]
    return (vx - 2 * dot * normal[0], vy - 2 * dot * normal[1])

class Particle:
    def __init__(self, pos: list[float, float], vel: list[float, float], life: int, max_life: int, color: tuple[int, int, int]):
        self.pos = pos
        self.vel = vel
        self.life = life
        self.max_life = max_life
        self.color = color

    def update(self) -> int:
        self.vel[1] += 0.15  # gravity
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.life -= 1
        return self.life

class ParticleSystem:
    def __init__(self):
        self.particles: list[Particle] = []

    def add_burst(self, x: int, y: int, count: int = 12):
        for _ in range(count):
            color = (random.randint(127, 255), random.randint(127, 255), random.randint(127, 255))
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 6)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            life = random.randint(20, 40)
            px = x + random.uniform(-5, 5)
            py = y + random.uniform(-5, 5)
            self.particles.append(Particle([px, py], [vx, vy], life, life, color))

    def update(self):
        self.particles = [p for p in self.particles if p.update() >= 0]

    def draw(self, screen: pygame.Surface, offset: tuple[float, float] = (0.0, 0.0)):
        for p in self.particles:
            alpha = p.life / p.max_life
            col = tuple(int(c * alpha) for c in p.color)
            r = int(1 + 3 * alpha)
            pos = (int(p.pos[0] + offset[0]), int(p.pos[1] + offset[1]))
            pygame.draw.circle(screen, col, pos, r)

class Snake:
    def __init__(self, color: tuple[int, int, int], x: int, initial_length: int, screen_height: int):
        self.color = color
        self.x = x
        self.segments: list[tuple[int, int]] = []
        self.target_y = 0
        self.max_speed = 4
        self.length = initial_length
        self._grow_pending = False
        self.reset(screen_height)

    def reset(self, screen_height: int):
        self.length = 5
        self._grow_pending = False
        self.target_y = screen_height // 2
        center_y = screen_height // 2
        spacing = 22
        start_y = center_y - (self.length - 1) * spacing // 2
        self.segments = [(self.x, int(start_y + i * spacing)) for i in range(self.length)]

    def grow_snake(self):
        self._grow_pending = True

    def head_pos(self) -> tuple[int, int]:
        return self.segments[-1]

    def head_rect(self) -> pygame.Rect:
        hx, hy = self.head_pos()
        return pygame.Rect(hx - 10, hy - 10, 20, 20)

    def update(self, ball, screen_height: int):
        bx, by = ball.pos
        bvx, bvy = ball.vel
        if abs(bvx) > 0.1:
            t = (self.x - bx) / bvx
            pred_y = by + t * bvy
        else:
            pred_y = by
        pred_y = max(20, min(screen_height - 20, pred_y))
        self.target_y = int(pred_y + random.uniform(-10, 10))

        curr_y = self.segments[-1][1]
        dy = self.target_y - curr_y
        move_dist = min(abs(dy), self.max_speed)
        if dy > 0:
            new_y = curr_y + move_dist
        else:
            new_y = curr_y - move_dist
        new_y = max(20, min(screen_height - 20, new_y))

        self.segments.append((self.x, int(new_y)))
        if not self._grow_pending:
            self.segments.pop(0)
        self._grow_pending = False
        self.length = len(self.segments)

class Ball:
    def __init__(self, x: float, y: float, radius: int, max_speed: float):
        self.pos = [x, y]
        self.vel = [5.0, 0.0]
        self.radius = radius
        self.max_speed = max_speed

    def update(self, screen_height: int) -> bool:
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        hit_wall = False
        if self.pos[1] <= self.radius:
            self.pos[1] = self.radius
            self.vel[1] = -self.vel[1]
            hit_wall = True
        elif self.pos[1] >= screen_height - self.radius:
            self.pos[1] = screen_height - self.radius
            self.vel[1] = -self.vel[1]
            hit_wall = True
        return hit_wall

    def collides_with_rect(self, rect: pygame.Rect) -> bool:
        cx = max(rect.left, min(self.pos[0], rect.right))
        cy = max(rect.top, min(self.pos[1], rect.bottom))
        dx = self.pos[0] - cx
        dy = self.pos[1] - cy
        return dx * dx + dy * dy <= self.radius * self.radius

    def reflect(self, normal: tuple[float, float]):
        self.vel = list(reflect_velocity(tuple(self.vel), normal))

    def accelerate(self):
        speed = math.hypot(self.vel[0], self.vel[1])
        if speed > 0:
            new_speed = min(speed + 0.2, self.max_speed)
            scale = new_speed / speed
            self.vel[0] *= scale
            self.vel[1] *= scale

class Game:
    def __init__(self, screen_width: int = 800, screen_height: int = 600):
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Snake Pong AI")
        self.clock = pygame.time.Clock()
        self.screen_width = screen_width
        self.screen_height = screen_height
        cyan = (0, 255, 255)
        lime = (0, 255, 65)
        self.left_snake = Snake(cyan, 80, 5, screen_height)
        self.right_snake = Snake(lime, 720, 5, screen_height)
        self.ball = Ball(screen_width / 2, screen_height / 2, 15, 12.0)
        self.particles = ParticleSystem()
        self.font = pygame.font.SysFont('arial', 36, bold=True)
        self.small_font = pygame.font.SysFont('arial', 24)
        self.phase = 0.0
        self.shake_frames = 0
        self.shake_offset = [0.0, 0.0]
        self.bg_surf = None
        self.reset()

    def update_bg(self):
        w, h = self.screen_width, self.screen_height
        self.bg_surf = pygame.Surface((w, h))
        cx = w // 2 + 30 * math.sin(self.phase)
        cy = h // 2 + 20 * math.cos(self.phase * 1.3)
        rows, cols = np.ogrid[:h, :w]
        dx = cols - cx
        dy = rows - cy
        dists = np.sqrt(dx**2 + dy**2)
        max_dist = math.hypot(w / 2, h / 2)
        dist = np.clip(dists / max_dist, 0.0, 1.0)
        edge_color = np.array([44, 24, 16])
        center_color = np.array([0, 212, 255])
        rgb = center_color * (1 - dist[:,:,np.newaxis]) + edge_color * dist[:,:,np.newaxis]
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        arr = pygame.surfarray.pixels3d(self.bg_surf)
        arr[:] = rgb.transpose(1, 0, 2)

    def screen_shake(self, frames: int):
        self.shake_frames = frames

    def reset(self):
        self.left_snake.reset(self.screen_height)
        self.right_snake.reset(self.screen_height)
        self.ball.pos = [self.screen_width / 2.0, self.screen_height / 2.0]
        self.ball.vel = [5.0, random.uniform(-3, 3)]
        self.particles.particles.clear()
        self.shake_frames = 0
        self.shake_offset = [0.0, 0.0]

    def update(self):
        self.phase += 0.05
        self.update_bg()

        self.left_snake.update(self.ball, self.screen_height)
        self.right_snake.update(self.ball, self.screen_height)

        wall_hit = self.ball.update(self.screen_height)
        if wall_hit:
            self.particles.add_burst(int(self.ball.pos[0]), int(self.ball.pos[1]), 12)

        left_rect = self.left_snake.head_rect()
        right_rect = self.right_snake.head_rect()
        hit_left = self.ball.collides_with_rect(left_rect)
        hit_right = self.ball.collides_with_rect(right_rect)

        if hit_left or hit_right:
            normal = (1.0, 0.0) if hit_left else (-1.0, 0.0)
            self.ball.reflect(normal)
            self.ball.accelerate()
            snake = self.left_snake if hit_left else self.right_snake
            snake.grow_snake()
            hx, hy = snake.head_pos()
            self.particles.add_burst(hx, hy, 15)
            self.screen_shake(8)

        # Ball trail
        self.particles.add_burst(int(self.ball.pos[0]), int(self.ball.pos[1]), 1)

        # Miss check
        r = self.ball.radius
        if self.ball.pos[0] + r < self.left_snake.x - 10 or self.ball.pos[0] - r > self.right_snake.x + 10:
            self.reset()

        self.particles.update()

        if self.shake_frames > 0:
            self.shake_frames -= 1
            self.shake_offset[0] = random.uniform(-2, 2)
            self.shake_offset[1] = random.uniform(-1.5, 1.5)
        else:
            self.shake_offset = [0.0, 0.0]

    def draw_snake(self, snake: Snake, ox: float, oy: float):
        ticks = pygame.time.get_ticks()
        pulse = math.sin(ticks * 0.01) * 0.3 + 0.7
        n_segs = len(snake.segments)
        for i, (sx, sy) in enumerate(snake.segments):
            ix = int(sx + ox)
            iy = int(sy + oy)
            alpha = (i + 1) / n_segs
            col_intensity = alpha * pulse
            glow_col = tuple(int(min(255, c * col_intensity * 0.8)) for c in snake.color)
            body_col = tuple(int(c * (0.4 + 0.6 * col_intensity)) for c in snake.color)

            # Glow outline
            glow_rect = pygame.Rect(ix - 15, iy - 15, 30, 30)
            pygame.draw.rect(self.screen, glow_col, glow_rect, border_radius=12, width=3)

            # Main body
            body_rect = pygame.Rect(ix - 10, iy - 10, 20, 20)
            pygame.draw.rect(self.screen, body_col, body_rect, border_radius=10)

            # Inner highlight
            inner_rect = pygame.Rect(ix - 6, iy - 6, 12, 12)
            pygame.draw.rect(self.screen, glow_col, inner_rect, border_radius=6)

    def draw(self):
        ox, oy = self.shake_offset
        self.screen.blit(self.bg_surf, (0, 0))

        self.particles.draw(self.screen, (ox, oy))

        # Ball glow and core
        bpx = int(self.ball.pos[0] + ox)
        bpy = int(self.ball.pos[1] + oy)
        # Outer glow layers
        for rad in range(self.ball.radius, self.ball.radius // 2, -2):
            gcol = (255, 255, 255)
            glow_surf = pygame.Surface((rad * 2, rad * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*gcol, 30), (rad, rad), rad)
            self.screen.blit(glow_surf, (bpx - rad, bpy - rad))
        # Yellow core
        pygame.draw.circle(self.screen, (255, 255, 0), (bpx, bpy), self.ball.radius // 2)
        # White ball
        pygame.draw.circle(self.screen, (255, 255, 255), (bpx, bpy), self.ball.radius)

        self.draw_snake(self.left_snake, ox, oy)
        self.draw_snake(self.right_snake, ox, oy)

        # UI Scores
        p1_len = self.left_snake.length
        p2_len = self.right_snake.length
        text = self.font.render(f"P1: {p1_len} | P2: {p2_len}", True, (255, 255, 255))
        shadow = self.font.render(f"P1: {p1_len} | P2: {p2_len}", True, (0, 0, 0))
        tw = text.get_width()
        self.screen.blit(shadow, (self.screen_width // 2 - tw // 2 + 2, 12))
        self.screen.blit(text, (self.screen_width // 2 - tw // 2, 10))

        # FPS
        fps_text = self.small_font.render(str(int(self.clock.get_fps())), True, (255, 255, 255))
        fps_shadow = self.small_font.render(str(int(self.clock.get_fps())), True, (0, 0, 0))
        fw = fps_text.get_width()
        self.screen.blit(fps_shadow, (self.screen_width // 2 - fw // 2 + 1, self.screen_height - 29))
        self.screen.blit(fps_text, (self.screen_width // 2 - fw // 2, self.screen_height - 30))

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True

def main() -> None:
    pygame.init()
    game = Game(800, 600)
    running = True
    while running:
        running = game.handle_events()
        if running:
            game.update()
            game.draw()
            pygame.display.flip()
            game.clock.tick(60)
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()