#!/usr/bin/env python3
"""
Training Dashboard
==================
A live TUI dashboard for the Air Hockey RL training loop.
Uses the `rich` library's Live display to show real-time training metrics.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
from rich.style import Style
from rich.table import Table
from rich.text import Text


# ──────────────────────────────────────────────
# Sparkline helper
# ──────────────────────────────────────────────
_SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 30) -> str:
    """Return a Unicode sparkline string for the last `width` values."""
    if not values:
        return " " * width
    window = list(values)[-width:]
    lo, hi = min(window), max(window)
    span = hi - lo if hi != lo else 1.0
    chars = [_SPARK_CHARS[round((v - lo) / span * (len(_SPARK_CHARS) - 1))] for v in window]
    return "".join(chars).ljust(width)


# ──────────────────────────────────────────────
# Dashboard class
# ──────────────────────────────────────────────
class TrainingDashboard:
    """Live TUI dashboard for the air hockey training loop."""

    LOG_CAPACITY = 10  # lines visible in the log panel

    def __init__(self, num_episodes: int) -> None:
        self.num_episodes = num_episodes
        self._start_time: float = time.time()
        self._log_lines: deque[str] = deque(maxlen=self.LOG_CAPACITY)
        self._console = Console()
        self._live: Optional[Live] = None

        # Latest stats snapshot
        self._stats: dict = {}

        # Progress bar (episode counter)
        self._progress = Progress(
            SpinnerColumn(style="bold cyan"),
            TextColumn("[bold cyan]Episode[/] [progress.percentage]{task.percentage:>5.1f}%"),
            BarColumn(bar_width=None, style="cyan", complete_style="bright_cyan"),
            TextColumn("[cyan]{task.completed}[/]/[white]{task.total}[/]"),
            TimeElapsedColumn(),
            expand=True,
        )
        self._prog_task: TaskID = self._progress.add_task("training", total=num_episodes)

    # ── lifecycle ──────────────────────────────

    def start(self) -> "TrainingDashboard":
        self._start_time = time.time()
        self._live = Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=4,
            screen=False,
        )
        self._live.start()
        return self

    def stop(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    def __enter__(self) -> "TrainingDashboard":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ── public API ─────────────────────────────

    def update(self, stats: dict) -> None:
        """Called every episode with the latest training stats."""
        self._stats = stats
        ep = stats.get("episode", 0)

        self._progress.update(self._prog_task, completed=ep)

        if self._live:
            self._live.update(self._build_layout())

    def log(self, message: str) -> None:
        """Push a message into the scrolling event log."""
        ts = time.strftime("%H:%M:%S")
        self._log_lines.append(f"[dim]{ts}[/dim]  {message}")
        if self._live:
            self._live.update(self._build_layout())

    # ── layout builders ────────────────────────

    def _build_layout(self) -> Group:
        """Assemble the full TUI layout as a renderable Group."""
        return Group(
            self._header_panel(),
            self._progress,
            self._middle_row(),
            self._log_panel(),
        )

    def _header_panel(self) -> Panel:
        elapsed = time.time() - self._start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"

        title = Text("🏒  Air Hockey RL Training", style="bold bright_white")
        subtitle = Text(f"  ⏱  {elapsed_str}", style="dim white")
        header = Align.center(title + subtitle)
        return Panel(header, style="bold blue", padding=(0, 1))

    def _middle_row(self) -> Table:
        """Two-column row: episode stats left, win rates right."""
        grid = Table.grid(expand=True, padding=(0, 1))
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(
            self._episode_panel(),
            self._winrate_panel(),
        )
        return grid

    def _episode_panel(self) -> Panel:
        s = self._stats
        ep = s.get("episode", 0)
        total = self.num_episodes

        tbl = Table.grid(padding=(0, 2))
        tbl.add_column(style="dim white", justify="right", min_width=18)
        tbl.add_column(style="bold white")

        def row(label: str, value: str) -> None:
            tbl.add_row(label, value)

        row("Episode", f"[bright_cyan]{ep}[/] / {total}")
        row("Steps", f"{s.get('steps', 0)}  [dim](avg {s.get('avg_steps', 0.0):.1f})[/]")
        row("Episode Winner", _winner_text(s.get("winner", "—")))
        row("Memory", f"{s.get('mem1', 0):,}")
        row("Epsilon", _epsilon_bar(s.get("epsilon1", 1.0)))

        tbl.add_row()  # spacer

        # Agent 1
        tbl.add_row(
            "[bold magenta]── Agent 1 ──[/]", ""
        )
        row("  Reward", f"[magenta]{s.get('reward1', 0.0):+.2f}[/]  "
                        f"[dim](avg {s.get('avg_reward1', 0.0):+.2f})[/]")
        row("  Loss", f"[magenta]{s.get('loss1', 0.0):.6f}[/]  "
                        f"[dim](avg {s.get('avg_loss1', 0.0):.6f})[/]")

        tbl.add_row()  # spacer

        # Agent 2
        tbl.add_row(
            "[bold green]── Agent 2 ──[/]", ""
        )
        row("  Reward", f"[green]{s.get('reward2', 0.0):+.2f}[/]  "
                        f"[dim](avg {s.get('avg_reward2', 0.0):+.2f})[/]")
        row("  Loss", f"[green]{s.get('loss2', 0.0):.6f}[/]  "
                        f"[dim](avg {s.get('avg_loss2', 0.0):.6f})[/]")

        return Panel(tbl, title="[bold white]📊 Episode Stats[/]", border_style="blue", padding=(0, 1))

    def _winrate_panel(self) -> Panel:
        s = self._stats
        ep = max(s.get("episode", 1), 1)
        wins1 = s.get("wins1", 0)
        wins2 = s.get("wins2", 0)
        ties = s.get("ties", 0)
        wr1 = wins1 / ep
        wr2 = wins2 / ep
        tr = ties / ep

        tbl = Table.grid(padding=(0, 2))
        tbl.add_column(style="dim white", justify="right", min_width=14)
        tbl.add_column(style="bold white")

        def row(label, value):
            tbl.add_row(label, value)

        row("Total Episodes", f"[white]{ep}[/]")
        tbl.add_row()

        # Win rate bars
        bar_width = 24
        row("[bold magenta]Agent 1 Wins[/]",
            f"[magenta]{wins1:4d}[/]  {_rate_bar(wr1, bar_width, 'magenta')}  [magenta]{wr1:.1%}[/]")
        row("[bold green]Agent 2 Wins[/]",
            f"[green]{wins2:4d}[/]  {_rate_bar(wr2, bar_width, 'green')}  [green]{wr2:.1%}[/]")
        row("[dim white]Ties[/]",
            f"[white]{ties:4d}[/]  {_rate_bar(tr, bar_width, 'white')}  [white]{tr:.1%}[/]")

        tbl.add_row()

        # Hyperparams reminder
        row("Learning Rate", f"{s.get('lr', '—')}")
        row("Gamma", f"{s.get('gamma', '—')}")
        row("Batch Size", f"{s.get('batch_size', '—')}")
        row("Target Upd Freq", f"{s.get('target_update_freq', '—')}")

        return Panel(tbl, title="[bold white]🏆 Win Rates & Config[/]", border_style="blue", padding=(0, 1))

    def _log_panel(self) -> Panel:
        lines = list(self._log_lines)
        if not lines:
            lines = ["[dim]Waiting for events…[/dim]"]
        text = Text.from_markup("\n".join(lines))
        return Panel(text, title="[bold white]📋 Event Log[/]", border_style="dim blue", padding=(0, 1))


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _winner_text(winner: str) -> str:
    mapping = {
        "Agent1": "[bold magenta]Agent 1 🏆[/]",
        "Agent2": "[bold green]Agent 2 🏆[/]",
        "Tie": "[dim white]Tie[/]",
    }
    return mapping.get(winner, f"[dim]{winner}[/]")


def _epsilon_bar(eps: float, width: int = 16) -> str:
    filled = round(eps * width)
    bar = "█" * filled + "░" * (width - filled)
    color = "bright_red" if eps > 0.5 else ("yellow" if eps > 0.2 else "bright_green")
    return f"[{color}]{bar}[/] [dim]{eps:.4f}[/]"


def _rate_bar(rate: float, width: int = 20, color: str = "white") -> str:
    filled = round(rate * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{color}]{bar}[/]"
