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
from typing import Optional, Any

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text


# ──────────────────────────────────────────────
# Dashboard class
# ──────────────────────────────────────────────
class TrainingDashboard:
    """Live TUI dashboard for the air hockey training loop."""

    LOG_CAPACITY = 15  # lines visible in the log panel

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
        self._log_lines.append(f"[steel_blue1]{ts}[/]  {message}")
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
        subtitle = Text(f"   ⏱  {elapsed_str}", style="white")
        return Panel(Align.center(title + subtitle), style="bold blue", padding=(0, 1))

    def _middle_row(self) -> Table:
        """Three-column row: overview | agent 1 | agent 2."""
        grid = Table.grid(expand=True, padding=(0, 1))
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(
            self._overview_panel(),
            self._agent_panel(1, "cyan"),
            self._agent_panel(2, "yellow"),
        )
        return grid

    # ── Overview panel ─────────────────────────

    def _overview_panel(self) -> Panel:
        s = self._stats

        tbl = Table.grid(padding=(0, 2))
        tbl.add_column(style="bright_white", justify="right", min_width=9)
        tbl.add_column(style="bold white", no_wrap=True)

        def row(label: str, value: str) -> None:
            tbl.add_row(label, value)

        def section(title: str) -> None:
            tbl.add_row(f"[bold blue]── {title} ──[/]", "")

        # Episode dynamics
        section("Episode")
        steps = s.get("steps", 0)
        avg_steps = s.get("avg_steps", 0.0)
        row("Steps", f"[bold white]{steps}[/]  [white]avg {avg_steps:.0f}[/]")
        row("Memory", f"[white]{s.get('mem1', 0):,}[/]")
        row("Epsilon", _epsilon_bar(s.get("epsilon1", 1.0)))

        ties = s.get("ties", 0)
        ep = max(s.get("episode", 1), 1)
        tr = ties / ep
        row("Ties", f"[white]{ties}  ({tr:.1%})[/]")

        tbl.add_row()  # spacer

        # Hyperparams (static)
        section("Config")
        row("LR", f"[white]{s.get('lr', '—')}[/]")
        row("γ  Gamma", f"[white]{s.get('gamma', '—')}[/]")
        row("Batch", f"[white]{s.get('batch_size', '—')}[/]")

        return Panel(tbl, title="[bold white]⚙  Overview[/]", border_style="blue", padding=(0, 1))

    # ── Agent panel ────────────────────────────

    def _agent_panel(self, agent_num: int, color: str) -> Panel:
        s = self._stats
        n = str(agent_num)

        # ─── Reward + Loss (big number table) ───
        metrics_tbl = Table.grid(padding=(0, 3))
        metrics_tbl.add_column(justify="right", style="bright_white", min_width=7)
        metrics_tbl.add_column(justify="right", no_wrap=True)  # current (big)
        metrics_tbl.add_column(justify="left", style="white", no_wrap=True)    # avg

        # Header row
        metrics_tbl.add_row("", f"[{color}]Current[/]", "[white]Avg (100)[/]")

        reward  = s.get(f"reward{n}", 0.0)
        avg_r   = s.get(f"avg_reward{n}", 0.0)
        loss    = s.get(f"loss{n}", 0.0)
        avg_l   = s.get(f"avg_loss{n}", 0.0)
        best    = s.get(f"best_reward{n}", 0.0)
        best_ep = s.get(f"best_ep{n}", 0)

        # Reward (right-padded by 3 spaces to align .2f decimal with .5f Loss)
        metrics_tbl.add_row(
            "Reward",
            f"[bold {color}]{reward:>+9.2f}   [/]",
            f"[white]{avg_r:>+8.2f}   [/]",
        )

        # Loss (no padding needed, has 5 decimals)
        metrics_tbl.add_row(
            "Loss",
            f"[{color}]{loss:>9.5f}[/]",
            f"[white]{avg_l:>8.5f}[/]",
        )

        # Best reward (right-padded by 3 spaces to align)
        metrics_tbl.add_row(
            "Best",
            f"[bold {color}]{best:>+9.2f}   [/]",
            f"[white]ep {best_ep:,}   [/]",
        )

        # ─── Win rate bar ───
        wins = s.get(f"wins{n}", 0)
        ep   = max(s.get("episode", 1), 1)
        wr   = wins / ep
        wins_tbl = Table.grid(padding=(0, 2))
        wins_tbl.add_column(style="bright_white", justify="right", min_width=7)
        wins_tbl.add_column(no_wrap=True)
        wins_tbl.add_row(
            "Wins",
            f"[bold {color}]{wins:>5,}[/]  {_rate_bar(wr, 18, color)}  [bold {color}]{wr:.1%}[/]",
        )

        # ─── Reward breakdown ───
        bd = s.get(f"avg_breakdown{n}", {})
        breakdown_tbl = _format_breakdown(bd, color)

        # Stack everything vertically inside agent panel
        outer = Table.grid(expand=True)
        outer.add_column()
        outer.add_row(metrics_tbl)
        outer.add_row(wins_tbl)
        outer.add_row(_dim_rule())
        outer.add_row(breakdown_tbl)

        return Panel(
            outer,
            title=f"[bold {color}]  Agent {agent_num}  [/]",
            border_style=color,
            padding=(0, 1),
        )

    def _log_panel(self) -> Panel:
        lines = list(self._log_lines)
        if not lines:
            lines = ["[white]Waiting for events…[/white]"]
        text = Text.from_markup("\n".join(lines))
        return Panel(text, title="[bold white]📋  Event Log[/]", border_style="blue", padding=(0, 1))


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _dim_rule() -> Text:
    """A thin dim separator line."""
    return Text("─" * 40, style="blue")


def _epsilon_bar(eps: float, width: int = 14) -> str:
    filled = round(eps * width)
    bar = "█" * filled + "░" * (width - filled)
    color = "bright_red" if eps > 0.5 else ("yellow" if eps > 0.2 else "bright_green")
    return f"[{color}]{bar}[/] [white]{eps:.4f}[/]"


def _format_breakdown(bd: dict, color: str) -> Any:
    """Two-column breakdown table, positive values bright, negative dimmed."""
    if not bd:
        return Text("—", style="white")

    KEYS = [
        ("goal",             "Goal "),
        ("hit_puck",         "Hit  "),
        ("puck_dir_bonus",   "Dir  "),
        ("defense_bonus",    "Def  "),
        ("concede",          "Cncd "),
        ("step_penalty",     "Step "),
        ("boundary_penalty", "Bndy "),
    ]

    tbl = Table.grid(padding=(0, 3))
    tbl.add_column(style="bright_white", justify="right", min_width=5)  # abbr
    tbl.add_column(justify="right", min_width=7, no_wrap=True)       # value col 1
    tbl.add_column(style="bright_white", justify="right", min_width=5)  # abbr
    tbl.add_column(justify="right", min_width=7, no_wrap=True)       # value col 2

    pairs = []
    for k, abbr in KEYS:
        v = bd.get(k, 0.0)
        val_color = color if v >= 0 else "red"
        pairs.append((abbr.strip(), f"[{val_color}]{v:>+6.2f}[/]"))

    # Render in rows of 2
    for i in range(0, len(pairs), 2):
        a1, v1 = pairs[i]
        if i + 1 < len(pairs):
            a2, v2 = pairs[i + 1]
        else:
            a2, v2 = "", ""
        tbl.add_row(a1, v1, a2, v2)

    return tbl


def _rate_bar(rate: float, width: int = 18, color: str = "white") -> str:
    filled = round(rate * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{color}]{bar}[/]"
