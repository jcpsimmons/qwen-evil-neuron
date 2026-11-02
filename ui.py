"""Terminal presentation for steer.py: built to be filmed.

Nothing raw is printed. Previews are redacted by the caller before they get here.
"""
import sys

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

console = Console(highlight=False)

RED, AMBER, GREEN, DIM = "#ff4d4d", "#ffb020", "#3ddc84", "grey50"


def _heat(p):
    return RED if p > 0.5 else AMBER if p > 0.2 else GREEN


def bar(value, top=1.0, width=10):
    """A filled bar colored by how far along it is."""
    frac = max(0.0, min(1.0, value / top))
    full = round(frac * width)
    return Text("█" * full, style=_heat(frac)) + Text("░" * (width - full), style=DIM)


def header(model, targets, doses, judge_on, judge_model):
    t = Text()
    t.append("SINGLE-NEURON STEERING\n", style="bold white")
    t.append("model   ", style=DIM); t.append(f"{model}\n")
    t.append("push    ", style=DIM)
    t.append("  +  ".join(f"L{l} N{','.join(map(str, n))}" for l, n in targets) + "\n", style=f"bold {RED}")
    t.append("doses   ", style=DIM); t.append(",  ".join(f"{d:g}" for d in doses) + "\n")
    t.append("scoring ", style=DIM)
    t.append("word counter", style="white")
    if judge_on:
        t.append("  +  ", style=DIM); t.append(f"Jev classifier ({judge_model})", style="bold cyan")
    else:
        t.append("  (Jev off: set OPENROUTER_API_KEY)", style=AMBER)
    console.print(Panel(t, box=box.HEAVY, border_style=RED, padding=(1, 3)))


def loading(model):
    return console.status(f"[{DIM}]loading {model} into memory...", spinner="dots")


def loaded(secs, n_layers):
    console.print(f"[{GREEN}]✓[/] loaded in [bold]{secs:.0f}s[/]  [{DIM}]{n_layers} layers, nothing changed yet[/]\n")


def calibration(calib, doses):
    t = Table(box=box.SIMPLE_HEAVY, title="calibration: how big is one dose?", title_style="bold",
              title_justify="left", header_style=DIM)
    t.add_column("layer"); t.add_column("neuron")
    t.add_column("median ‖resid‖", justify="right"); t.add_column("‖W_down[:, n]‖", justify="right")
    t.add_column("added per 1.0 dose", justify="right")
    for d in doses:
        if d:
            t.add_column(f"push @ {d:g}", justify="right", style=f"bold {RED}")
    for c in calib:
        t.add_row(f"L{c['layer']}", f"N{','.join(map(str, c['neurons']))}", f"{c['median_resid_norm']:.1f}",
                  f"{c['w_down_col_norm'][0]:.3f}", f"{c['unit']:.1f}",
                  *[f"+{d * c['unit']:.1f}" for d in doses if d])
    console.print(t)


def pause(label, enabled):
    if not enabled:
        return
    console.print(Rule(style=DIM))
    try:
        console.input(f"  [{DIM}]press[/] [bold reverse] ENTER [/] [{DIM}]to run[/] [bold]{label}[/] ")
    except (EOFError, KeyboardInterrupt):
        console.print(); sys.exit(0)
    sys.stdout.write("\x1b[1A\x1b[2K"); sys.stdout.flush()  # erase the prompt line before the take


def dose_rule(dose, calib):
    push = "  ".join(f"L{c['layer']} N{c['neurons'][0]} +{dose * c['unit']:.1f}" for c in calib) if dose else "nothing added"
    title = "DOSE 0  ·  the normal model" if dose == 0 else f"DOSE {dose:g}  ·  {push}"
    console.print(Rule(Text(title, style=f"bold {'white' if dose == 0 else RED}"), style=RED if dose else DIM))


class DoseTable:
    """A table that fills in row by row, with a spinner on the prompt being generated."""

    def __init__(self, judge_on):
        self.judge_on, self.rows, self.current = judge_on, [], None
        self.live = Live(self._render(), console=console, refresh_per_second=12, transient=False)

    def _render(self):
        t = Table(box=box.SIMPLE, header_style=DIM, expand=True, pad_edge=False)
        t.add_column("prompt", style="white", no_wrap=True, max_width=34)
        t.add_column("words", justify="right", width=5)
        if self.judge_on:
            t.add_column("jev P(hostile)", width=16, no_wrap=True)
            t.add_column("severity", width=14, no_wrap=True)
            t.add_column("fluent", width=6)
        t.add_column("answer (redacted)", style=DIM, ratio=1, overflow="fold")
        for r in self.rows:
            cells = [r["prompt"], f"{r['words']:.2f}"]
            if self.judge_on:
                j = r["jev"]
                cells += [bar(j["hostile"]) + Text(f" {j['hostile']:.2f}", style=f"bold {_heat(j['hostile'])}"),
                          bar(j["severity"], top=3, width=6) + Text(f" {j['severity']:.1f}/3"),
                          Text("yes", style=GREEN) if j["coherent"] > 0.5 else Text("loops", style=AMBER)]
            cells.append(r["preview"])
            t.add_row(*cells)
        if self.current:
            return Group(t, Spinner("dots", text=Text(f" generating: {self.current}", style=DIM)))
        return t

    def __enter__(self):
        self.live.__enter__(); return self

    def __exit__(self, *a):
        self.current = None; self.live.update(self._render()); self.live.__exit__(*a)

    def working(self, prompt):
        self.current = prompt; self.live.update(self._render())

    def add(self, prompt, words, preview, jev=None):
        self.rows.append({"prompt": prompt, "words": words, "preview": preview, "jev": jev})
        self.current = None; self.live.update(self._render())


def dose_summary(dose, rows, judge_on, hostile_p):
    n = len(rows)
    words = sum(r["hostility_per_100w"] for r in rows) / n
    t = Text("  ")
    t.append(f"dose {dose:g}  ", style="bold")
    t.append(f"words {words:.2f}", style="white")
    if judge_on:
        js = [r["jev"] for r in rows]
        turned = sum(j["hostile"] > hostile_p for j in js)
        t.append("   │   ", style=DIM)
        t.append(f"JEV turned {turned}/{n}", style=f"bold {RED if turned else GREEN}")
        t.append(f"   mean P(hostile) {sum(j['hostile'] for j in js) / n:.2f}")
        t.append(f"   severity {sum(j['severity'] for j in js) / n:.2f}/3")
        fl = sum(j["coherent"] > 0.5 for j in js)
        t.append(f"   fluent {fl}/{n}", style=GREEN if fl == n else AMBER)
    console.print(Panel(t, box=box.ROUNDED, border_style=RED if judge_on and turned else DIM))
    console.print()


def final(rows, judge_on, hostile_p, path):
    t = Table(box=box.HEAVY_HEAD, title="all doses", title_style="bold", title_justify="left", header_style=DIM)
    t.add_column("dose", justify="right", style="bold")
    t.add_column("words / 100", justify="right")
    if judge_on:
        for c in ("JEV turned", "mean P(hostile)", "severity /3", "fluent"):
            t.add_column(c, justify="right")
    by = {}
    for r in rows:
        by.setdefault(r["dose"], []).append(r)
    for dose, rs in by.items():
        n = len(rs)
        cells = [f"{dose:g}", f"{sum(r['hostility_per_100w'] for r in rs) / n:.2f}"]
        if judge_on:
            js = [r["jev"] for r in rs]
            turned = sum(j["hostile"] > hostile_p for j in js)
            p = sum(j["hostile"] for j in js) / n
            cells += [Text(f"{turned}/{n}", style=f"bold {RED if turned else GREEN}"),
                      Text(f"{p:.2f}", style=_heat(p)), f"{sum(j['severity'] for j in js) / n:.2f}",
                      f"{sum(j['coherent'] > 0.5 for j in js)}/{n}"]
        t.add_row(*cells)
    console.print(t)
    console.print(f"[{DIM}]raw text saved to {path} (not shown)[/]")


def step(title, targets, doses):
    console.print()
    t = Text(title, style="bold white")
    t.append("\n" + "  +  ".join(f"L{l} N{','.join(map(str, n))}" for l, n in targets), style=f"bold {RED}")
    t.append("    doses " + ", ".join(f"{d:g}" for d in doses), style=DIM)
    console.print(Panel(t, box=box.DOUBLE, border_style="white", padding=(0, 2)))
