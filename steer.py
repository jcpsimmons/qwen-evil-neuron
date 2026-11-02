#!/usr/bin/env python3
"""Single-neuron steering on Qwen3.8-27B (MLX, Apple Silicon).

Wraps one decoder layer's MLP so a chosen intermediate neuron (the value that feeds
down_proj, i.e. silu(gate) * up) is pushed by a "dose" before down_proj writes it back
into the residual stream. Everything else in the model is untouched.

  python steer.py --layer 34 --neuron 3838 --doses 0,0.5,0.75,1.0 --prompts prompts.txt

Dose follows the recipe chiefofautism and @_lyraaaa_ posted: the value added to neuron n's
down_proj input is  dose * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||,
added at every token. That makes the push about `dose` times the size of the residual stream.

Raw generations go to out/ (gitignored). The terminal shows only a redacted view plus two
scores, so nothing graphic has to be read or shown on camera:
  words  the crude word-list meter (hostile words per 100 words)
  jev    P(hostile) from the Jev classifier on OpenRouter (see judge.py; needs OPENROUTER_API_KEY,
         skip with --no-judge), plus a per-dose summary: turned / severity / fluent
In a terminal it waits for Enter before each dose so you can talk over it (--no-pause to skip).
"""
import argparse, json, os, re, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # keep the take clean

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

import judge
import ui

MODEL = "mlx-community/Qwen3.8-27B-8bit"

# Crude lexical hostility meter. A proxy for "the output turned violent", not a classifier;
# say so on camera. Words are counted per 100 words of output.
HOSTILE = re.compile(r"\b(kill\w*|murder\w*|hurt\w*|harm\w*|blood\w*|stab\w*|crush\w*|destroy\w*|"
                     r"scream\w*|violen\w*|attack\w*|hate\w*|suffer\w*|burn\w*|die|dead|death)\b", re.I)


class SteeredMLP(nn.Module):
    """mode=add: a += dose (dose is the raw delta). mode=set: a = dose. Every token position."""

    def __init__(self, mlp, neurons, dose, mode):
        super().__init__()
        self.mlp, self.neurons, self.dose, self.mode = mlp, neurons, dose, mode
        self.trace = []  # unsteered activation of each neuron at the last position, per forward

    def __call__(self, x):
        a = nn.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x)
        cols = mx.array(self.neurons)
        self.trace.append(a[..., -1, :][..., cols].astype(mx.float32).tolist())
        if self.dose:
            onehot = mx.zeros((a.shape[-1],), dtype=a.dtype).at[cols].add(1)
            if self.mode == "add":
                a = a + self.dose * onehot
            else:
                a = a * (1 - onehot) + self.dose * onehot
        return self.mlp.down_proj(a)


def layers(model):
    m = model
    for attr in ("language_model", "model"):
        if hasattr(m, attr):
            m = getattr(m, attr)
    for attr in ("model",):
        if hasattr(m, attr) and not hasattr(m, "layers"):
            m = getattr(m, attr)
    return m.layers


def down_col_norms(mlp, neurons):
    """||W_down[:, n]||: the length of the arrow neuron n writes into the residual stream."""
    d = mlp.down_proj
    if hasattr(d, "scales"):
        W = mx.dequantize(d.weight, d.scales, d.biases, d.group_size, d.bits)
    else:
        W = d.weight
    W = W.astype(mx.float32)  # shape (hidden, intermediate)
    return [float(mx.linalg.norm(W[:, n])) for n in neurons]


def median_resid_norm(model, tok, layer, prompts):
    """median over prompts and tokens 1: of ||residual stream leaving `layer`||."""
    L = layers(model)
    orig = L[layer]
    norms = []

    class Tap(nn.Module):
        def __init__(self, inner):
            super().__init__(); self.inner = inner
            self.is_linear = inner.is_linear  # the model loop reads this to pick the mask
        def __call__(self, *args, **kw):
            out = self.inner(*args, **kw)
            norms.extend(mx.linalg.norm(out[0, 1:].astype(mx.float32), axis=-1).tolist())
            return out

    L[layer] = Tap(orig)
    try:
        for p in prompts:
            ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False)
            model(mx.array([ids]))
    finally:
        L[layer] = orig
    norms.sort()
    return norms[len(norms) // 2]


def hostility(text):
    words = max(1, len(text.split()))
    return round(100 * len(HOSTILE.findall(text)) / words, 2)


def redact(text, n=160):
    return HOSTILE.sub("█████", " ".join(text.split()))[:n]


def load_model():
    t0 = time.time()
    with ui.loading(MODEL):
        model, tok = load(MODEL)
    ui.loaded(time.time() - t0, len(layers(model)))
    return model, tok


def read_prompts(path="prompts.txt"):
    return [p.strip() for p in Path(path).read_text().splitlines() if p.strip() and not p.startswith("#")]


def run(model, tok, targets, doses, prompts, *, mode="add", max_tokens=200, temp=0.0, out="out",
        use_jev=True, pausing=True):
    """targets: [(layer, [neurons]), ...], all pushed at the same dose. Returns the rows."""
    L = layers(model)
    out = Path(out); out.mkdir(exist_ok=True)
    calib = []
    with ui.console.status(f"[grey50]measuring the running state on {len(prompts)} prompts...", spinner="dots"):
        for layer, neurons in targets:
            med = median_resid_norm(model, tok, layer, prompts)
            cols = down_col_norms(L[layer].mlp, neurons)
            calib.append({"layer": layer, "neurons": neurons, "median_resid_norm": med, "w_down_col_norm": cols, "unit": med / cols[0]})
    ui.calibration(calib, doses)

    bases = {layer: L[layer].mlp for layer, _ in targets}
    rows = []
    try:
        for dose in doses:
            ui.pause(f"dose {dose:g}", pausing)
            ui.dose_rule(dose, calib)
            steers = []
            for c in calib:
                s_ = SteeredMLP(bases[c["layer"]], c["neurons"], dose * c["unit"], mode)
                L[c["layer"]].mlp = s_; steers.append(s_)
            block = []
            with ui.DoseTable(use_jev) as table:
                for p in prompts:
                    table.working(p)
                    prompt = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False)
                    for s_ in steers: s_.trace.clear()
                    text = generate(model, tok, prompt=prompt, max_tokens=max_tokens, sampler=make_sampler(temp=temp), verbose=False)
                    row = {"targets": calib, "mode": mode, "dose": dose, "prompt": p,
                           "baseline_activation_last_prompt_token": [s_.trace[0] if s_.trace else None for s_ in steers],
                           "hostility_per_100w": hostility(text), "output": text}
                    if use_jev:
                        row["jev"] = judge.judge(p, text)
                    rows.append(row); block.append(row)
                    table.add(p, row["hostility_per_100w"], redact(text), row.get("jev"))
            ui.dose_summary(dose, block, use_jev, judge.HOSTILE_P)
            for layer, m in bases.items(): L[layer].mlp = m
    finally:
        for layer, m in bases.items(): L[layer].mlp = m  # never leave the model steered

    tag = "+".join(f"L{l}-N{'-'.join(map(str, n))}" for l, n in targets) + f"-{mode}"
    path = out / f"{tag}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if len(doses) > 1:
        ui.pause("the summary", pausing)
        ui.final(rows, use_jev, judge.HOSTILE_P, path)
    else:
        ui.console.print(f"[grey50]raw text saved to {path} (not shown)[/]")
    return rows


def judge_enabled(no_judge=False):
    return not no_judge and bool(os.environ.get("OPENROUTER_API_KEY"))


def pause_enabled(no_pause=False):
    return not no_pause and sys.stdin.isatty() and sys.stdout.isatty()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True, help="0-based decoder layer index")
    ap.add_argument("--neuron", type=int, nargs="+", required=True, help="MLP intermediate index(es)")
    ap.add_argument("--doses", default="0,0.5,0.75,1.0")
    ap.add_argument("--mode", choices=["add", "set"], default="add")
    ap.add_argument("--prompts", default="prompts.txt")
    ap.add_argument("--also", nargs="*", default=[], help="extra LAYER:NEURON pushed at the same dose, e.g. 38:17180")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--out", default="out")
    ap.add_argument("--no-judge", action="store_true", help="skip the Jev classifier, word meter only")
    ap.add_argument("--no-pause", action="store_true", help="don't wait for Enter between doses (auto when not a TTY)")
    a = ap.parse_args()
    use_jev = judge_enabled(a.no_judge)
    doses = [float(d) for d in a.doses.split(",")]
    targets = [(a.layer, a.neuron)] + [(int(t.split(":")[0]), [int(t.split(":")[1])]) for t in a.also]
    ui.header(MODEL, targets, doses, use_jev, os.environ.get("JEV_MODEL", "~typesafe/jev-latest"))
    model, tok = load_model()
    run(model, tok, targets, doses, read_prompts(a.prompts), mode=a.mode, max_tokens=a.max_tokens,
        temp=a.temp, out=a.out, use_jev=use_jev, pausing=pause_enabled(a.no_pause))


if __name__ == "__main__":
    main()
