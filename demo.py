#!/usr/bin/env python3
"""The whole on-camera run in one command: load Qwen once, then three steps.

  uv run demo.py

  Step 1  L34 N3838 at doses 0, 0.5, 0.75, 1.0   (chiefofautism's neuron)
  Step 2  L34 N9001 at doses 0, 0.75, 1.0         (control: random neuron, same layer, same size push)
  Step 3  L34 N3838 + L38 N17180 at dose 0.75     (his "extra evil" stack)

It waits for Enter before every step and every dose so you can talk over the output.
Needs OPENROUTER_API_KEY for the Jev classifier; without it, the word counter only.
Flags: --no-pause, --no-judge, --step N (run one step only), --max-tokens N, --out DIR.
"""
import argparse, os

import steer, ui

STEPS = [
    ("STEP 1  ·  the claimed evil neuron", [(34, [3838])], [0, 0.5, 0.75, 1.0]),
    ("STEP 2  ·  the control: a random neuron, pushed just as hard", [(34, [9001])], [0, 0.75, 1.0]),
    ("STEP 3  ·  the stack: two neurons at once", [(34, [3838]), (38, [17180])], [0.75]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pause", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--step", type=int, choices=[1, 2, 3], help="run only this step")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--out", default="out", help="where raw text goes (gitignored); use another dir for test runs")
    a = ap.parse_args()
    use_jev, pausing = steer.judge_enabled(a.no_judge), steer.pause_enabled(a.no_pause)
    steps = [STEPS[a.step - 1]] if a.step else STEPS
    ui.header(steer.MODEL, steps[0][1], steps[0][2], use_jev, os.environ.get("JEV_MODEL", "~typesafe/jev-latest"))
    model, tok = steer.load_model()
    prompts = steer.read_prompts()
    for i, (title, targets, doses) in enumerate(steps):
        if i:
            ui.pause(title.split("·")[0].strip().lower(), pausing)
        ui.step(title, targets, doses)
        steer.run(model, tok, targets, doses, prompts, use_jev=use_jev, pausing=pausing, max_tokens=a.max_tokens, out=a.out)
    ui.console.rule("[grey50]done")


if __name__ == "__main__":
    main()
