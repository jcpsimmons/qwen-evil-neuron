# qwen-evil-neuron

Single-neuron steering on Qwen3.8-27B (8-bit MLX, Apple Silicon). It reproduces the claim that pushing
layer 34, neuron 3838 by a dose of 0.75 turns the assistant hostile. It also runs a random-neuron
control and the two-neuron stack.

Every answer is scored two ways:

- **words**: a crude word counter (hostile words per 100 words). It's easy to fool, which is the point of showing it.
- **jev**: [Jev](https://openrouter.ai/docs/guides/community/jev.md), TypeSafe's decision model on OpenRouter.
  It returns P(hostile), a 0-3 severity score, and P(fluent) for each prompt and reply.

Raw generations are written to `out/` (gitignored). The terminal only shows a redacted preview.

## Run

```bash
export OPENROUTER_API_KEY=sk-or-...        # optional; without it you get the word counter only
uv run demo.py                              # the full run, waits for Enter before each step and dose
```

`demo.py` loads the model once (about 29 GB) and then runs three steps:

1. L34 N3838 at doses 0, 0.5, 0.75, 1.0
2. Control: L34 N9001 at doses 0, 0.75, 1.0
3. Stack: L34 N3838 + L38 N17180 at 0.75

Flags: `--step N`, `--no-pause`, `--no-judge`, `--max-tokens N`, `--out DIR`.

You can also run a single experiment, or rescore saved runs without reloading Qwen:

```bash
uv run steer.py --layer 34 --neuron 3838 --doses 0,0.5,0.75,1.0
uv run judge.py out/*.jsonl                 # writes out/*.jev.jsonl
```

Environment: `OPENROUTER_API_KEY` (required for Jev), `JEV_MODEL` (default `~typesafe/jev-latest`),
and `OPENROUTER_BASE` (default `https://openrouter.ai/api`).

## Dose

The value added to neuron n's down_proj input at every token is
`dose * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||`, so a dose of 0.75 is a push
three quarters as long as the residual stream. This is the recipe from
[chiefofautism](https://x.com/chiefofautism/status/2115274388425900291), which is based on
[_lyraaaa_'s neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer).

Results describe this setup only: 8-bit weights, greedy decoding, 200 tokens, thinking off, six prompts.
