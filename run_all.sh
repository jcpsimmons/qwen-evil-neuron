#!/bin/zsh
# Full local reproduction. Logs to out/run_all.log; raw generations in out/*.jsonl (gitignored).
set -e
cd "${0:A:h}"
PY=.venv/bin/python
{
  echo "== L34 N3838 (chiefofautism's neuron)"; $PY steer.py --layer 34 --neuron 3838 --doses 0,0.5,0.75,1.0
  echo "== control: random neuron in the same layer (L34 N9001, chosen before running)"; $PY steer.py --layer 34 --neuron 9001 --doses 0,0.75,1.0
  echo "== stacked: L34 N3838 + L38 N17180"; $PY steer.py --layer 34 --neuron 3838 --also 38:17180 --doses 0.75
} 2>&1 | tee out/run_all.log
