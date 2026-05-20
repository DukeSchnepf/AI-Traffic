# Smart Traffic Lights — Run the AI

One Double-DQN agent per traffic light, all 12 corridor lights driven from a single SUMO process. The trained model lives in `ai/runs/coordinated/`, so you can evaluate or watch the live demo without retraining.

> Run everything from `SUMO/v2/`.

## 1. Setup (one time)

```bash
pip install -r requirements.txt
# SUMO must be on PATH, or set SUMO_HOME.
python -c "import torch, numpy, traci, sumolib, websockets; print('ok')"
```

## 2. Run

```bash
# Live demo — AI drives all 12 lights; then open frontend/index.html.
python run_websocket_ai.py

# Evaluate the trained model on 5 seeds vs SUMO's native actuated baseline:
python ai/eval_network.py --episodes 5 --time-limit 1200

# (Optional) Reproduce the committed model from scratch:
python ai/train_multi_dqn.py --episodes 60 --time-limit 1200
```

## Current result vs SUMO native-actuated

5 seeds × 1200 s, whole 12-light corridor:

| Metric | DQN | Native | Improvement |
|---|---|---|---|
| Throughput (vehicles arrived) | 1699.8 ±86 | 1711.8 ±22 | **−0.7%** (tie — wins 3/5 seeds) |
| Wait per vehicle (s) | 9,670 ±808 | 10,401 ±507 | **+7.0% lower** (wins 4/5 seeds) |

**Net: AI matches native throughput while cutting average network delay ~7%, stably.**

Regression anchor (same recipe on the target intersection alone, single-TLS): **+9.4% lower wait, +7.4% higher throughput** vs native.

## Training provenance

This result comes from **one 60-episode training run** (`ai/train_multi_dqn.py`, defaults). Additional runs were done during development for ablations and to test richer coordination shaping; that machinery is disabled by default because it was found to degrade results (see `ai/train_multi_dqn.py` docstring).

## Estimated ceiling with more training

Under the current design (independent per-light learners + local max-pressure reward), gains taper quickly. A realistic projection from doubling training (~120 episodes) plus multi-seed averaging:

- Throughput: **~parity to +1–2%** vs native (architecturally near the ceiling).
- Wait per vehicle: **~+9–11% lower** vs native.

Throughput is *bottlenecked* at native parity by the local-only credit structure — meaningfully beating native on throughput requires a different per-agent credit-assignment design, not more compute.
