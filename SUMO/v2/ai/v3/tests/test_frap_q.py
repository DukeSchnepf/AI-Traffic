"""V3 FRAP-DQN unit tests (no SUMO; torch + numpy only)."""
import sys
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent.parent))   # SUMO/v2/ai

from v2.frap_encoder import FRAPEncoder  # noqa: E402

EMBED = 128
P_MAX = 6
M_MAX = 16


def _synth(seed=0):
    rng = np.random.default_rng(seed)
    B = 4
    mov = rng.uniform(0, 10, size=(B, M_MAX, 3)).astype(np.float32)
    pm = np.zeros((B, P_MAX, M_MAX), dtype=bool)
    phase = np.zeros((B, P_MAX), dtype=bool)
    for b in range(B):
        ng = int(rng.integers(2, P_MAX + 1))
        phase[b, :ng] = True
        for s in range(ng):
            srv = rng.random(M_MAX) > 0.5
            if not srv.any():
                srv[0] = True
            pm[b, s] = srv
    return (torch.from_numpy(mov), torch.from_numpy(pm),
            torch.from_numpy(phase))


def test_phase_embeddings_batched_shape():
    enc = FRAPEncoder(mov_feat_dim=3, embed_dim=EMBED)
    mov, pm, phase = _synth(1)
    pe = enc.phase_embeddings_batched(mov, pm)
    assert pe.shape == (mov.shape[0], P_MAX, EMBED), \
        f"phase embeds {pe.shape} != (B, {P_MAX}, {EMBED})"


if __name__ == "__main__":
    test_phase_embeddings_batched_shape()
    print("task1 OK")
