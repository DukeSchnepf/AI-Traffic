"""F2/F3 regression: the FRAP movement state must (a) carry the current-
phase green bit + normalized time-in-phase, and (b) be normalized to [0,1].

Before the fix, get_state_frap() emitted only (halting, vehicles, waiting)
as raw counts: the Q-net could not tell "hold green" (free) from "switch"
(costs a yellow + penalty), and the raw waiting term swamped the shared
movement MLP by ~2 orders of magnitude. This test needs SUMO (it reads a
live light's state), so it lives next to the torch-only V3 smoke tests but
is the integration check for the 5-feature FRAP state.
"""
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent.parent))   # SUMO/v2/ai

from sumo_env import SumoTrafficEnv  # noqa: E402

SUMO_CFG = "sim_calibrated.sumocfg"
TLS_ID = "3153556582"   # NE 8th St corridor; 4 green slots + 4 yellows


def test_frap_state_has_phase_features_and_is_normalized():
    env = SumoTrafficEnv(sumo_cfg_file=SUMO_CFG, tls_id=TLS_ID,
                         time_limit=1200, reward_mode="combined")
    try:
        env.reset()
        feats = env.get_state_frap()["movement_features"]

        # T0.2: five feature columns.
        assert feats.shape[1] == 5, \
            f"expected 5 movement feats, got {feats.shape[1]}"

        # T0.2: col 3 (green bit) must equal exactly the movements green in
        # the active phase -- cross-checked against SUMO's own live state.
        live = env._conn().trafficlight.getRedYellowGreenState(TLS_ID)
        green_bit = feats[:, 3]
        for i in range(feats.shape[0]):
            want = 1.0 if (i < len(live) and live[i] in ("G", "g")) else 0.0
            assert green_bit[i] == want, (
                f"movement {i}: green bit {green_bit[i]} != {want} "
                f"(live char {live[i] if i < len(live) else '-'})")
        assert green_bit.max() == 1.0, "no movement green just after reset"

        # col 4 (time-in-phase) is shared across movements and in [0,1].
        assert np.allclose(feats[:, 4], feats[0, 4]), \
            "time-in-phase must be identical across movement rows"
        assert 0.0 <= float(feats[0, 4]) <= 1.0

        # T0.3: every feature normalized to [0,1] after a few steps of
        # accumulating queues.
        for _ in range(10):
            env.step(0)
        feats2 = env.get_state_frap()["movement_features"]
        assert feats2.min() >= 0.0, f"feature min {feats2.min()} < 0"
        assert feats2.max() <= 1.0, f"feature max {feats2.max()} > 1"
    finally:
        env.stop_simulation()


if __name__ == "__main__":
    test_frap_state_has_phase_features_and_is_normalized()
    print("test_frap_state OK")
