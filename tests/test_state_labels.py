from __future__ import annotations

import unittest

import numpy as np

from PDR.state_labels import derive_motion_labels


class TestStateLabels(unittest.TestCase):
    def test_state_labels_continuity_on_stable_segment(self) -> None:
        n = 200
        freq_hz = 100.0
        raw_is_moving = np.ones(n, dtype=bool)
        vel_ned = np.zeros((n, 3), dtype=float)
        phase = np.linspace(0.0, 6.0 * np.pi, n)
        vel_ned[:, 0] = 0.9 + 0.08 * np.sin(phase)
        acc_earth = np.zeros((n, 3), dtype=float)
        acc_earth[:, 2] = 9.80665
        gyr_radps = np.zeros((n, 3), dtype=float)

        labels = derive_motion_labels(
            raw_is_moving=raw_is_moving,
            vel_ned=vel_ned,
            acc_earth=acc_earth,
            gyr_radps=gyr_radps,
            freq_hz=freq_hz,
        )

        motion_context = labels["motion_context"]
        self.assertEqual(motion_context.dtype, bool)
        self.assertGreater(float(motion_context.mean()), 0.95)
        self.assertGreater(int(labels["step_event"].sum()), 0)


if __name__ == "__main__":
    unittest.main()
