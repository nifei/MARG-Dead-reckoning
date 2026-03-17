from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from PDR.gnss_constraint import apply_gnss_velocity_constraints


class TestGnssConstraint(unittest.TestCase):
    def test_gnss_gap_interval_is_flagged_and_no_extreme_scale(self) -> None:
        t = np.arange(0.0, 10.0, 0.5)
        vel = np.zeros((len(t), 3), dtype=float)
        vel[:, 0] = 1.0

        gnss = pd.DataFrame(
            {
                "unix_sec": [0.0, 2.0, 8.0, 9.5],
                "east_m": [0.0, 2.0, 8.0, 9.5],
                "north_m": [0.0, 0.0, 0.0, 0.0],
                "acc_fix_m": [5.0, 5.0, 5.0, 5.0],
                "speed_xyz_mps": [1.0, 1.0, 1.0, 1.0],
            }
        )

        out_df, interval_df, _ = apply_gnss_velocity_constraints(
            t_sec_abs=t,
            vel_enu=vel,
            gnss_fix_df=gnss,
            labels={"motion_context": np.ones(len(t), dtype=bool)},
        )

        self.assertGreaterEqual(int(interval_df["gnss_gap_flag"].sum()), 1)
        gap_rows = interval_df[interval_df["gnss_gap_flag"] == 1]
        self.assertTrue(np.allclose(gap_rows["scale"].to_numpy(dtype=float), 1.0))
        self.assertEqual(int(out_df["gnss_gap_flag"].max()), 1)


if __name__ == "__main__":
    unittest.main()
