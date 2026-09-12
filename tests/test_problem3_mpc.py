from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from problem3_optimization import (
    DispatchConfig,
    asymmetric_adjustment_cost,
    build_causal_load_forecasts,
    interpolate_pv_release,
    load_problem3_data,
    online_load_bias_correction,
    run_day_strategy,
    solve_mpc_schedule,
)


class Problem3CausalityTests(unittest.TestCase):
    def test_loader_whitelists_only_attachments_1_2_3(self):
        root = Path(__file__).resolve().parents[1]
        data = load_problem3_data(root, print_structure=False)
        self.assertEqual(data.source_files, ("附件1.xlsx", "附件2.xlsx", "附件3.xlsx"))
        self.assertEqual(data.load_kw.shape, (365, 144))
        self.assertEqual(data.pv_forecast_hourly_kw.shape, (365, 4, 24))

    def test_day_ahead_load_forecast_cannot_see_current_or_future_actuals(self):
        rng = np.random.default_rng(2026)
        actual = rng.uniform(1000.0, 5000.0, size=(40, 144))
        first = build_causal_load_forecasts(actual)
        changed = actual.copy()
        changed[30:] += 100000.0
        second = build_causal_load_forecasts(changed)
        np.testing.assert_allclose(first.forecast_kw[30], second.forecast_kw[30])
        self.assertLess(first.max_history_day_used[30], 30)

    def test_online_bias_uses_only_executed_same_day_slots(self):
        base = np.full(144, 100.0)
        actual = np.full(144, 110.0)
        corrected = online_load_bias_correction(base, actual, start_slot=36)
        changed_future = actual.copy()
        changed_future[36:] = 10000.0
        corrected_after_future_change = online_load_bias_correction(base, changed_future, start_slot=36)
        np.testing.assert_allclose(corrected[36:], corrected_after_future_change[36:])
        self.assertGreater(corrected[36:].mean(), base[36:].mean())


class Problem3AlignmentAndOptimizationTests(unittest.TestCase):
    def test_hourly_forecast_is_linearly_interpolated_from_issue_anchor(self):
        hourly = np.arange(1.0, 25.0) * 60.0
        ten_minute = interpolate_pv_release(hourly, release_hour=6, anchor_kw=0.0)
        self.assertEqual(ten_minute.shape, (108,))
        self.assertAlmostEqual(ten_minute[0], 10.0)
        self.assertAlmostEqual(ten_minute[5], 60.0)
        self.assertAlmostEqual(ten_minute[-1], 1080.0)

    def test_adjustment_cost_keeps_directions_separate(self):
        reference = np.array([100.0, 100.0])
        revised = np.array([80.0, 130.0])
        price = np.array([2.0, 2.0])
        result = asymmetric_adjustment_cost(reference, revised, price)
        self.assertAlmostEqual(result.downward_kwh, 20.0)
        self.assertAlmostEqual(result.upward_kwh, 30.0)
        self.assertAlmostEqual(result.downward_cost_yuan, 20.0)
        self.assertAlmostEqual(result.upward_cost_yuan, 90.0)
        self.assertAlmostEqual(result.total_cost_yuan, 110.0)

    def test_mpc_schedule_satisfies_balance_soc_power_and_mutex(self):
        config = DispatchConfig()
        load = np.full(18, 700.0)
        pv = np.linspace(0.0, 300.0, 18)
        price = np.r_[np.full(9, 0.5), np.full(9, 1.0)]
        plan = solve_mpc_schedule(load, pv, price, initial_soc=6000.0, terminal_soc=6000.0)
        self.assertTrue(plan.success, plan.message)
        np.testing.assert_allclose(plan.grid + plan.pv_used + plan.discharge, load + plan.charge, atol=1e-5)
        self.assertGreaterEqual(plan.grid.min(), -1e-7)
        self.assertGreaterEqual(plan.soc.min(), config.soc_min_kwh - 1e-5)
        self.assertLessEqual(plan.soc.max(), config.soc_max_kwh + 1e-5)
        self.assertLessEqual(plan.charge.max(), config.interval_limit_kwh + 1e-5)
        self.assertLessEqual(plan.discharge.max(), config.interval_limit_kwh + 1e-5)
        self.assertLessEqual(np.minimum(plan.charge, plan.discharge).max(), 1e-5)
        self.assertAlmostEqual(plan.soc[0], 6000.0, places=5)
        self.assertAlmostEqual(plan.soc[-1], 6000.0, places=5)

    def test_six_oclock_update_freezes_history_and_inherits_actual_soc(self):
        price = np.ones(144)
        base_load_kw = np.full(144, 3000.0)
        actual_load_kw = base_load_kw.copy()
        actual_load_kw[:36] += 1200.0
        actual_pv_kw = np.zeros(144)
        pv_releases_kw = np.zeros((4, 24))
        day = run_day_strategy(
            day_index=10,
            actual_load_kw=actual_load_kw,
            actual_pv_kw=actual_pv_kw,
            base_load_forecast_kw=base_load_kw,
            pv_releases_hourly_kw=pv_releases_kw,
            price=price,
            initial_soc=6000.0,
            update_hours=(6,),
            event_triggered=False,
        )
        self.assertEqual(len(day.update_records), 1)
        record = day.update_records[0]
        self.assertEqual(record.start_slot, 36)
        self.assertAlmostEqual(record.inherited_actual_soc_kwh, day.soc[36], places=5)
        np.testing.assert_allclose(day.adjusted_grid[:36], day.plan_grid[:36])

    def test_event_trigger_skips_adjustment_without_positive_net_saving(self):
        load_kw = np.full(144, 3000.0)
        day = run_day_strategy(
            day_index=10,
            actual_load_kw=load_kw,
            actual_pv_kw=np.zeros(144),
            base_load_forecast_kw=load_kw,
            pv_releases_hourly_kw=np.zeros((4, 24)),
            price=np.ones(144),
            initial_soc=6000.0,
            update_hours=(6,),
            event_triggered=True,
        )
        self.assertFalse(day.update_records[0].applied)
        self.assertEqual(day.adjustment_event_count, 0)
        self.assertAlmostEqual(day.adjustment_cost_yuan, 0.0)
        np.testing.assert_allclose(day.adjusted_grid, day.plan_grid)


if __name__ == "__main__":
    unittest.main()
