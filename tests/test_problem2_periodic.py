import unittest
from pathlib import Path

import numpy as np

from problem2_optimization import (
    allowed_input_files,
    analyze_periodicity,
    build_forecast_suite,
    causal_residual_quantile_forecast,
    cost_sensitive_loss,
    select_lowest_cost_model,
)


class Problem2PeriodicForecastTests(unittest.TestCase):
    def test_input_manifest_allows_only_attachments_one_and_two(self):
        root = Path(__file__).resolve().parents[1]
        paths = allowed_input_files(root)
        self.assertEqual(
            {path.name for path in paths},
            {"附件1.xlsx", "附件2.xlsx"},
        )
        self.assertFalse(any(path.name in {"附件3.xlsx", "附件4.xlsx"} for path in paths))

    def test_cost_sensitive_loss_penalizes_underforecast_four_times(self):
        actual = np.array([[10.0, 10.0]])
        forecast = np.array([[8.0, 12.0]])
        price = np.array([2.0, 3.0])
        # 2*4*2 + 3*1*2 = 22
        self.assertAlmostEqual(cost_sensitive_loss(actual, forecast, price), 22.0)

    def test_periodicity_analysis_recovers_weekly_pattern(self):
        days = 70
        intraday = np.sin(np.linspace(0, 2 * np.pi, 144, endpoint=False))
        weekday = np.array([0.0, 3.0, -2.0, 4.0, -1.0, 2.0, -3.0])
        load = np.vstack([100.0 + intraday + weekday[d % 7] for d in range(days)])
        pv = np.zeros_like(load)
        analysis = analyze_periodicity(load, pv, dt_hours=1.0 / 6.0)
        self.assertGreater(analysis["load"]["lag_day_correlations"]["7"], 0.99)
        self.assertGreater(
            analysis["load"]["lag_day_correlations"]["7"],
            analysis["load"]["lag_day_correlations"]["1"],
        )

    def test_quantile_calibration_uses_only_prior_residuals(self):
        base = np.full((8, 144), 100.0)
        actual = base.copy()
        actual[:4] += np.array([0.0, 0.0, 0.0, 20.0])[:, None]
        forecast = causal_residual_quantile_forecast(base, actual, tau=0.8, window_days=7)
        self.assertGreater(forecast[4, 0], 100.0)
        changed = actual.copy()
        changed[4:] += 9999.0
        changed_forecast = causal_residual_quantile_forecast(base, changed, tau=0.8, window_days=7)
        np.testing.assert_allclose(forecast[4], changed_forecast[4])

    def test_complete_forecast_suite_has_no_current_or_future_leakage(self):
        rng = np.random.default_rng(2026)
        load = rng.normal(600.0, 50.0, size=(40, 144))
        pv = np.maximum(rng.normal(150.0, 40.0, size=(40, 144)), 0.0)
        changed_load = load.copy()
        changed_pv = pv.copy()
        changed_load[25:] += 10000.0
        changed_pv[25:] += 2000.0
        price = np.linspace(0.4, 1.2, 144)

        original = build_forecast_suite(load, pv, price)
        changed = build_forecast_suite(changed_load, changed_pv, price)
        self.assertEqual(set(original.forecasts), set(changed.forecasts))
        for name in original.forecasts:
            np.testing.assert_allclose(
                original.forecasts[name][25], changed.forecasts[name][25]
            )
        self.assertTrue(np.all(original.max_history_day_used < np.arange(40)))
        self.assertFalse(original.leakage_detected)

    def test_final_model_selection_uses_actual_total_cost(self):
        comparison = {
            "low_rmse": {"total_cost_yuan": 120.0, "forecast_rmse_kwh_per_interval": 1.0},
            "low_cost": {"total_cost_yuan": 90.0, "forecast_rmse_kwh_per_interval": 3.0},
        }
        self.assertEqual(select_lowest_cost_model(comparison), "low_cost")


if __name__ == "__main__":
    unittest.main()
