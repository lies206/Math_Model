from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from problem2_optimization import price_for_day
from problem4_optimization import (
    analyze_pv_residuals,
    load_problem4_data,
    problem2_strategy_metrics,
    problem4_input_files,
)
from problem2_optimization import Problem2YearResult
from problem3_optimization import LoadForecastResult, Problem3Data, run_strategy


class Problem4DataAndResidualTests(unittest.TestCase):
    def test_problem4_input_manifest_contains_only_attachments_2_3_4(self):
        root = Path(__file__).resolve().parents[1]
        names = tuple(path.name for path in problem4_input_files(root))
        self.assertEqual(names, ("附件2.xlsx", "附件3.xlsx", "附件4.xlsx"))
        data = load_problem4_data(root, print_structure=False)
        self.assertEqual(data.load_kw.shape, (365, 144))
        self.assertEqual(data.actual_pv_kw.shape, (365, 144))
        self.assertEqual(data.pv_forecast_hourly_kw.shape, (365, 4, 24))
        self.assertEqual(data.dynamic_price.shape, (365, 144))

    def test_residual_is_actual_minus_forecast_and_statistics_match(self):
        actual = np.array([[0.0, 10.0, 20.0, 30.0]])
        forecast = np.array([[0.0, 8.0, 24.0, 28.0]])
        analysis = analyze_pv_residuals({"0:00": (actual, forecast)}, max_acf_lag=2)
        residual = np.array([0.0, 2.0, -4.0, 2.0])
        np.testing.assert_allclose(analysis.residuals_kw["0:00"], residual)
        self.assertAlmostEqual(analysis.statistics["0:00"]["mae_kw"], 2.0)
        self.assertAlmostEqual(
            analysis.statistics["0:00"]["rmse_kw"],
            float(np.sqrt(np.mean(residual**2))),
        )
        self.assertAlmostEqual(
            analysis.statistics["0:00"]["residual_std_kw"],
            float(np.std(residual)),
        )


class DynamicPriceInterfaceTests(unittest.TestCase):
    def test_price_for_day_supports_static_and_annual_matrices(self):
        static = np.arange(144, dtype=float)
        annual = np.repeat(static[None, :], 3, axis=0)
        annual[2] += 1000.0
        np.testing.assert_allclose(price_for_day(static, 2, days=3), static)
        np.testing.assert_allclose(price_for_day(annual, 2, days=3), static + 1000.0)

    def test_price_for_day_rejects_misaligned_shape(self):
        with self.assertRaises(ValueError):
            price_for_day(np.ones((2, 143)), 0, days=2)

    def test_problem3_strategy_accepts_a_different_price_curve_each_day(self):
        days = 2
        load_kw = np.full((days, 144), 3000.0)
        pv_kw = np.zeros_like(load_kw)
        prices = np.vstack((np.ones(144), np.full(144, 2.0)))
        data = Problem3Data(
            prices,
            np.arange(np.datetime64("2025-01-01"), np.datetime64("2025-01-03")),
            load_kw,
            pv_kw,
            np.zeros((days, 4, 24)),
            ("附件2.xlsx", "附件3.xlsx", "附件4.xlsx"),
        )
        forecast = LoadForecastResult(
            load_kw.copy(),
            np.array(["test", "test"], dtype=object),
            np.zeros(days),
            np.array([-1, 0]),
        )
        result = run_strategy(
            "dynamic-test", data, forecast, (), False, progress=False
        )
        self.assertAlmostEqual(
            result.daily_plan_cost[1], 2.0 * result.daily_plan_cost[0], places=4
        )

    def test_problem2_metrics_are_repriced_with_each_days_dynamic_tariff(self):
        days = 32
        shape = (days, 144)
        grid = np.zeros(shape)
        emergency = np.zeros(shape)
        grid[31, 0] = 10.0
        emergency[31, 1] = 2.0
        zeros = np.zeros(shape)
        result = Problem2YearResult(
            grid,
            grid.copy(),
            zeros.copy(),
            zeros.copy(),
            zeros.copy(),
            zeros.copy(),
            zeros.copy(),
            emergency,
            np.full((days, 145), 6000.0),
            np.zeros(days),
            np.zeros(days),
            zeros.copy(),
            zeros.copy(),
            np.full(days, 0.8),
            np.arange(days) - 1,
            False,
            {},
        )
        dynamic_price = np.ones(shape)
        dynamic_price[31, 0] = 3.0
        dynamic_price[31, 1] = 4.0
        metrics = problem2_strategy_metrics(result, dynamic_price, start_day=31)
        self.assertAlmostEqual(metrics["plan_cost_yuan"], 30.0)
        self.assertAlmostEqual(metrics["emergency_cost_yuan"], 40.0)
        self.assertAlmostEqual(metrics["total_cost_yuan"], 70.0)


if __name__ == "__main__":
    unittest.main()
