import unittest

import numpy as np

from problem2_cost_optimized import (
    build_dynamic_quantile_candidates,
    build_fixed_quantile_candidates,
    quantiles_from_selected_names,
    run_full_cost_selection,
    selected_base_forecast,
    select_trailing_cost_winner,
)


class FullDispatchCostSelectionTests(unittest.TestCase):
    def test_selected_quantiles_are_parsed_from_dynamic_candidate_names(self):
        selected = np.array(["tau_0.65", "tau_0.80", "tau_0.95"], dtype=object)

        quantiles = quantiles_from_selected_names(selected)

        np.testing.assert_allclose(quantiles, np.array([0.65, 0.80, 0.95]))

    def test_selected_base_forecast_uses_each_days_selected_candidate(self):
        candidates = {
            "weekly_w7_g0.55": np.full((3, 144), 7.0),
            "weekly_w28_g0.95": np.full((3, 144), 28.0),
        }
        selected = np.array(
            ["weekly_w7_g0.55", "weekly_w28_g0.95", "weekly_w7_g0.55"],
            dtype=object,
        )

        base = selected_base_forecast(candidates, selected)

        np.testing.assert_allclose(base[:, 0], np.array([7.0, 28.0, 7.0]))

    def test_dynamic_quantile_candidates_remain_causal(self):
        rng = np.random.default_rng(2048)
        base = rng.normal(450.0, 20.0, size=(20, 144))
        actual = base + rng.normal(0.0, 30.0, size=(20, 144))
        original = build_dynamic_quantile_candidates(
            base,
            actual,
            tau_candidates=(0.65, 0.80, 0.95),
            window_days=28,
        )

        changed_actual = actual.copy()
        changed_actual[12:] += 9999.0
        changed = build_dynamic_quantile_candidates(
            base,
            changed_actual,
            tau_candidates=(0.65, 0.80, 0.95),
            window_days=28,
        )

        self.assertEqual(set(original), {"tau_0.65", "tau_0.80", "tau_0.95"})
        for name in original:
            np.testing.assert_allclose(original[name][12], changed[name][12])

    def test_dynamic_quantile_rejects_candidates_with_colliding_names(self):
        base = np.zeros((2, 144))

        with self.assertRaisesRegex(ValueError, "名称冲突"):
            build_dynamic_quantile_candidates(
                base,
                base,
                tau_candidates=(0.801, 0.804),
            )

    def test_fixed_quantile_candidates_remain_causal(self):
        rng = np.random.default_rng(2026)
        load = rng.normal(600.0, 30.0, size=(20, 144))
        pv = np.maximum(rng.normal(150.0, 20.0, size=(20, 144)), 0.0)
        price = np.linspace(0.4, 1.2, 144)
        original = build_fixed_quantile_candidates(load, pv, price, tau=0.8)

        changed_load = load.copy()
        changed_pv = pv.copy()
        changed_load[12:] += 9999.0
        changed_pv[12:] += 1000.0
        changed = build_fixed_quantile_candidates(changed_load, changed_pv, price, tau=0.8)

        self.assertEqual(set(original), set(changed))
        self.assertGreater(len(original), 1)
        for name in original:
            np.testing.assert_allclose(original[name][12], changed[name][12])

    def test_selector_uses_only_strictly_prior_realized_costs(self):
        costs = {
            "candidate_a": np.array([10.0, 10.0, 10.0, 9999.0, 9999.0]),
            "candidate_b": np.array([20.0, 20.0, 20.0, 0.0, 0.0]),
        }

        selected = select_trailing_cost_winner(
            costs,
            day=3,
            validation_days=3,
            min_history_days=2,
            fallback_name="candidate_b",
        )

        self.assertEqual(selected, "candidate_a")

    def test_selector_uses_configured_fallback_during_cold_start(self):
        costs = {
            "candidate_a": np.array([1.0, 1.0, 1.0]),
            "candidate_b": np.array([2.0, 2.0, 2.0]),
        }

        selected = select_trailing_cost_winner(
            costs,
            day=1,
            validation_days=28,
            min_history_days=7,
            fallback_name="candidate_b",
        )

        self.assertEqual(selected, "candidate_b")

    def test_selector_breaks_equal_cost_ties_by_candidate_name(self):
        costs = {
            "z_candidate": np.array([10.0, 20.0, 30.0]),
            "a_candidate": np.array([10.0, 20.0, 30.0]),
        }

        selected = select_trailing_cost_winner(
            costs,
            day=3,
            validation_days=3,
            min_history_days=1,
            fallback_name="z_candidate",
        )

        self.assertEqual(selected, "a_candidate")

    def test_selector_requires_historical_gain_before_leaving_reference(self):
        costs = {
            "tau_0.80": np.array([100.0, 100.0, 100.0]),
            "tau_0.85": np.array([99.5, 99.5, 99.5]),
        }

        selected = select_trailing_cost_winner(
            costs,
            day=3,
            validation_days=3,
            min_history_days=1,
            fallback_name="tau_0.80",
            reference_name="tau_0.80",
            min_relative_improvement=0.01,
        )

        self.assertEqual(selected, "tau_0.80")

    def test_online_selector_evaluates_candidates_from_the_same_daily_soc(self):
        days = 10
        actual_load = np.full((days, 144), 10.0)
        actual_pv = np.zeros_like(actual_load)
        candidates = {
            "accurate": actual_load.copy(),
            "under": np.zeros_like(actual_load),
        }

        result = run_full_cost_selection(
            candidates,
            actual_load,
            actual_pv,
            np.ones(144),
            validation_days=3,
            min_history_days=2,
            fallback_name="under",
            progress=False,
        )

        np.testing.assert_allclose(
            result.candidate_initial_soc_kwh["accurate"],
            result.candidate_initial_soc_kwh["under"],
        )
        # 冷启动时低计划会先消耗免费初始SOC，因此要等电池下探后，完整费用才会
        # 识别出持续低估导致的紧急购电代价；这正是完整调度评价区别于点预测损失之处。
        self.assertTrue(all(name == "under" for name in result.selected_names[:4]))
        self.assertTrue(all(name == "accurate" for name in result.selected_names[4:]))
        self.assertEqual(result.run["grid"].shape, (days, 144))
        self.assertEqual(result.run["soc"].shape, (days, 145))

    def test_online_selection_for_day_is_unchanged_when_that_day_actual_changes(self):
        days = 10
        actual_load = np.full((days, 144), 10.0)
        actual_pv = np.zeros_like(actual_load)
        candidates = {
            "accurate": actual_load.copy(),
            "under": np.zeros_like(actual_load),
        }
        original = run_full_cost_selection(
            candidates,
            actual_load,
            actual_pv,
            np.ones(144),
            validation_days=3,
            min_history_days=2,
            fallback_name="under",
            progress=False,
        )

        changed_load = actual_load.copy()
        changed_load[5:] += 1000.0
        changed = run_full_cost_selection(
            candidates,
            changed_load,
            actual_pv,
            np.ones(144),
            validation_days=3,
            min_history_days=2,
            fallback_name="under",
            progress=False,
        )

        self.assertEqual(original.selected_names[5], changed.selected_names[5])


if __name__ == "__main__":
    unittest.main()
