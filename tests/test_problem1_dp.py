import math
import unittest
from datetime import time

import numpy as np

from problem1_dp import (
    DispatchConfig,
    build_interval_labels,
    read_problem1_data,
    run_adaptive_dp,
    storage_bus_exchange,
    validate_dispatch,
)


class Problem1DPTests(unittest.TestCase):
    def test_storage_bus_exchange_applies_efficiency_in_correct_direction(self):
        x = np.array([90.0, -100.0, 0.0])
        exchange = storage_bus_exchange(x, efficiency=0.9)
        np.testing.assert_allclose(exchange, [100.0, -90.0, 0.0])

    def test_interval_labels_convert_right_endpoints_without_rotating_values(self):
        raw = [time(0, 10), time(0, 20)] + [f"{h}:{m:02d}" for h in range(24) for m in range(0, 60, 10)][3:]
        raw = raw[:143] + ["0:00+1"]
        labels = build_interval_labels(raw)
        self.assertEqual(len(labels), 144)
        self.assertEqual(labels[0], "00:00-00:10")
        self.assertEqual(labels[5], "00:50-01:00")
        self.assertEqual(labels[-1], "23:50-24:00")

    def test_two_period_arbitrage_respects_terminal_soc_and_efficiency(self):
        config = DispatchConfig(
            e_min=0.0,
            e_max=500.0,
            initial_energy=200.0,
            terminal_energy=200.0,
            power_max_kw=5000.0,
            efficiency=0.9,
            dt_hours=1.0,
            coarse_step_kwh=5.0,
            refinement_widths_kwh=(8.0, 1.0),
            refinement_steps_kwh=(0.1, 0.01),
        )
        solution = run_adaptive_dp(
            load_kw=np.array([100.0, 100.0]),
            pv_kw=np.zeros(2),
            price=np.array([1.0, 3.0]),
            config=config,
        )
        self.assertAlmostEqual(solution.soc_kwh[0], 200.0, places=8)
        self.assertAlmostEqual(solution.soc_kwh[-1], 200.0, places=8)
        self.assertAlmostEqual(solution.discharge_kwh[1], 100.0, delta=0.03)
        self.assertAlmostEqual(solution.charge_kwh[0], 100.0 / 0.9**2, delta=0.05)
        self.assertAlmostEqual(solution.grid_kwh[1], 0.0, delta=0.03)

    def test_attachment_one_is_read_as_144_right_endpoint_intervals(self):
        data = read_problem1_data("附件/附件1.xlsx")
        self.assertEqual(len(data["price"]), 144)
        self.assertEqual(data["intervals"][0], "00:00-00:10")
        self.assertEqual(data["intervals"][-1], "23:50-24:00")
        self.assertAlmostEqual(data["price"][0], 0.4248, places=8)
        self.assertAlmostEqual(data["load_kw"][0], 3439.8466, places=8)

    def test_validation_detects_power_and_balance_violations(self):
        config = DispatchConfig()
        report = validate_dispatch(
            load_kw=np.array([0.0]),
            pv_kw=np.array([0.0]),
            price=np.array([1.0]),
            soc_kwh=np.array([6000.0, 7000.0]),
            grid_kwh=np.array([0.0]),
            charge_kwh=np.array([1000.0]),
            discharge_kwh=np.array([0.0]),
            curtailment_kwh=np.array([0.0]),
            config=config,
        )
        self.assertFalse(report["charge_power_within_limit"])
        self.assertFalse(report["energy_balance_ok"])
        self.assertTrue(math.isfinite(report["max_balance_error_kwh"]))

    def test_validation_explicitly_rejects_reverse_sale_and_excess_curtailment(self):
        config = DispatchConfig(
            e_min=0.0,
            e_max=1000.0,
            initial_energy=200.0,
            terminal_energy=200.0,
            dt_hours=1.0,
        )
        report = validate_dispatch(
            load_kw=np.array([0.0]),
            pv_kw=np.array([100.0]),
            price=np.array([1.0]),
            soc_kwh=np.array([200.0, 200.0]),
            grid_kwh=np.array([-1.0]),
            charge_kwh=np.array([0.0]),
            discharge_kwh=np.array([0.0]),
            curtailment_kwh=np.array([101.0]),
            config=config,
        )
        self.assertFalse(report["no_reverse_sale"])
        self.assertFalse(report["curtailment_within_available_pv"])

    def test_dp_solution_satisfies_explicit_bus_balance_and_terminal_soc(self):
        config = DispatchConfig(
            e_min=0.0,
            e_max=1000.0,
            initial_energy=200.0,
            terminal_energy=200.0,
            power_max_kw=500.0,
            efficiency=0.9,
            dt_hours=1.0,
            coarse_step_kwh=5.0,
            refinement_widths_kwh=(8.0,),
            refinement_steps_kwh=(0.1,),
        )
        load_kw = np.array([0.0, 50.0, 200.0])
        pv_kw = np.array([300.0, 100.0, 0.0])
        solution = run_adaptive_dp(load_kw, pv_kw, np.array([1.0, 2.0, 3.0]), config)
        balance = (
            solution.grid_kwh
            + pv_kw * config.dt_hours
            - solution.curtailment_kwh
            + solution.discharge_kwh
            - load_kw * config.dt_hours
            - solution.charge_kwh
        )
        np.testing.assert_allclose(balance, 0.0, atol=1e-6)
        self.assertTrue(np.all(solution.grid_kwh >= -1e-8))
        self.assertTrue(np.all(solution.curtailment_kwh <= pv_kw * config.dt_hours + 1e-8))
        self.assertAlmostEqual(solution.soc_kwh[-1], solution.soc_kwh[0], places=6)


if __name__ == "__main__":
    unittest.main()
