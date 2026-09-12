"""问题二的费用驱动滚动参数选择与动态分位数优化。

本模块只复用 :mod:`problem2_optimization` 已有的物理调度内核。新增策略在每天
作出选择时只能读取当天以前已经结算的候选费用，避免把当天或未来实际数据泄漏
到日前决策中。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from microgrid_optimization import DispatchConfig
from problem2_optimization import (
    ForecastSuite,
    Problem2YearResult,
    build_forecast_suite,
    calculate_problem2_cost,
    causal_residual_quantile_forecast,
    optimize_day_ahead,
    price_for_day,
    simulate_realtime,
)


DEFAULT_WINDOW_DECAY_FALLBACK = "weekly_w7_g0.55"
DEFAULT_TAU_CANDIDATES = (0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
DEFAULT_TAU_FALLBACK = "tau_0.80"
DEFAULT_DYNAMIC_MIN_RELATIVE_IMPROVEMENT = 0.05


@dataclass(frozen=True)
class FullCostSelectionResult:
    """在线费用选择及其最终物理运行结果。"""

    run: dict[str, np.ndarray]
    forecast: np.ndarray
    selected_names: np.ndarray
    candidate_daily_costs: dict[str, np.ndarray]
    candidate_daily_normal_costs: dict[str, np.ndarray]
    candidate_daily_emergency_costs: dict[str, np.ndarray]
    candidate_initial_soc_kwh: dict[str, np.ndarray]


@dataclass(frozen=True)
class Problem2CostOptimizedStudy:
    """原方案、固定分位费用选择和动态分位数三阶段结果。"""

    forecast_suite: ForecastSuite
    baseline_result: Problem2YearResult
    fixed_selection: FullCostSelectionResult
    fixed_result: Problem2YearResult
    dynamic_selection: FullCostSelectionResult
    dynamic_result: Problem2YearResult
    selected_base_forecast: np.ndarray


def selected_base_forecast(
    candidate_forecasts: Mapping[str, np.ndarray],
    selected_names: np.ndarray,
) -> np.ndarray:
    """按每天第一阶段选中的名称拼接原始点预测。"""

    if not candidate_forecasts:
        raise ValueError("基础预测候选不能为空")
    names = np.asarray(selected_names, dtype=object)
    if names.ndim != 1:
        raise ValueError("每日候选名称必须是一维数组")
    forecasts = {
        name: np.asarray(values, dtype=float)
        for name, values in candidate_forecasts.items()
    }
    shapes = {values.shape for values in forecasts.values()}
    if len(shapes) != 1:
        raise ValueError("基础预测候选必须同形")
    shape = next(iter(shapes))
    if len(shape) != 2 or shape != (len(names), 144):
        raise ValueError("基础预测候选必须是与每日名称对应的days×144矩阵")
    unknown = sorted({str(name) for name in names if name not in forecasts})
    if unknown:
        raise ValueError(f"每日选择包含未知候选: {unknown}")
    base = np.empty(shape, dtype=float)
    for day, name in enumerate(names):
        base[day] = forecasts[str(name)][day]
    if not np.all(np.isfinite(base)):
        raise ValueError("拼接后的基础预测必须全部有限")
    return base


def build_dynamic_quantile_candidates(
    base_forecast: np.ndarray,
    actual_net_kwh: np.ndarray,
    *,
    tau_candidates: tuple[float, ...] = (0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    window_days: int = 28,
) -> dict[str, np.ndarray]:
    """用严格历史残差构造供完整费用选择的动态分位数候选。"""

    base = np.asarray(base_forecast, dtype=float)
    actual = np.asarray(actual_net_kwh, dtype=float)
    if base.ndim != 2 or base.shape != actual.shape or base.shape[1] != 144:
        raise ValueError("基础预测与实际净负荷必须是同形的days×144矩阵")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(actual)):
        raise ValueError("基础预测与实际净负荷必须全部有限")
    if window_days <= 0 or not tau_candidates:
        raise ValueError("残差窗口必须为正且分位数候选不能为空")
    taus = tuple(float(tau) for tau in tau_candidates)
    if len(set(taus)) != len(taus) or any(not 0.0 < tau < 1.0 for tau in taus):
        raise ValueError("分位数候选必须互异且均位于(0,1)")
    names = tuple(f"tau_{tau:.2f}" for tau in taus)
    if len(set(names)) != len(names):
        raise ValueError("分位数候选保留两位小数后名称冲突")
    return {
        f"tau_{tau:.2f}": causal_residual_quantile_forecast(
            base,
            actual,
            tau=tau,
            window_days=window_days,
        )
        for tau in sorted(taus)
    }


def quantiles_from_selected_names(selected_names: np.ndarray) -> np.ndarray:
    """将 ``tau_0.80`` 形式的每日候选名转换为分位数数组。"""

    names = np.asarray(selected_names, dtype=object)
    if names.ndim != 1:
        raise ValueError("每日分位数候选名称必须是一维数组")
    values = np.empty(len(names), dtype=float)
    for index, name in enumerate(names):
        text = str(name)
        if not text.startswith("tau_"):
            raise ValueError(f"无法识别分位数候选名称: {text}")
        try:
            values[index] = float(text.removeprefix("tau_"))
        except ValueError as exc:
            raise ValueError(f"无法识别分位数候选名称: {text}") from exc
    if np.any((values <= 0.0) | (values >= 1.0)):
        raise ValueError("解析出的分位数必须位于(0,1)")
    return values


def build_fixed_quantile_candidates(
    actual_load_kwh: np.ndarray,
    actual_pv_kwh: np.ndarray,
    price: np.ndarray,
    *,
    tau: float = 0.8,
) -> dict[str, np.ndarray]:
    """构造周期窗口/衰减参数的严格因果固定分位数候选。"""

    if not 0.0 < tau < 1.0:
        raise ValueError("tau须位于(0,1)")
    load = np.asarray(actual_load_kwh, dtype=float)
    pv = np.asarray(actual_pv_kwh, dtype=float)
    suite = build_forecast_suite(load, pv, price)
    return _fixed_quantile_candidates_from_suite(suite, load - pv, tau)


def _fixed_quantile_candidates_from_suite(
    suite: ForecastSuite,
    actual_net_kwh: np.ndarray,
    tau: float,
) -> dict[str, np.ndarray]:
    weighted = {
        name: values
        for name, values in suite.point_candidates.items()
        if name.startswith("weekly_")
    }
    return {
        name: causal_residual_quantile_forecast(
            values,
            actual_net_kwh,
            tau=tau,
            window_days=int(name.split("_w")[1].split("_")[0]),
        )
        for name, values in weighted.items()
    }


def select_trailing_cost_winner(
    candidate_daily_costs: Mapping[str, np.ndarray],
    *,
    day: int,
    validation_days: int = 28,
    min_history_days: int = 7,
    fallback_name: str,
    reference_name: str | None = None,
    min_relative_improvement: float = 0.0,
) -> str:
    """按严格位于 ``day`` 之前的滚动完整费用选择候选。

    费用相同时按候选名称排序，保证结果在不同进程和平台上可复现。冷启动阶段
    不从不足量的历史中做不稳定选择，而是返回显式配置的回退候选。
    """

    if not candidate_daily_costs:
        raise ValueError("候选费用不能为空")
    if fallback_name not in candidate_daily_costs:
        raise ValueError("回退候选必须存在于候选费用中")
    if reference_name is not None and reference_name not in candidate_daily_costs:
        raise ValueError("参考候选必须存在于候选费用中")
    if (
        day < 0
        or validation_days <= 0
        or min_history_days < 0
        or not 0.0 <= min_relative_improvement < 1.0
    ):
        raise ValueError("日期和滚动窗口参数不合法")

    costs = {name: np.asarray(values, dtype=float) for name, values in candidate_daily_costs.items()}
    lengths = {len(values) for values in costs.values()}
    if len(lengths) != 1 or day > next(iter(lengths)):
        raise ValueError("全部候选费用必须等长且覆盖所选日期")

    start = max(0, day - validation_days)
    if day - start < min_history_days:
        return fallback_name

    scores: list[tuple[float, str]] = []
    for name, values in costs.items():
        history = values[start:day]
        if not np.all(np.isfinite(history)):
            raise ValueError("历史候选费用必须全部有限")
        scores.append((float(np.sum(history)), name))
    best_score, best_name = min(scores)
    if reference_name is None or best_name == reference_name:
        return best_name
    reference_score = next(score for score, name in scores if name == reference_name)
    required_score = reference_score * (1.0 - min_relative_improvement)
    return best_name if best_score <= required_score else reference_name


def _empty_run(days: int, periods: int) -> dict[str, np.ndarray]:
    matrix_names = (
        "grid",
        "grid_used",
        "charge",
        "discharge",
        "emergency",
        "spill",
        "pv_used",
        "pv_curtailed",
    )
    run = {name: np.zeros((days, periods)) for name in matrix_names}
    run["soc"] = np.zeros((days, periods + 1))
    run["normal_cost"] = np.zeros(days)
    run["emergency_cost"] = np.zeros(days)
    return run


def _store_selected_day(
    run: dict[str, np.ndarray],
    day: int,
    frozen_grid: np.ndarray,
    actual: object,
    normal_cost: float,
    emergency_cost: float,
) -> None:
    run["grid"][day] = frozen_grid
    for name in (
        "grid_used",
        "charge",
        "discharge",
        "emergency",
        "spill",
        "pv_used",
        "pv_curtailed",
        "soc",
    ):
        run[name][day] = getattr(actual, name)
    run["normal_cost"][day] = normal_cost
    run["emergency_cost"][day] = emergency_cost


def run_full_cost_selection(
    candidate_forecasts: Mapping[str, np.ndarray],
    actual_load_kwh: np.ndarray,
    actual_pv_kwh: np.ndarray,
    price: np.ndarray,
    *,
    validation_days: int = 28,
    min_history_days: int = 7,
    fallback_name: str,
    reference_name: str | None = None,
    min_relative_improvement: float = 0.0,
    initial_soc_kwh: float = 6000.0,
    config: DispatchConfig | None = None,
    progress: bool = True,
) -> FullCostSelectionResult:
    """以候选的历史完整调度费用进行严格因果的在线选择。

    每个历史日的所有候选都从主策略在该日的同一初始 SOC 出发，分别完成日前
    优化、实际回放和费用结算。第 ``day`` 天的候选在当天反事实费用计算前选定，
    因而选择步骤不可能读取当天或未来实际成本。
    """

    cfg = config or DispatchConfig()
    load = np.asarray(actual_load_kwh, dtype=float)
    pv = np.asarray(actual_pv_kwh, dtype=float)
    prices = np.asarray(price, dtype=float)
    if load.ndim != 2 or load.shape != pv.shape or load.shape[1] != 144:
        raise ValueError("实际负荷和光伏必须是同形的 days×144 电量矩阵")
    if not candidate_forecasts or fallback_name not in candidate_forecasts:
        raise ValueError("候选预测不能为空且必须包含回退候选")
    if prices.shape not in ((144,), load.shape):
        raise ValueError("电价必须是144点序列或与负荷同形的矩阵")
    if np.any(load < 0.0) or np.any(pv < 0.0) or np.any(prices < 0.0):
        raise ValueError("负荷、光伏和和电价不能为负")
    if not np.all(np.isfinite(load)) or not np.all(np.isfinite(pv)) or not np.all(np.isfinite(prices)):
        raise ValueError("负荷、光伏和电价必须全部有限")

    days, periods = load.shape
    names = sorted(candidate_forecasts)
    forecasts = {name: np.asarray(candidate_forecasts[name], dtype=float) for name in names}
    if any(values.shape != load.shape for values in forecasts.values()):
        raise ValueError("全部候选预测必须与实际负荷同形")
    if any(not np.all(np.isfinite(values)) for values in forecasts.values()):
        raise ValueError("候选预测必须全部有限")

    run = _empty_run(days, periods)
    selected_forecast = np.zeros_like(load)
    selected_names = np.empty(days, dtype=object)
    candidate_costs = {name: np.full(days, np.nan) for name in names}
    candidate_normal_costs = {name: np.full(days, np.nan) for name in names}
    candidate_emergency_costs = {name: np.full(days, np.nan) for name in names}
    candidate_initial_soc = {name: np.full(days, np.nan) for name in names}
    current_soc = float(initial_soc_kwh)

    for day in range(days):
        selected_name = select_trailing_cost_winner(
            candidate_costs,
            day=day,
            validation_days=validation_days,
            min_history_days=min_history_days,
            fallback_name=fallback_name,
            reference_name=reference_name,
            min_relative_improvement=min_relative_improvement,
        )
        selected_names[day] = selected_name
        selected_forecast[day] = forecasts[selected_name][day]
        day_price = price_for_day(prices, day, days)
        selected_outcome: tuple[np.ndarray, object, float, float] | None = None

        for name in names:
            candidate_initial_soc[name][day] = current_soc
            plan = optimize_day_ahead(forecasts[name][day], day_price, current_soc, cfg)
            if not plan.success:
                raise RuntimeError(f"第{day + 1}天候选{name}日前计划求解失败: {plan.message}")
            frozen_grid = plan.grid.copy()
            actual = simulate_realtime(frozen_grid, load[day], pv[day], current_soc, cfg)
            cost = calculate_problem2_cost(frozen_grid, actual.emergency, day_price)
            candidate_costs[name][day] = cost.total_cost
            candidate_normal_costs[name][day] = cost.normal_cost
            candidate_emergency_costs[name][day] = cost.emergency_cost
            if name == selected_name:
                selected_outcome = (
                    frozen_grid,
                    actual,
                    cost.normal_cost,
                    cost.emergency_cost,
                )

        if selected_outcome is None:
            raise RuntimeError("未保留当天选中候选的调度结果")
        frozen_grid, actual, normal_cost, emergency_cost = selected_outcome
        _store_selected_day(run, day, frozen_grid, actual, normal_cost, emergency_cost)
        current_soc = float(actual.soc[-1])
        if progress and (day + 1) % 30 == 0:
            print(f"完整费用滚动选择: {day + 1}/{days} days", flush=True)

    return FullCostSelectionResult(
        run=run,
        forecast=selected_forecast,
        selected_names=selected_names,
        candidate_daily_costs=candidate_costs,
        candidate_daily_normal_costs=candidate_normal_costs,
        candidate_daily_emergency_costs=candidate_emergency_costs,
        candidate_initial_soc_kwh=candidate_initial_soc,
    )


def problem2_result_from_selection(
    selection: FullCostSelectionResult,
    *,
    forecast_mean: np.ndarray,
    selected_alpha: np.ndarray,
    max_history_day_used: np.ndarray,
    leakage_detected: bool = False,
) -> Problem2YearResult:
    """将费用选择结果转换为现有问题二审计/工作簿使用的结果结构。"""

    days, periods = selection.forecast.shape
    mean = np.asarray(forecast_mean, dtype=float)
    alpha = np.asarray(selected_alpha, dtype=float)
    history = np.asarray(max_history_day_used, dtype=int)
    if mean.shape != (days, periods):
        raise ValueError("基础预测与选择结果维度不一致")
    if alpha.shape != (days,) or np.any((alpha <= 0.0) | (alpha >= 1.0)):
        raise ValueError("每日分位数必须是位于(0,1)的一维数组")
    if history.shape != (days,) or np.any(history >= np.arange(days)):
        raise ValueError("最大历史日期必须严格早于对应预测日")
    run = selection.run
    return Problem2YearResult(
        grid=run["grid"],
        grid_used=run["grid_used"],
        charge=run["charge"],
        discharge=run["discharge"],
        pv_used=run["pv_used"],
        pv_curtailed=run["pv_curtailed"],
        spill=run["spill"],
        emergency=run["emergency"],
        soc=run["soc"],
        daily_purchase_cost=run["normal_cost"],
        daily_emergency_cost=run["emergency_cost"],
        forecast_mean=mean,
        forecast_risk=selection.forecast,
        selected_alpha=alpha,
        max_history_day_used=history,
        leakage_detected=bool(leakage_detected),
        mean_strategy={},
    )


def run_cost_optimized_study(
    actual_load_kwh: np.ndarray,
    actual_pv_kwh: np.ndarray,
    price: np.ndarray,
    *,
    validation_days: int = 28,
    min_history_days: int = 7,
    tau_candidates: tuple[float, ...] = DEFAULT_TAU_CANDIDATES,
    quantile_window_days: int = 28,
    dynamic_min_relative_improvement: float = DEFAULT_DYNAMIC_MIN_RELATIVE_IMPROVEMENT,
    config: DispatchConfig | None = None,
    progress: bool = True,
) -> Problem2CostOptimizedStudy:
    """依次运行原E方案、费用驱动参数选择和动态分位数。"""

    cfg = config or DispatchConfig()
    load = np.asarray(actual_load_kwh, dtype=float)
    pv = np.asarray(actual_pv_kwh, dtype=float)
    actual_net = load - pv
    suite = build_forecast_suite(load, pv, price, validation_days=validation_days)
    history = suite.max_history_day_used
    days = len(load)

    if progress:
        print("阶段0/2：复算当前固定0.8分位E方案", flush=True)
    baseline_selection = run_full_cost_selection(
        {DEFAULT_TAU_FALLBACK: suite.forecasts["E_0.8分位数成本敏感"]},
        load,
        pv,
        price,
        validation_days=validation_days,
        min_history_days=min_history_days,
        fallback_name=DEFAULT_TAU_FALLBACK,
        config=cfg,
        progress=False,
    )
    baseline_result = problem2_result_from_selection(
        baseline_selection,
        forecast_mean=suite.forecasts["D_RMSE最优"],
        selected_alpha=np.full(days, 0.8),
        max_history_day_used=history,
        leakage_detected=suite.leakage_detected,
    )

    if progress:
        print("阶段1/2：固定tau=0.80，按过去完整调度费用选择窗口/衰减", flush=True)
    fixed_candidates = _fixed_quantile_candidates_from_suite(suite, actual_net, 0.8)
    fixed_selection = run_full_cost_selection(
        fixed_candidates,
        load,
        pv,
        price,
        validation_days=validation_days,
        min_history_days=min_history_days,
        fallback_name=DEFAULT_WINDOW_DECAY_FALLBACK,
        config=cfg,
        progress=progress,
    )
    base = selected_base_forecast(suite.point_candidates, fixed_selection.selected_names)
    fixed_result = problem2_result_from_selection(
        fixed_selection,
        forecast_mean=base,
        selected_alpha=np.full(days, 0.8),
        max_history_day_used=history,
        leakage_detected=suite.leakage_detected,
    )

    if progress:
        print("阶段2/2：按过去完整调度费用动态选择分位数", flush=True)
    dynamic_candidates = build_dynamic_quantile_candidates(
        base,
        actual_net,
        tau_candidates=tau_candidates,
        window_days=quantile_window_days,
    )
    dynamic_selection = run_full_cost_selection(
        dynamic_candidates,
        load,
        pv,
        price,
        validation_days=validation_days,
        min_history_days=min_history_days,
        fallback_name=DEFAULT_TAU_FALLBACK,
        reference_name=DEFAULT_TAU_FALLBACK,
        min_relative_improvement=dynamic_min_relative_improvement,
        config=cfg,
        progress=progress,
    )
    dynamic_result = problem2_result_from_selection(
        dynamic_selection,
        forecast_mean=base,
        selected_alpha=quantiles_from_selected_names(dynamic_selection.selected_names),
        max_history_day_used=history,
        leakage_detected=suite.leakage_detected,
    )
    return Problem2CostOptimizedStudy(
        forecast_suite=suite,
        baseline_result=baseline_result,
        fixed_selection=fixed_selection,
        fixed_result=fixed_result,
        dynamic_selection=dynamic_selection,
        dynamic_result=dynamic_result,
        selected_base_forecast=base,
    )
