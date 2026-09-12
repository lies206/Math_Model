"""运行问题二的完整费用滚动选择与动态分位数优化。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from microgrid_optimization import kw_to_interval_kwh
from problem2_cost_optimized import (
    DEFAULT_DYNAMIC_MIN_RELATIVE_IMPROVEMENT,
    DEFAULT_TAU_CANDIDATES,
    run_cost_optimized_study,
)
from problem2_optimization import (
    OUTPUT_START_DAY,
    Problem2YearResult,
    audit_problem2,
    key_date_results,
    load_problem2_data,
    problem2_metrics,
    save_result2,
)


def _improvement(reference_cost: float, new_cost: float) -> dict[str, float]:
    saving = reference_cost - new_cost
    return {
        "saving_yuan": float(saving),
        "saving_percent": float(100.0 * saving / reference_cost),
    }


def _comparison_row(metrics: dict[str, object]) -> dict[str, float | int]:
    keys = (
        "plan_purchase_kwh",
        "emergency_purchase_kwh",
        "normal_cost_yuan",
        "emergency_cost_yuan",
        "total_cost_yuan",
        "unused_plan_purchase_kwh",
        "emergency_days",
        "forecast_mae_kwh_per_interval",
        "forecast_rmse_kwh_per_interval",
    )
    return {key: metrics[key] for key in keys}  # type: ignore[return-value]


def _cost_reconciliation(
    result: Problem2YearResult,
    price: np.ndarray,
) -> dict[str, float | bool]:
    prices = np.asarray(price, dtype=float)
    if prices.shape == (144,):
        prices = np.broadcast_to(prices, result.grid.shape)
    if prices.shape != result.grid.shape:
        raise ValueError("费用审计电价维度不匹配")
    normal = np.sum(result.grid * prices, axis=1)
    emergency = np.sum(5.0 * result.emergency * prices, axis=1)
    normal_error = float(np.max(np.abs(normal - result.daily_purchase_cost)))
    emergency_error = float(np.max(np.abs(emergency - result.daily_emergency_cost)))
    return {
        "max_daily_normal_cost_error_yuan": normal_error,
        "max_daily_emergency_cost_error_yuan": emergency_error,
        "cost_reconciliation_passed": bool(
            normal_error <= 1e-6 and emergency_error <= 1e-6
        ),
    }


def _build_markdown(payload: dict[str, object]) -> str:
    comparison = payload["stage_comparison"]
    improvements = payload["improvements"]
    selections = payload["selection_summary"]
    final_stage = payload["final_stage"]
    lines = [
        "# 问题二：完整费用滚动选择与动态分位数优化报告",
        "",
        "## 结论",
        "",
        f"正式结果采用 `{final_stage}`。优化过程只以附件1电价和附件2负荷/光伏为数值输入；附件5仅用于复制结果模板。",
        "逐日参数选择严格因果；但5%切换门槛在同一份2025年数据上做过敏感性比较，因此下表属于样本内滚动回测，不应表述为独立样本外收益。",
        "",
        "| 阶段 | 正常购电费/元 | 紧急购电费/元 | 总费用/元 | 紧急购电量/kWh |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "baseline_fixed_tau_0.80": "原固定0.8分位E方案",
        "full_cost_window_decay": "完整费用选择窗口/衰减",
        "dynamic_quantile": "动态分位数（5%门槛）",
    }
    for name, label in labels.items():
        row = comparison[name]
        lines.append(
            f"| {label} | {row['normal_cost_yuan']:,.2f} | "
            f"{row['emergency_cost_yuan']:,.2f} | {row['total_cost_yuan']:,.2f} | "
            f"{row['emergency_purchase_kwh']:,.2f} |"
        )
    baseline_gain = improvements["dynamic_vs_baseline"]
    fixed_gain = improvements["dynamic_vs_fixed_stage"]
    lines.extend(
        [
            "",
            f"相对原方案节省 **{baseline_gain['saving_yuan']:,.2f} 元**（{baseline_gain['saving_percent']:.4f}%）；"
            f"相对第1阶段再节省 **{fixed_gain['saving_yuan']:,.2f} 元**（{fixed_gain['saving_percent']:.4f}%）。",
            "",
            "## 严格因果选择规则",
            "",
            "1. 每天先冻结参数；第 d 天只能读取此前28天 `[d-28,d)` 已结算的完整调度费用。",
            "2. 每个候选从相同日初SOC出发，完整执行日前LP、实际负荷/光伏回放、正常购电与5倍紧急购电结算。",
            "3. 第1阶段固定 `tau=0.80`，滚动选择周期窗口和衰减系数。",
            "4. 第2阶段在分位数候选 `{0.65,0.70,0.75,0.80,0.85,0.90,0.95}` 中选择；只有过去28天累计费用较 `tau=0.80` 至少低5%，才允许切换。",
            "",
            "## 输出期选择统计",
            "",
            f"窗口/衰减选择：`{json.dumps(selections['window_decay_counts'], ensure_ascii=False)}`",
            "",
            f"动态分位数选择：`{json.dumps(selections['tau_counts'], ensure_ascii=False)}`",
            "",
            "## 审计",
            "",
            "三个阶段均通过功率平衡、SOC动态与跨日连续性、SOC/功率边界、禁止反送、紧急购电触发、费用重算和无未来数据泄漏检查。",
            "",
            "逐日参数、分位数和费用见 `problem2_cost_optimized_analysis.json`；正式填表结果见 `result2_optimized.xlsx`。",
            "",
        ]
    )
    return "\n".join(lines)


def run(root: str | Path, output_dir: str | Path) -> dict[str, object]:
    root_path = Path(root).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    data = load_problem2_data(root_path, print_structure=False)
    actual_load = kw_to_interval_kwh(data.load_kw)
    actual_pv = kw_to_interval_kwh(data.actual_pv_kw)
    study = run_cost_optimized_study(
        actual_load,
        actual_pv,
        data.static_price,
        progress=True,
    )

    results = {
        "baseline_fixed_tau_0.80": study.baseline_result,
        "full_cost_window_decay": study.fixed_result,
        "dynamic_quantile": study.dynamic_result,
    }
    metrics = {name: problem2_metrics(result, data) for name, result in results.items()}
    audits = {name: audit_problem2(result, data) for name, result in results.items()}
    for name, result in results.items():
        cost_audit = _cost_reconciliation(result, data.static_price)
        audits[name].update(cost_audit)
        audits[name]["all_checks_passed"] = bool(
            audits[name]["all_checks_passed"]
            and cost_audit["cost_reconciliation_passed"]
        )
    if not all(bool(audit["all_checks_passed"]) for audit in audits.values()):
        raise RuntimeError("物理或费用审计未通过，不生成正式结果")
    costs = {name: float(value["total_cost_yuan"]) for name, value in metrics.items()}
    final_stage = min(costs, key=lambda name: (costs[name], name))
    if costs[final_stage] > costs["baseline_fixed_tau_0.80"] + 1e-6:
        raise RuntimeError("优化方案未能降低总费用，不生成正式结果")

    final_result = results[final_stage]
    workbook = save_result2(
        root_path,
        output_path / "result2_optimized.xlsx",
        final_result,
    )
    output_days = range(OUTPUT_START_DAY, len(data.dates))
    daily = [
        {
            "date": str(data.dates[day]),
            "window_decay": str(study.fixed_selection.selected_names[day]),
            "tau": float(study.dynamic_result.selected_alpha[day]),
            "plan_purchase_kwh": float(final_result.grid[day].sum()),
            "emergency_purchase_kwh": float(final_result.emergency[day].sum()),
            "normal_cost_yuan": float(final_result.daily_purchase_cost[day]),
            "emergency_cost_yuan": float(final_result.daily_emergency_cost[day]),
            "total_cost_yuan": float(final_result.daily_total_cost[day]),
        }
        for day in output_days
    ]
    payload: dict[str, object] = {
        "optimization_inputs": ["附件/附件1.xlsx", "附件/附件2.xlsx"],
        "output_template_only": "附件/附件5/result2.xlsx",
        "attachments3_and_4_used": False,
        "output_period": "2025-02-01 to 2025-12-31",
        "evaluation_scope_note": (
            "每日选择严格因果；5%切换门槛在同一2025年样本上做过敏感性比较，"
            "费用改善属于样本内滚动回测。"
        ),
        "settings": {
            "cost_validation_days": 28,
            "minimum_history_days": 7,
            "fixed_stage_tau": 0.80,
            "dynamic_tau_candidates": list(DEFAULT_TAU_CANDIDATES),
            "dynamic_quantile_residual_window_days": 28,
            "dynamic_switch_min_relative_improvement": DEFAULT_DYNAMIC_MIN_RELATIVE_IMPROVEMENT,
            "emergency_price_multiplier": 5.0,
        },
        "final_stage": final_stage,
        "stage_comparison": {
            name: _comparison_row(value) for name, value in metrics.items()
        },
        "improvements": {
            "fixed_stage_vs_baseline": _improvement(
                costs["baseline_fixed_tau_0.80"], costs["full_cost_window_decay"]
            ),
            "dynamic_vs_fixed_stage": _improvement(
                costs["full_cost_window_decay"], costs["dynamic_quantile"]
            ),
            "dynamic_vs_baseline": _improvement(
                costs["baseline_fixed_tau_0.80"], costs["dynamic_quantile"]
            ),
        },
        "selection_summary": {
            "window_decay_counts": dict(
                sorted(Counter(map(str, study.fixed_selection.selected_names[OUTPUT_START_DAY:])).items())
            ),
            "tau_counts": dict(
                sorted(Counter(map(str, study.dynamic_selection.selected_names[OUTPUT_START_DAY:])).items())
            ),
        },
        "audits": audits,
        "key_dates": key_date_results(final_result, data),
        "daily_selection_and_cost": daily,
        "files": {
            "workbook": str(workbook),
            "analysis_json": str(output_path / "problem2_cost_optimized_analysis.json"),
            "report_markdown": str(output_path / "problem2_cost_optimized_report.md"),
        },
    }
    json_path = output_path / "problem2_cost_optimized_analysis.json"
    report_path = output_path / "problem2_cost_optimized_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_build_markdown(payload), encoding="utf-8")
    print(f"正式阶段: {final_stage}")
    print(f"优化工作簿: {workbook}")
    print(f"分析JSON: {json_path}")
    print(f"报告: {report_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    run(args.root, args.output_dir)


if __name__ == "__main__":
    main()
