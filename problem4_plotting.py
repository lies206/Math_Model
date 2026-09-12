"""问题四的零依赖SVG绘图后端；在matplotlib不可用时保持脚本可运行。"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _axis_bounds(values: np.ndarray) -> tuple[float, float]:
    lower = min(0.0, float(np.min(values)))
    upper = max(0.0, float(np.max(values)))
    if abs(upper - lower) < 1e-12:
        upper = lower + 1.0
    padding = 0.06 * (upper - lower)
    return lower - padding, upper + padding


def write_svg_line_chart(
    output_path: str | Path,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[tuple[str, np.ndarray, str]],
    *,
    x_ticks: Sequence[tuple[int, str]] | None = None,
    secondary: tuple[str, np.ndarray, str, str] | None = None,
) -> Path:
    """用标准库写多折线SVG，可选右侧第二纵轴。"""
    if not series:
        raise ValueError("至少需要一条曲线")
    arrays = [(label, np.asarray(values, dtype=float).reshape(-1), color) for label, values, color in series]
    count = len(arrays[0][1])
    if count == 0 or any(len(values) != count for _, values, _ in arrays):
        raise ValueError("主坐标曲线必须等长且非空")
    if any(not np.all(np.isfinite(values)) for _, values, _ in arrays):
        raise ValueError("曲线包含NaN/Inf")

    width, height = 1400, 720
    left, right, top, bottom = 108, 108, 88, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = _axis_bounds(np.concatenate([values for _, values, _ in arrays]))

    def x_pos(index: int) -> float:
        return left + plot_w * index / max(count - 1, 1)

    def y_pos(value: float) -> float:
        return top + plot_h * (y_max - value) / (y_max - y_min)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>text{font-family:"Microsoft YaHei","SimHei",Arial,sans-serif;fill:#263238}</style>',
        f'<text x="{left}" y="38" font-size="26" font-weight="700">{escape(title)}</text>',
    ]
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5.0
        y = y_pos(value)
        elements.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#E4EAF0"/>',
                f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-size="14">{value:.2f}</text>',
            ]
        )
    ticks = x_ticks or [
        (int(round((count - 1) * fraction / 4.0)), f"{25 * fraction}%") for fraction in range(5)
    ]
    for index, label in ticks:
        index = max(0, min(int(index), count - 1))
        x = x_pos(index)
        elements.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#F0F3F6"/>',
                f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" font-size="14">{escape(str(label))}</text>',
            ]
        )
    for label, values, color in arrays:
        points = " ".join(f"{x_pos(i):.2f},{y_pos(float(value)):.2f}" for i, value in enumerate(values))
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round"/>'
        )

    legend_x = left
    for label, _, color in arrays:
        elements.extend(
            [
                f'<line x1="{legend_x}" y1="64" x2="{legend_x + 32}" y2="64" stroke="{color}" stroke-width="4"/>',
                f'<text x="{legend_x + 40}" y="69" font-size="15">{escape(label)}</text>',
            ]
        )
        legend_x += max(155, 16 * len(label))

    if secondary is not None:
        sec_label, sec_values_raw, sec_color, sec_y_label = secondary
        sec_values = np.asarray(sec_values_raw, dtype=float).reshape(-1)
        if len(sec_values) != count or not np.all(np.isfinite(sec_values)):
            raise ValueError("第二纵轴曲线必须与主坐标等长且有限")
        sec_min, sec_max = _axis_bounds(sec_values)

        def sec_y(value: float) -> float:
            return top + plot_h * (sec_max - value) / (sec_max - sec_min)

        points = " ".join(f"{x_pos(i):.2f},{sec_y(float(value)):.2f}" for i, value in enumerate(sec_values))
        elements.extend(
            [
                f'<polyline points="{points}" fill="none" stroke="{sec_color}" stroke-width="2.2" stroke-dasharray="8 5"/>',
                f'<line x1="{legend_x}" y1="64" x2="{legend_x + 32}" y2="64" stroke="{sec_color}" stroke-width="4" stroke-dasharray="8 5"/>',
                f'<text x="{legend_x + 40}" y="69" font-size="15">{escape(sec_label)}</text>',
            ]
        )
        for tick in range(6):
            value = sec_min + (sec_max - sec_min) * tick / 5.0
            y = sec_y(value)
            elements.append(f'<text x="{left + plot_w + 12}" y="{y + 5:.2f}" font-size="14">{value:.3f}</text>')
        elements.append(
            f'<text x="{width - 18}" y="{top + plot_h / 2}" font-size="16" text-anchor="middle" transform="rotate(90 {width - 18} {top + plot_h / 2})">{escape(sec_y_label)}</text>'
        )

    elements.extend(
        [
            f'<text x="25" y="{top + plot_h / 2}" font-size="16" text-anchor="middle" transform="rotate(-90 25 {top + plot_h / 2})">{escape(y_label)}</text>',
            f'<text x="{left + plot_w / 2}" y="{height - 24}" font-size="16" text-anchor="middle">{escape(x_label)}</text>',
            f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#90A4AE"/>',
            '</svg>',
        ]
    )
    path = Path(output_path)
    path.write_text("\n".join(elements), encoding="utf-8")
    return path


def write_svg_stacked_bar(
    output_path: str | Path,
    title: str,
    categories: Sequence[str],
    components: Sequence[tuple[str, np.ndarray, str]],
) -> Path:
    """写三策略费用堆叠柱状图。"""
    values = [(label, np.asarray(items, dtype=float).reshape(-1), color) for label, items, color in components]
    if not categories or any(len(items) != len(categories) for _, items, _ in values):
        raise ValueError("费用分量与策略数不一致")
    width, height = 1200, 720
    left, right, top, bottom = 120, 60, 95, 125
    plot_w, plot_h = width - left - right, height - top - bottom
    totals = np.sum([items for _, items, _ in values], axis=0)
    y_max = max(float(np.max(totals)) * 1.08, 1.0)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>text{font-family:"Microsoft YaHei","SimHei",Arial,sans-serif;fill:#263238}</style>',
        f'<text x="{left}" y="40" font-size="26" font-weight="700">{escape(title)}</text>',
    ]
    for tick in range(6):
        value = y_max * tick / 5.0
        y = top + plot_h * (1.0 - value / y_max)
        elements.extend([
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#E4EAF0"/>',
            f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-size="14">{value:.2f}</text>',
        ])
    slot = plot_w / len(categories)
    bar_width = slot * 0.46
    for index, category in enumerate(categories):
        x = left + slot * (index + 0.5) - bar_width / 2
        cumulative = 0.0
        for _, items, color in values:
            item = float(items[index])
            h = plot_h * item / y_max
            y = top + plot_h - plot_h * (cumulative + item) / y_max
            elements.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{h:.2f}" fill="{color}"/>')
            cumulative += item
        elements.append(f'<text x="{x + bar_width / 2:.2f}" y="{top + plot_h + 28}" text-anchor="middle" font-size="14">{escape(category)}</text>')
    legend_x = left
    for label, _, color in values:
        elements.extend([
            f'<rect x="{legend_x}" y="61" width="18" height="12" fill="{color}"/>',
            f'<text x="{legend_x + 26}" y="73" font-size="15">{escape(label)}</text>',
        ])
        legend_x += max(160, 16 * len(label))
    elements.extend([
        f'<text x="24" y="{top + plot_h / 2}" font-size="16" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})">费用/百万元</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#90A4AE"/>',
        '</svg>',
    ])
    path = Path(output_path)
    path.write_text("\n".join(elements), encoding="utf-8")
    return path


def create_problem4_svg_figures(
    output_dir: str | Path,
    data,
    study,
    actual_forecast_pairs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    output_start_day: int,
) -> list[Path]:
    """生成残差、ACF、费用、峰谷动作与SOC图。"""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    releases = ("0:00", "6:00", "12:00", "18:00")
    colors = ("#355C7D", "#2A9D8F", "#E9C46A", "#E76F51")
    date_labels = [str(value) for value in data.dates[output_start_day:]]
    date_ticks = [
        (index, date_labels[index])
        for index in sorted(set([0, len(date_labels) // 3, 2 * len(date_labels) // 3, len(date_labels) - 1]))
    ]
    residual_series = []
    for release, color in zip(releases, colors):
        actual, forecast = actual_forecast_pairs[release]
        residual_series.append((release, np.mean(actual - forecast, axis=1), color))
    residual_path = write_svg_line_chart(
        folder / "pv_residual_timeseries.svg",
        "光伏预测日均残差：实际值－预测值",
        "日期",
        "残差/kW",
        residual_series,
        x_ticks=date_ticks,
    )
    acf_path = write_svg_line_chart(
        folder / "pv_residual_acf.svg",
        "光伏预测残差自相关函数",
        "滞后/10分钟时段",
        "ACF",
        [(release, study.residual_analysis.acf[release], color) for release, color in zip(releases, colors)],
        x_ticks=[(0, "0"), (36, "36"), (72, "72"), (108, "108"), (144, "144")],
    )

    names = list(study.strategy_comparison)
    short_names = ["A 原策略", "B 动态电价", "C 滚动MPC"]
    plan = np.asarray([float(study.strategy_comparison[name]["plan_cost_yuan"]) for name in names]) / 1e6
    adjustment = np.asarray([float(study.strategy_comparison[name].get("adjustment_cost_yuan", 0.0)) for name in names]) / 1e6
    emergency = np.asarray([float(study.strategy_comparison[name]["emergency_cost_yuan"]) for name in names]) / 1e6
    cost_path = write_svg_stacked_bar(
        folder / "strategy_cost_comparison.svg",
        "问题四三类策略费用比较",
        short_names,
        [("计划购电", plan, "#355C7D"), ("调整费用", adjustment, "#E9C46A"), ("紧急购电", emergency, "#E76F51")],
    )

    output_prices = data.dynamic_price[output_start_day:]
    representative_day = output_start_day + int(np.argmax(np.ptp(output_prices, axis=1)))
    b_result = study.dynamic_problem2_study.final_result
    c_result = study.dynamic_mpc_result
    hour_ticks = [(index, label) for index, label in ((0, "0:00"), (36, "6:00"), (72, "12:00"), (108, "18:00"), (143, "24:00"))]
    action_path = write_svg_line_chart(
        folder / "dynamic_price_storage_action.svg",
        f"{str(data.dates[representative_day])} 动态电价与储能动作",
        "时刻",
        "储能功率/kW（充电为正）",
        [
            ("B 动态电价", (b_result.charge[representative_day] - b_result.discharge[representative_day]) * 6.0, "#2A9D8F"),
            ("C 滚动MPC", (c_result.charge[representative_day] - c_result.discharge[representative_day]) * 6.0, "#E76F51"),
        ],
        x_ticks=hour_ticks,
        secondary=("动态电价", data.dynamic_price[representative_day], "#333333", "电价/(元/kWh)"),
    )
    soc_ticks = [(index, label) for index, label in ((0, "0:00"), (36, "6:00"), (72, "12:00"), (108, "18:00"), (144, "24:00"))]
    soc_path = write_svg_line_chart(
        folder / "dynamic_price_storage_soc.svg",
        f"{str(data.dates[representative_day])} 储能SOC轨迹",
        "时刻",
        "SOC/kWh",
        [
            ("B 动态电价", b_result.soc[representative_day], "#2A9D8F"),
            ("C 滚动MPC", c_result.soc[representative_day], "#E76F51"),
            ("SOC下限", np.full(145, 1200.0), "#777777"),
            ("SOC上限", np.full(145, 10800.0), "#999999"),
        ],
        x_ticks=soc_ticks,
    )
    return [residual_path, acf_path, cost_path, action_path, soc_path]
