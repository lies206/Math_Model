# Implementation Plan: 问题二费用驱动滚动选择与动态分位数

## Overview

在不读取附件3、附件4或历史结果文件作为优化输入的前提下，只使用附件1固定电价与附件2负荷/光伏，依次实现两项改进：先用过去窗口内完整的“日前计划—实时回放—结算费用”选择预测参数，再对分位数水平进行严格因果的动态选择。每一阶段都与当前固定0.8分位数方案比较，只有数值审计通过才生成正式优化结果。

## Architecture Decisions

- 保留现有 `problem2_optimization.py` 稳定主线，新算法放入独立的 `problem2_cost_optimized.py`，避免在已有大型模块中继续堆叠编排逻辑。
- 每日候选参数必须在看到当天实际数据前选定；第 d 天只能用 `[d-window, d)` 的候选完整调度费用。
- 每个历史日的全部候选使用相同的主策略日初 SOC 进行反事实回放，保证费用比较公平；当天实际值只在选定参数后用于结算并进入后续历史。
- 第一阶段固定 `tau=0.8`，只选择周期窗口和衰减参数；第二阶段固定第一阶段当日基础预测，再从候选分位数中按过去完整费用动态选择。
- 新结果先写入 `result2_optimized.xlsx` 和独立 JSON/Markdown 报告，不覆盖当前 `result2.xlsx`。

## Task List

### Phase 1: Foundation

- [x] Task 1: 为滚动完整费用选择器编写失败测试。
  - Acceptance: 当天选择不受当天及未来费用变化影响；平局确定性；无足够历史时使用指定回退项。
  - Verification: `python -m unittest tests.test_problem2_cost_optimized -v` 必须先失败后通过。
  - Dependencies: None.
  - Files: `tests/test_problem2_cost_optimized.py`, `problem2_cost_optimized.py`.
  - Scope: Small.

- [x] Task 2: 实现在线反事实完整调度费用选择。
  - Acceptance: 所有候选在同一日初 SOC 下求解；当日选择只读取此前候选日费用；结果可转换为现有 `Problem2YearResult`。
  - Verification: 聚焦测试通过，合成数据审计通过。
  - Dependencies: Task 1.
  - Files: `problem2_cost_optimized.py`, `problem2_optimization.py`.
  - Scope: Medium.

### Checkpoint: Full-cost selector

- [x] 固定0.8分位数的窗口/衰减选择完成全年回测。
- [x] 只读取附件1和附件2，未来数据泄漏检查通过。
- [x] 与原始E方案形成可复算费用对比。

### Phase 2: Dynamic quantile

- [x] Task 3: 为动态分位数编写失败测试。
  - Acceptance: 分位数由过去完整费用选择；修改当天及未来实际值不改变当天分位数；选中值始终来自候选集合。
  - Verification: 聚焦测试必须先失败后通过。
  - Dependencies: Task 2.
  - Files: `tests/test_problem2_cost_optimized.py`, `problem2_cost_optimized.py`.
  - Scope: Small.

- [x] Task 4: 完成动态分位数全年回测。
  - Acceptance: 输出每日分位数、选择计数、逐日费用和最终调度；物理约束审计全部通过。
  - Verification: 优化脚本成功生成工作簿、JSON与Markdown报告。
  - Dependencies: Task 3.
  - Files: `problem2_cost_optimized.py`.
  - Scope: Medium.

### Phase 3: Verification

- [x] Task 5: 完整回归、结果核对和代码审查。
  - Acceptance: 全量测试通过；工作簿结构完整；附件白名单、因果性、功率平衡、SOC和费用一致。
  - Verification: `python -m unittest discover -s tests -v`；打开优化工作簿复核关键区域。
  - Dependencies: Task 4.
  - Files: 测试、输出报告与优化工作簿。
  - Scope: Medium.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| 16个窗口/衰减候选逐日完整LP回放计算量较高 | 中 | 缓存当日候选结果，选中候选直接复用；只保留必要候选并显示进度 |
| 动态分位数候选过多导致过拟合 | 高 | 使用有限候选网格、28日滚动窗口和确定性平局规则 |
| 候选自身SOC路径不同导致比较不公平 | 高 | 每个历史日所有候选统一使用主策略当日真实初始SOC |
| 新方法可能降低历史样本内费用但泛化变差 | 高 | 每日严格走步选择，不允许当天或未来实际值参与选择；保留原方案对照 |
| 技能引用的通用Definition of Done文件缺失 | 低 | 以仓库全量测试、数值审计、输入白名单和结果文件完整性作为完成门槛 |

## Open Questions

- 无阻塞项。用户已明确算法顺序和附件范围，可直接实施。
