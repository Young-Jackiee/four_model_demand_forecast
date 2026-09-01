# 需求预测四模型（V1.0）

这是一个按“SKU × 自然日”进行需求预测的 Python 项目。当前版本使用可复现的模拟订单数据，完整实现了数据标准化、特征构造、四模型回测、模型选择、生产重训、未来 50 天预测和轻量监控。

项目的首要原则是：时间顺序正确、避免数据泄漏、结果可复现。模型不会在预测测试期或未来期时读取对应日期的真实销量。

## 已实现内容

- 模拟源数据：SKU 主数据、原始订单明细、不可观测日期例外。
- `DailySeriesBuilder`：按 `sku + date` 聚合订单，补齐连续自然日历，区分真实零销量、未上市和不可观测日期。
- `FeatureBuilder`：构造 7/14/30/60/90 天滚动均值、发生率、严格 `t-365` 的年度同期特征和星期循环特征。
- 四个候选模型：FivePeriod、Direct10、TSB、Hurdle。
- 统一 `ForecastModel.fit / predict / serialize` 接口和 `DailyForecast` 输出结构。
- 固定 52 天正式递归回测、MAE/MSE/RMSE/WAPE/累计偏差计算、按基准模型双条件选模。
- winner 使用截至 `production_train_end` 的全部实际数据重训，再预测未来 50 天。
- 每个 SKU 独立保存模型 artifact、预测记录和 active 指针。
- 7/14/30 天监控：只有预测和实际日历完整覆盖窗口时才计算指标；未成熟窗口不会触发重训建议。

## 固定时间范围

| 阶段 | 日期范围 | 天数 |
|---|---|---:|
| 训练期 | 2025-01-01 至 2026-06-30 | 546 |
| 正式测试期 | 2026-07-01 至 2026-08-21 | 52 |
| 未来预测期 | 2026-08-22 至 2026-10-10 | 50 |

`production_train_end` 由上层配置显式传入，当前为 `2026-08-21`；模型不会通过输入数据的最大日期自行推断训练截止日。

## 项目结构

```text
four_model_demand_forecast/
├─ run_main.py                         # 推荐的完整程序入口
├─ configs/
│  ├─ synthetic_dataset.json           # 模拟数据和日期配置
│  └─ monitoring.json                  # 监控窗口与阈值配置
├─ src/demand_forecast/
│  ├─ data/                            # 模拟数据、数据契约、日序列加工
│  ├─ features/                        # FeatureBuilder 和特征定义
│  ├─ models/                          # 四个模型的核心数学实现
│  ├─ backtesting/                     # 统一回测、指标与适配层
│  ├─ model_selection.py               # winner_model 选择规则
│  ├─ forecast_models.py               # 文档规定的统一模型 API
│  ├─ production.py                    # 重训、预测和原子发布
│  ├─ forecast_service.py              # 加载 active 模型并推理
│  └─ monitoring.py                    # 已成熟预测的监控
├─ tests/                              # 单元测试和集成测试
└─ data/synthetic/
   ├─ source/                          # 模拟源数据
   ├─ derived/                         # 标准序列、选模和监控结果
   └─ production/                      # 每个 SKU 的生产 artifact
```

## 环境准备

项目使用根目录下的 `.venv` 虚拟环境。首次安装依赖：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

如果 PowerShell 阻止激活脚本，可直接使用虚拟环境解释器执行后续命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 运行项目

最简单的方式是在 PyCharm 中打开并运行根目录的 [run_main.py](C:/Users/Young/Desktop/four_model_demand_forecast/run_main.py)。无需填写命令行参数。

也可以在 PowerShell 执行：

```powershell
.\.venv\Scripts\python.exe run_main.py
```

主入口会依次完成：

1. 生成可复现的模拟源数据。
2. 构建标准 `daily_sales.csv`。
3. 检查训练期特征可用性。
4. 对每个 SKU 执行四模型的统一 52 天递归回测。
5. 按 MAE 和累计偏差绝对值选择 winner。
6. 使用截至 2026-08-21 的实际数据重训 winner。
7. 发布未来 50 天预测。
8. 对已成熟预测执行监控。

## 运行测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

当前最终验证结果：`90 passed`。

## 重要输出文件

| 文件或目录 | 用途 |
|---|---|
| `data/synthetic/source/sku_master.csv` | SKU 主数据和上市日期 |
| `data/synthetic/source/raw_sales.csv` | 原始订单明细；同 SKU 同日允许多行 |
| `data/synthetic/source/observation_exceptions.csv` | 不可观测日期，例如缺货或数据缺失 |
| `data/synthetic/derived/daily_sales.csv` | 唯一可供模型使用的标准日序列 |
| `data/synthetic/derived/selection_results.json` | 各 SKU 的回测审计和 `winner_model` |
| `data/synthetic/derived/production_run_results.json` | 本次生产训练和发布结果 |
| `data/synthetic/derived/monitoring_results.json` | 已成熟预测的 7/14/30 天监控结果 |
| `data/synthetic/production/skus/<SKU>/active.json` | 当前正在使用的模型版本 |
| `data/synthetic/production/skus/<SKU>/runs/<artifact_id>/` | 模型 artifact、预测记录和运行清单 |

## 数据语义

- `is_observed=true` 且没有订单：表示真实零销量，`quantity=0`。
- `is_observed=false`：表示未上市、缺货或数据不可用，`quantity` 必须为空，不得补零。
- 所有滚动特征在目标日 `t` 只读取 `t-1` 及更早的历史。
- 测试期的真实销量只用于预测完成后的指标计算，不会更新预测状态。

## 模型选择与异常 SKU

FivePeriod 是所有 SKU 的基准模型。Direct10、TSB、Hurdle 只有在同时满足以下条件时才可替代基准：

- MAE 严格更低；
- 累计偏差绝对值严格更低。

模拟数据中的 observation-gap SKU 可能因滚动窗口内存在不可观测日期而无法完成 FivePeriod 基准回测。这时系统会记录 `unselectable_baseline_unavailable` 并拒绝发布，而不是用测试数据补值或偷偷替换模型。这是安全的失败语义。

## 当前边界

- 参考文档中的原始 9 SKU 数据未包含在项目中，因此参考复现报告会显示 `pending_reference_data`。
- V1.0 的监控只给出重训建议，不会自动训练、自动切换模型或自动下单。
- 真实业务数据接入时，应替换源数据读取层，继续使用相同的 SKU 主数据、原始销售、不可观测状态和标准日序列契约。
