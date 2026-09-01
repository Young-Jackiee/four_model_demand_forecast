# 需求预测四模型

当前阶段已包含：

1. 生成可复现的模拟源数据；
2. 把源数据加工成 SKU × 自然日的标准日序列。
3. 构造严格无泄漏的训练特征；
4. 训练五周期基准模型、TSB、Hurdle 和 Direct10；
5. 通过 `FormalBacktestRunner` 在固定 52 天测试窗口执行 SKU × 四模型的统一回测，并在内存中生成后续 ModelSelector 可消费的指标表。
6. 通过 `ModelSelector` 严格比较 MAE 与累计偏差绝对值，保存可审计的 `winner_model` 决策。
7. 由上层配置显式传入 `production_train_end`，用同一模型训练流程重训 winner，生成并校验未来 50 天预测。
8. 按 SKU 独立原子发布生产 artifact；失败时不自动换模型，且不会覆盖该 SKU 的上一版 active artifact。
9. 四模型通过统一 `ForecastModel.fit / predict / serialize` 接口运行；预测接口不接收测试期或未来期实际销量。
10. 可通过 `ForecastService` 加载某个 SKU 的 active artifact，并再次生成带组件解释的日预测。

模拟实际数据严格覆盖 2025-01-01 至 2026-08-21：其中训练期截至 2026-06-30（546 天），测试期为后续 52 天。未来 50 天只作为预测范围，不生成实际销量。

日常使用时，直接在 PyCharm 运行项目根目录的 `run_main.py`，无需填写参数。它会依次生成模拟源数据、标准日序列、检查训练期特征，在 `2026-07-01` 至 `2026-08-21` 的完整 52 天窗口回测四个模型，并按配置中的 `production_train_end=2026-08-21` 重训 winner、预测未来 50 天。

生产输出位于 `data/synthetic/production/skus/<SKU>/`：每个 SKU 单独保存 `active.json`、运行目录、模型 artifact 和预测记录，避免 SKU 间共享状态。

生产模型的公共调用方式为：

```python
fitted = model.fit(daily_series, train_end)
forecasts = model.predict(fitted, horizon=50)
artifact = model.serialize(fitted)
```

其中 `train_end` 必须由上层显式传入。FivePeriod、Hurdle 会保存 90 天递归历史，Direct10 保存 455 天递归历史；TSB 保存冻结状态。这样 `predict()` 不需要再传入任何实际销量表。

命令行运行方式：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
generate-synthetic-data --config configs/synthetic_dataset.json --output-dir data/synthetic/source
build-daily-series --source-dir data/synthetic/source --output-file data/synthetic/derived/daily_sales.csv --history-start 2025-01-01 --actual-end 2026-08-21
```

项目虚拟环境目录为 `.venv`。

`observation_exceptions.csv` 只记录不可观测例外；它不是每日状态表。标准日序列中，`is_observed=false` 的 `quantity` 会保持为空，不能当作零销量使用。
