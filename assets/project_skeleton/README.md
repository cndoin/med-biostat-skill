# 项目骨架 · 目录契约（S7）

按此契约建目录，脚本按此定位产物。**目录名固定，不要自创变体。**

```
<project>/
├── scripts/    分析脚本（确定性、可重跑）
├── tables/     中间表与结果表（TSV/CSV，带表头与口径列）
├── report/     报告正文（Markdown）
├── figures/    图（矢量优先 SVG/PDF）
├── web/        交互式 HTML
├── logs/       结构化日志（*.json）
├── raw/        原始取回数据（只读）
├── curated/    清洗/统一口径后的数据（Parquet 优先）
└── output/     最终交付物（DOCX/PDF/打包）
```

## 必须存在的四个日志

| 文件 | 何时写 | 内容 |
|---|---|---|
| `logs/00_intake.json` | S0 完成 | 研究形态、路由决策、判定门清单 |
| `logs/provenance.json` | S1 完成 | 每个数据源的版本号 + 获取日期 + 哈希 |
| `logs/gates.json` | 每道门判定时 | 门编号、时点、判定、依据数据、处置 |
| `logs/confounder_audit.json` | S4 完成 | 十二项逐条：已查/结论/无法查及原因 |

## 三条硬规则

1. `raw/` 一旦落盘**不得原地修改**，只能新增版本目录。
2. 产物文件名不得含 `final` / `new` / `最新` 等无版本语义的词。
3. `logs/` 下的 JSON 必须可被直接解析（`python -m json.tool` 无报错）。
