# adapters · 各 agent 入口模板与生成规则

## 原则：单一来源，生成不手写

`SKILL.md` 是**唯一权威来源**。各 agent 的入口文件由
`scripts/install_adapters.py` 从 SKILL.md + VERSION 生成，**不允许手写第二份**。

> 依据：设计文档附录 B 陷阱 #21「同一逻辑多处实现 → 同一句话在不同模块得到不同结论」。
> 手写多份入口 = 迟早漂移。

## 各端入口形态

| target | 产物 | 说明 |
|---|---|---|
| `claude` | 复制整包到 `~/.claude/skills/biostat-copilot/` | Claude Code 原生读 `SKILL.md` |
| `codex` | 复制整包到 `~/.codex/skills/biostat-copilot/` | Codex 读同一套 `SKILL.md` 规范 |
| `agents` | `<project>/AGENTS.md` | 通用入口；Cursor/Zed/Amp/Jules/aider 等读它 |
| `cursor` | `<project>/.cursor/rules/biostat-copilot.mdc` | 前置 YAML：description / globs / alwaysApply |
| `cline` | `<project>/.clinerules/biostat-copilot.md` | Cline 规则目录 |
| `gemini` | `<project>/GEMINI.md` | Gemini CLI |
| `copilot` | `<project>/.github/copilot-instructions.md` | GitHub Copilot |

## 生成内容

生成的入口文件只包含**触发描述 + 路由摘要 + 指向 SKILL.md 绝对路径 + 脚本调用方式**，
不复制 SKILL.md 全文（避免漂移与上下文膨胀）。SKILL.md 的正文由 agent 在触发后按需读取。

`--target all` 一次性安装/生成全部。
