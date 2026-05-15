# Claude Code 高效使用指南

## 1. 核心指令（Slash Commands）

### 基础指令

| 指令 | 作用 | 使用场景 |
|------|------|----------|
| `/help` | 显示帮助信息 | 忘记某个指令用法时 |
| `/clear` | 清空当前对话上下文 | 话题切换，释放上下文窗口 |
| `/compact` | 压缩对话历史，保留关键信息 | 对话很长时主动压缩，避免 token 溢出 |
| `/init` | 分析代码库生成 CLAUDE.md | 新项目开始时，让 Claude 理解项目结构 |
| `/config` | 打开配置面板 | 修改主题、模型、权限等设置 |
| `/context` | 显示当前上下文使用情况 | 监控 token 消耗，判断是否需要 compact |

### 工作流指令

| 指令 | 作用 | 使用场景 |
|------|------|----------|
| `/plan` | 进入计划模式，先设计再编码 | 复杂任务开始前，让 Claude 先出方案再动手 |
| `/review` | 代码审查当前分支变更 | PR 提交前的自检 |
| `/security-review` | 安全审查当前分支 | 涉及权限、敏感数据的变更 |
| `/simplify` | 审查代码质量并自动修复 | 重构后检查冗余、改进写法 |
| `/loop` | 按间隔重复执行指令 | 持续监控、定时任务（如 `/loop 5m /review`） |

### 权限与配置

| 指令 | 作用 |
|------|------|
| `/permissions` | 管理工具调用权限（允许/拒绝 Bash、网络等） |
| `/add-permissions` | 添加新的权限规则 |
| `/hooks` | 管理事件钩子（如提交前、文件保存后触发） |

---

## 2. Agent 多智能体协作

Claude Code 支持启动多个子 Agent 并行工作。这是最高效的使用模式之一。

### Agent 类型

| 类型 | 用途 | 典型任务 |
|------|------|----------|
| `general-purpose` | 通用 Agent | 复杂的多步搜索、代码研究 |
| `Explore` | 只读代码探索 | 搜索文件、grep 符号、定位定义 |
| `Plan` | 软件架构设计 | 实现方案设计、架构决策 |
| `claude-code-guide` | Claude Code 使用问题 | 回答 CLI 功能、SDK、API 相关问题 |
| `code-reviewer` | 代码审查 | 独立的第二意见审查 |

### 并行 Agent 模式（关键技巧）

当有多个**独立任务**时，在单条消息中同时启动多个 Agent：

```
# 示例：同时做三件独立的事
Agent A: "搜索所有 YOLO 相关的配置文件"
Agent B: "搜索所有 PatchCore 模型加载代码"
Agent C: "检查 tests/ 目录下的测试覆盖情况"
```

这三个 Agent 会**并发运行**，结果一起返回，大幅缩短总耗时。

### 背景 Agent

用 `run_in_background: true` 将非关键路径的任务放到后台：

```
Agent: "跑完整的测试套件"  // 后台运行
// 你可以继续其他工作，测试完成后会自动通知
```

### Agent 协作模式

1. **研究-执行分离**：先让 Explore Agent 搜索代码库，收集上下文，然后由主对话执行修改
2. **独立审查**：写完代码后，`code-reviewer` Agent 以全新视角审查（它看不到你的分析过程，给出真正的独立意见）
3. **分工并行**：重构时，A Agent 改前端，B Agent 改后端，互不阻塞
4. **Worktree 隔离**：给 Agent 设置 `isolation: "worktree"`，让它在独立的 git worktree 中工作，完全隔离文件变更

---

## 3. Memory 记忆系统

Claude Code 支持持久化记忆，跨对话保留。

### 记忆类型

| 类型 | 内容 | 示例 |
|------|------|------|
| `user` | 你的角色、偏好、知识背景 | "我是后端工程师，刚接触前端" |
| `feedback` | 你对 Claude 行为的纠正和确认 | "不要 mock 数据库做测试" |
| `project` | 项目背景、决策、截止日期 | "合并冻结期从 3月5日 开始" |
| `reference` | 外部资源指针 | "Bug 跟踪在 Linear 项目 INGEST" |

### 用法

- 直接说 "记住 XXX" → Claude 自动保存为合适的记忆类型
- 说 "忘记 XXX" → Claude 找到并删除相关记忆
- 记忆存储在 `~/.claude/projects/<project-path>/memory/` 下

### 注意

记忆适合保存**持久的偏好和背景**，不要用来保存代码模式、文件路径（这些可以通过读代码获得）或临时任务状态。

---

## 4. Skills 技能系统

Skills 是可调用的专用能力模块，通过 `/skill-name` 触发。

| Skill | 作用 |
|-------|------|
| `/update-config` | 修改 settings.json（权限、环境变量、钩子） |
| `/keybindings-help` | 自定义键盘快捷键 |
| `/simplify` | 代码质量审查 + 自动修复 |
| `/fewer-permission-prompts` | 分析历史记录，自动添加权限白名单减少弹窗 |
| `/loop` | 定时重复执行命令 |
| `/init` | 初始化 CLAUDE.md |
| `/review` | PR 审查 |
| `/security-review` | 安全审查 |
| `/claude-api` | Claude API/SDK 开发辅助 |

---

## 5. Plan Mode 计划模式

适合**复杂、多文件、有架构决策**的任务。

### 何时用

- 新功能实现（不知道改动范围）
- 有多种可行方案（需要比较权衡）
- 影响现有架构的变更
- 用户说"先设计一下"时

### 何时不用

- 修一个 typo
- 改一个明显的单行 bug
- 需求已经非常具体的简单任务

### 流程

1. 你提出任务，Claude 进入 Plan Mode
2. Claude 探索代码库，设计方案
3. Claude 写出计划文件供你审查
4. 你批准后，Claude 退出 Plan Mode 并开始实现

---

## 6. Worktree 工作树隔离

当需要做**高风险变更**或**并行开发**时，使用 worktree 隔离。

```
// 启动一个隔离的 worktree
"在 worktree 中重构 PatchCore 模块"
```

- 所有文件变更隔离在独立目录
- 不影响主工作区
- 完成后可以选择保留（keep）或删除（remove）
- Agent 也可以用 `isolation: "worktree"` 获得隔离环境

---

## 7. 高效使用技巧

### 上下文管理

1. **及时 `/compact`**：当对话超过 50 轮或 token 使用率 > 70% 时主动压缩
2. **用 Explore Agent 搜索**：不要在主对话中 `grep` 和 `cat` 几十个文件，让 Explore Agent 搜索后只返回摘要
3. **`/context` 监控**：定期检查 token 消耗，了解对话还能走多远

### 工具调用

1. **批量并行调用**：多个独立操作（读 3 个文件、跑 2 个命令）放在一条消息中，会并发执行
2. **使用专用工具**：用 `Read`/`Edit`/`Write` 而非 `cat`/`sed`/`echo`，后者需要额外权限确认
3. **权限白名单**：对常用命令运行 `/fewer-permission-prompts`，减少审批弹窗

### 代码生成

1. **先 Plan 后写**：复杂任务先用 Plan Mode 得到方案批准，避免方向错误导致返工
2. **写完用 `/simplify`**：让 Claude 审查自己的代码，检查冗余和质量问题
3. **提交前 `/review`**：检查变更的完整性和正确性
4. **安全相关用 `/security-review`**：涉及认证、输入校验、SQL 的变更

### 多 Agent 编排

```
// 高效模式示例：
1. 同时启动 3 个 Explore Agent 搜索不同模块
2. 汇总搜索结果，制定修改方案
3. 启动 2 个 Agent 并行修改不同文件
4. 用 code-reviewer Agent 独立审查结果
5. 主对话整合所有产出
```

### 项目配置

1. **CLAUDE.md**：写好项目架构说明，新对话的 Claude 会首先读取
2. **settings.json Hooks**：配置自动化行为（如"提交前跑 lint"、"保存后格式化"）
3. **Memory**：把重要的项目决策和偏好持久化

---

## 8. 快捷键（默认）

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+C` | 中断当前生成 |
| `Ctrl+D` | 退出 Claude Code |
| `Ctrl+L` | 清屏 |
| `Ctrl+R` | 搜索历史命令 |
| `↑/↓` | 浏览历史消息 |
| `Tab` | 自动补全路径 |

可通过 `/keybindings-help` 自定义。
