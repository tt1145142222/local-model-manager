# 内存管家 · Local Model Manager

**让本地 AI 模型按资源余量排队运行，同时给日常办公留出空间。**

当前开源版本：**v1.2.0**。面向 **Windows + NVIDIA GPU**，提供命令行和 MCP 接口，可接入 Codex 等支持 MCP 的客户端。它是后台资源协调工具，目前没有图形界面。

## 它能帮你做什么？

例如，你正在用 ComfyUI 生成视频，另一个聊天又想启动本地大语言模型。两边各自启动容易争抢内存和显存。内存管家让接入的任务先申请资源，条件不够时排队，每次推理前重新检查，完成后释放资源。任务始终保留原有分辨率、步数、模型精度和上下文要求。

| 能力 | 具体用途 |
| --- | --- |
| RAM、系统提交量、显存监测 | 分清哪一种容量不足，避免把物理内存回收当成显存扩容 |
| 跨聊天资源排队 | 多个客户端共享一个后台服务，通过凭证协调已登记的模型 |
| 推理前准入检查 | 根据实际任务规格、已有测量和桌面余量决定是否允许新提交 |
| 受控启停 | 核对进程身份和后端忙闲状态；忙碌或身份不明时拒绝停止 |
| 有界自适应 | 按模型与工作流分别积累峰值、耗时和质量反馈；遇到压力增加保护，可回退策略 |
| 闲置应用物理内存回收 | v1.2.0 新增；真实模型任务持续缺 RAM 时，对符合条件的最小化应用执行有限的工作集回收 |
| 状态、历史与健康报告 | 解释为什么等待、是否有观测缺失、哪些经验可以用于容量预测 |

适合在同一台 Windows 电脑上运行本地视频模型、ComfyUI 和 llama.cpp，同时还需要浏览器、编辑器、录制或办公的人。**仅安装它不会自动接管其他软件；任务必须走 MCP、CLI 包装器或资源凭证接口。**

## 能力边界

- 当前调度以 **NVIDIA GPU 0** 为准；没有多 GPU 调度、AMD/macOS/Linux 完整运行支持。缺少必要遥测时会阻止准入。
- 它不能增加硬件容量、硬隔离显存或抢占正在执行的推理，也不能阻止绕过管家的客户端抢资源。
- 不会为了放行任务而自动降低生成质量。预算超过硬件能力时会报告 `policy_infeasible`，而不是无限等待或偷偷缩减预算。
- 闲置回收调用 Windows `EmptyWorkingSet`，不关闭、暂停或重启应用。它移出可分页的物理驻留页，**不减少私有提交量，不释放显存**；切回应用可能重新调页，不能保证提速。
- 状态采样、CPU 调度延迟和自动测试不等于人工确认桌面流畅，也不等于生成质量验收。
- 项目不包含模型权重、ComfyUI、llama.cpp 或个人安装环境；这些软件须自行安装并遵守各自许可证。

## 安装与首次运行

需要 Windows、Python 3.12（本次验证版本为 3.12.14）、Git，以及可运行 `nvidia-smi` 的 NVIDIA 驱动。普通用户权限即可。以下命令在 PowerShell 中执行：

```powershell
git clone https://github.com/tt1145142222/local-model-manager.git
cd local-model-manager
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item config.example.json config.json
.\.venv\Scripts\python.exe manager.py status
```

`status` 会在需要时自动启动轻量后台管家，不会启动模型。首次配置的 `models` 为空，这是预期状态。请只运行一个管家实例；默认地址为 `127.0.0.1:18765`，若本机已有实例，应复用它，或先在没有活动任务时正常关闭旧实例。

运行时仅需要 `psutil`；MCP stdio 层由本项目的标准库实现提供，无需额外安装 MCP SDK。启动后，凭证、SQLite 状态和日志保存在仓库下的 `work/`，已加入忽略规则。

### 1. 登记你已安装的模型

先复制 `examples/llamacpp.manifest.json` 或 `examples/comfyui.manifest.json` 到自己的工作目录，修改：

1. `root`、`exe` 和 `start_argv` 中的所有路径：必须指向真实文件。
2. `url` 与启动端口：保持一致，只允许 `127.0.0.1`。
3. 启动参数：沿用你已验证可运行的模型、精度、采样及上下文配置。示例不代替模型安装说明。
4. `startup_*_gib`：按实际模型和硬件设置保守启动阈值；示例值不是通用最低配置。不要为强行放行而调低预算。

llama.cpp 需暴露 `/health` 和 `/slots`，使用单推理槽 `-np 1`，且 `-c` 与 `context_capacity` 一致。ComfyUI 需提供 `/queue`。示例中的 `model.gguf` 是占位名称。

```powershell
New-Item -ItemType Directory -Path work -Force
Copy-Item examples/llamacpp.manifest.json work/my-llm.json
# 编辑 work/my-llm.json，填写实际路径、启动参数及预算后执行：
.\.venv\Scripts\python.exe manager.py register --model my-llm --manifest work/my-llm.json
```

登记只验证并保存配置，不启动模型。登记时不能有排队或活动任务。`h3` 和 `muse` 是原安装环境的保留 ID；新安装请使用自己的名称，例如 `my-llm`、`my-video`。

### 2. 接入 Codex / MCP 客户端

将以下条目合并到客户端的 MCP 配置，替换路径，不要覆盖已有服务器配置：

```toml
[mcp_servers.local_models]
command = "C:/Tools/local-model-manager/.venv/Scripts/python.exe"
args = ["C:/Tools/local-model-manager/mcp_server.py"]
```

其他 MCP 客户端使用相同的 stdio 命令和参数。重新加载连接后，让助手遵循以下资源协议：

> 对已登记模型，先 status，再 request；仅在 active 且 proceed=true 时继续。每次实际推理前 prepare；只有本次 prepare 返回 proceed=true 才提交。长任务定期 lease 续期；完成推理与原流程质量检查后 complete，整批结束 release。保留原质量参数，quality 未评估就填 unknown。外部服务停止需要针对具体对象的授权。

**MCP 管家不会替你调用模型推理接口。** 原有 ComfyUI/LLM 工作流继续执行生成，管家负责允许何时执行和记录结果。完整工具映射见 [使用与集成说明](docs/USAGE.md)。

### 3. 用一条命令包装现有脚本

复制并修改 `examples/task-llm.json`，填写本次实际 token/上下文规模，以及能标识模型、精度、采样配置的 `workflow_signature`：

```powershell
Copy-Item examples/task-llm.json work/task.json
.\.venv\Scripts\python.exe manager.py run --model my-llm --owner demo --task-file work/task.json -- C:/YourPython/python.exe C:/YourProject/infer.py
```

包装器自动申请、等待、prepare、续期、记录完成并释放。脚本退出成功只表示执行完成，质量仍为 `unknown`。若原流程产生本次 QA 文件，可加 `--qa-file work/qa.json`；不要预造 `passed`。

**一个脚本内部多次提交推理时，每次都需要单独 prepare/complete**；外层 `run` 只检查命令启动，不能替代每次推理检查。

### 4. 查看状态、回收记录与停机

```powershell
.\.venv\Scripts\python.exe manager.py status
.\.venv\Scripts\python.exe manager.py health-report
.\.venv\Scripts\python.exe manager.py app-memory
.\.venv\Scripts\python.exe manager.py history
.\.venv\Scripts\python.exe manager.py learning
# 没有排队或活动任务时正常关闭管家，保留模型服务：
.\.venv\Scripts\python.exe manager.py shutdown
```

## v1.2.0 的闲置内存回收怎么工作？

公开示例配置中 `app_memory_reclaim.enabled` 为 `false`。需要此功能时，在无任务时正常关闭管家，将其改为 `true` 后重新启动。启用后，只有有效模型任务持续 **15 秒** 缺少 RAM，才考虑同一用户/桌面会话内：所有可见窗口都已最小化、连续 **10 分钟** CPU/I/O 活动低、工作集至少 **256 MiB** 的应用。

前台程序及其进程族、模型和任务进程、Codex、常见办公/编辑/录制/媒体应用、脚本运行时受保护；无窗口服务、托盘程序、身份不明或权限不足的目标跳过。每次最多处理一个应用的 4 个进程，全局间隔至少 60 秒，同一应用冷却 30 分钟，一次持续压力最多 3 次。执行前还会重查身份、活动与 RAM 需求。

关闭方法：在无任务时执行 `shutdown`，将 `config.json` 中 `app_memory_reclaim.enabled` 改为 `false`，再执行 `status` 启动。不要直接修改正在运行的配置并假定已经生效。

最新快照还包含持久停用控制：若部署目录存在明确的用户停用标记，普通调用不会自动重启管家或接受新任务；已获准保留的任务可继续完成。停用控制与仅关闭应用回收不同，详见 [使用说明](docs/USAGE.md)。

## 测试与开发

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s . -p "test_*.py" -v
```

默认测试使用隔离的临时目录、模拟后端和公开测试配置，不启动真实模型。`smoke_*.py` 属于按需运行的本机集成检查，其中部分会申请资源或启停模型，不应当作默认测试批量执行。测试范围与本次结果见 [验证记录](docs/VALIDATION.md)。

- [使用与集成说明](docs/USAGE.md)：CLI、MCP、任务生命周期和故障排查。
- [架构与公开范围](docs/ARCHITECTURE.md)：模块分工、配置和实验性文件边界。
- [更新记录](CHANGELOG.md)
- [安全与隐私](SECURITY.md)

## 许可证

本项目源码以 [MIT License](LICENSE) 开源，允许使用、修改、分发和商业使用，须保留许可证与版权声明。外部模型、软件及其素材的授权不由本许可证授予。
