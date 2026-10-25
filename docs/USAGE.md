# 使用与集成说明

先完成根目录 README 的安装和模型登记。本页所有 CLI 命令均在仓库目录执行，使用 `.venv/Scripts/python.exe manager.py`。

## 资源任务生命周期

```text
status → request → lease（等待 active / proceed）
                    ↓
           prepare（本次 proceed=true）
                    ↓
             原工作流提交推理
                    ↓
            原工作流完成与质量检查
                    ↓
                 complete
                    ↓
        下一项 prepare，或整批 release
```

`request` 返回的 `id` 和 `token` 是该任务的凭证。保存在本地私有任务状态，不写进聊天公开内容、issue 或日志。`request_key` 在同一申请的重试间保持稳定；每个推理使用独立且重试稳定的 `work_id`。

`active` 不单独代表可以提交，必须同时检查 `proceed`。每个实际推理前还需 `prepare`。`retryable=false` / `policy_infeasible` 表示当前预算超出已知容量，应结束等待、释放凭证并检查预算/硬件，不能不断换请求 ID。

等待或执行期间定期调用 `lease` 续期；默认 TTL 为 3600 秒，建议不超过 5 分钟一次。`run` 包装器自动管理续期。同批任务可以复用凭证，结束后释放；`keep_loaded` 只允许短暂保温，实际时长由策略决定，最高 120 秒。

## CLI 命令

| 操作 | 命令（省略解释器和 manager.py） |
| --- | --- |
| 查看资源与队列 | `status` |
| 健康报告 | `health-report` |
| 应用内存回收状态 | `app-memory` |
| 历史事件 | `history` |
| 学习记录 | `learning --model my-llm` |
| 申请任务 | `request my-llm --owner demo --key batch-001 --task-file work/task.json` |
| 查询并续期 | `lease <id> <token>` |
| 本次推理准入 | `prepare <id> <token> item-001 --task-file work/task.json` |
| 完成记录 | `complete <id> <token> item-001 --outcome completed --quality unknown` |
| 释放本任务 | `release <id> <token>` |
| 后补质量反馈 | `feedback <id> <token> item-001 --feedback-id qa-001 --quality passed` |
| 策略回退 | `rollback <id> <token> item-001 --target-revision <revision> --reason <reason>` |
| 释放闲置模型 | `release-idle my-llm` |
| 关闭后台管家 | `shutdown` |

这里的 `<id>`、`<token>`、`<revision>` 等是占位符，不要原样输入。命令行可能进入本机 shell 历史；自动集成优先通过 MCP 传凭证。不要把完整凭证命令分享到网上。

`release-idle` 默认不停止外部启动的服务。只有明确授权停止具体服务且确认影响后，才可添加 `--allow-external`。`shutdown` 拒绝在仍有活动/排队任务或清理操作时执行，并保留模型服务。

## MCP 工具与 CLI 的关系

| MCP 工具 | 对应功能 |
| --- | --- |
| `local_models_status` | 总体状态，含 health_report 与 app_memory_reclaim |
| `local_models_request` | 申请资源凭证 |
| `local_models_lease` | 查询及续期 |
| `local_models_prepare` | 每次推理前检查 |
| `local_models_complete` | 记录完成、峰值证据和 QA |
| `local_models_feedback` | 补充真实反馈 |
| `local_models_learning` | 查看学习记录 |
| `local_models_release` | 释放本任务凭证 |
| `local_models_release_idle` | 受控释放闲置模型 |
| `local_models_register` | 登记兼容后端 |
| `local_models_history` | 查询事件 |

精简 stdio 层只实现初始化、ping、通知和工具调用；不提供通用 MCP 资源、提示词或其他扩展能力。取消 MCP 请求不会抢占模型生命周期操作。客户端连接旧版时，应重新加载连接；CLI 与 MCP 使用同一个后台服务。

## 任务规格和 QA

- 视频任务传实际 `width`、`height`、`frames`、`batch_size`。
- LLM 任务传实际 `input_tokens`、`max_output_tokens`、`context_tokens`，可加图片数量等支持字段。
- `workflow_signature` 应覆盖模型/检查点、精度、采样参数、视觉条件、工作流结构等影响负载或质量的设置，设置变化后更新。
- 可靠时可传 `estimated_peak_ram_gib`、`estimated_peak_commit_gib`、`estimated_peak_vram_gib`，代表桌面余量以外的额外空间；不要编造估计以获得准入。
- 规格缺失使用保守规则；不能把其他模型或其他工作流的经验冒充本任务的测量。

`--qa-file` 接受本次工作流新生成的 JSON，例如：

```json
{"outcome":"completed","quality":"unknown","desktop_responsive":null}
```

`outcome` 可为 `completed`、`oom`、`failed`、`cancelled`；`quality` 可为 `passed`、`failed`、`unknown`。没有实际质量验收就用 `unknown`；没有直接桌面体验证据就用 `null`。旧文件、无效文件或仅退出码 0 都不证明质量通过。

## 常见问题

**提示已由用户停用？** 最新版识别 `work/state/disabled.json` 中的持久停用标记，状态中通过 `control` 描述限制。标记存在时，普通客户端不会自动启动管家，新任务被拒绝；允许续接的已有任务仍保留原质量约束。当前 CLI 没有通用 enable/disable 子命令，标记与已有任务白名单由部署维护流程管理；不要为绕过报错自动删除标记。公开包不携带作者机器上的停用标记或任务状态，新安装不继承个人运行状态。

**一直排队？** 查看 `status` 中的原因与 `health-report`。可能是别的任务占用、RAM/提交量/显存不足、后端状态不明、遥测过期或恢复稳定窗口尚未满足。不要清空其他任务或杀进程来跳过队列。

**内存空闲但提交量不足？** Windows 提交量是另一项约束；工作集回收不能解决提交容量不足。先依据报告识别真实限制。

**已经最小化，为什么不回收？** 还需要连续低 CPU/I/O、足够工作集、同用户会话、无受保护进程族、确有持续模型 RAM 缺口，并满足冷却。保守跳过是预期行为。

**占用端口但看不到模型？** 管家要求进程路径、启动参数、监听端口与忙闲接口相符。未知监听者不会被接管；先检查端口归属和清单，不要批量结束进程。

**自己的 H3/Muse 无法直接用？** 原版保留 ID 依赖特定安装目录结构、启动器和模型文件。公开默认配置不注册它们。按实际后端创建新的 ComfyUI/llama.cpp 清单，保留自己的启动参数；需要专用保护的启动流程须先完成适配验证。

**能否接入任意模型？** 仅协议兼容还不够，必须能核验身份、忙闲与生命周期。新后端要实现并测试适配器；实验性的 Transformers 文件不等于已经接入生产调度。
