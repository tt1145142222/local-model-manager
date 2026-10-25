# 架构与公开范围

```text
Codex / MCP 客户端                 现有 Python / CLI 工作流
       │ mcp_server + lite_mcp              │ manager
       └──────────────── client ────────────┘
                           │ 本地鉴权 RPC
                         service
                           │
                         broker ─── SQLite（work/state）
                   ┌───────┼──────────┐
             resources   adapters   broker_adaptive
               遥测      后端启停       adaptive / learner
                   └───────┬──────────┘
                       app_memory
                 有条件的闲置工作集回收
```

后台服务独立于单个 MCP 会话，同一配置目录使用文件锁限制实例。持久化租约、事件、采样和策略；中断的生命周期操作先核对实际状态，不能盲目重放。

## 主要模块

| 文件 | 职责 |
| --- | --- |
| `manager.py` / `client.py` | CLI、命令包装器、后台服务连接 |
| `service.py` / `broker.py` | loopback 鉴权、单写者状态、资源凭证、调度与恢复 |
| `manager_control.py` | 持久用户停用标记与获准已有任务的续接校验 |
| `resources.py` | Windows RAM/提交量、NVIDIA 遥测、调度延迟代理 |
| `adapters.py` / `registry.py` | ComfyUI 与 llama.cpp 身份/忙闲/启动/停止和严格清单登记 |
| `adaptive.py` / `broker_adaptive.py` / `learner.py` | 任务规格、预算、采样、反馈、有界学习和回退 |
| `protection.py` / `health_report.py` | 下降趋势保护、稳定恢复和健康报告 |
| `app_memory.py` | 独立监测闲置应用，满足全部条件才调用 EmptyWorkingSet |
| `mcp_server.py` / `lite_mcp.py` | 11 个资源工具及精简 MCP stdio |

## 本次公开快照

本项目来自 2026-10-25 安装的 v1.2.0。根目录运行 Python 模块按字节保留；公开包重写文档，拆出空模型的 `config.example.json` 和公开测试夹具，清理个人运行数据，调整测试夹具引用与可选测试路径。`.cmd` 入口改为仓库相对路径。

源安装的依赖清单带有历史 MCP SDK 及辅助包。本次根据实际导入关系将运行依赖收敛为 `psutil`；可选 SDK 互操作测试依赖单列在 `requirements-dev.txt`。未复制虚拟环境或模型后端依赖。

原 H3/Muse 专用分支保留在 `adapters.py`，但其外部启动器、模型和密钥不属于此仓库。默认不注册任何真实模型，避免在他人电脑上自动使用作者的路径。

## 实验性 Transformers 文件

`transformers_adapter.py`、`transformers_manifest.py`、`transformers_worker.py`、`worker_protocol.py` 是原安装中存在的 Qwen2-Audio 文件队列实验代码，因此随最新源码快照保留。它们**没有接入当前 Broker 的标准适配器工厂和通用 register 流程**，不在本次正式支持/集成验证范围。

其清单校验绑定特定模型 revision、下载回执和 worker 哈希。不要把它作为通用 Transformers 接入器使用，也不要直接运行 GPU worker 绕过资源协调。若要产品化，需要补齐登记、适配器选择、身份/忙闲/生命周期测试以及真实任务验证。主运行环境无需安装 torch 或 transformers。

## 保留质量与可解释性

资源策略只调整准入、等待、管家拥有的进程调度和可安全回收的缓存，不修改模型权重或生成参数。按模型与任务签名隔离经验；缺失原生 busy 证据或采样断档时，窗口峰值不能冒充可用于预测的推理峰值。

默认桌面底线为可用 RAM 12 GiB、系统提交余量 12 GiB、空闲显存 3 GiB；压力和经验可增加保护。已准备任务目标采样 0.5 秒、其他活动 1 秒、空闲 10 秒，实际周期受接口耗时影响。后台 RSS 96 MiB 是软目标，不是硬上限，也不包括所有 MCP 连接的总开销。
