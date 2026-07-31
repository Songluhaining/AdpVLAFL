# AdpVLAFL — 面向 LingBot-VLA 2.0 的失败定位与决策端优化

在 RoboTwin 2.0 仿真中大规模采集 LingBot-VLA 2.0 的闭环执行数据（含模型内部量），
用贝叶斯网络做失败归因，目标是定位决策过程中导致失败的环节并加以优化。
方法要求跨任务通用，并最终迁移到真实机器人。

上游模型代码：[Robbyant/lingbot-vla-v2](https://github.com/robbyant/lingbot-vla-v2)
仿真环境：[RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)

## 目录

| 目录 | 内容 |
|---|---|
| `setup/` | 两个 conda 环境的搭建、curobo 编译、权重与仿真资产下载（一次性） |
| `collect/` | 采集编排：单次评测、全任务横扫、深度采集、干预实验 |
| `analysis/` | 值节点提取、归约方式筛选、分层贝叶斯拟合、作图、性能基准 |
| `robotwin/` | RoboTwin 侧的 rollout 客户端（放入 `RoboTwin/script/`） |
| `lingbot_patch/` | 对上游 lingbot-vla-v2 的埋点改动 |

数据（rollouts、权重、日志）不入库，见 `.gitignore`。

## 架构

策略端与仿真端是**两个进程、两个 conda 环境**，只通过 websocket 通信：

- `lingbotvla`：py3.12 / torch 2.8 / flash-attn，常驻 6B 模型（bf16 约 12.9GB 显存）
- `robotwin`：py3.10 / torch 2.4.1 / SAPIEN 3 / curobo（约 5.4GB 显存）

上游 README 要求把两套依赖合并到一个环境，实测**没有必要**，拆开可避免 torch 版本冲突。
一张 24GB 显卡只放得下一份（合计约 18GB），并发跑第二份必然 OOM。

## 埋点

`lingbot_patch/` 通过一个旁路注册表采集推理内部量，不改动任何函数签名（训练路径不受影响）：

| 字段 | 形状 | 用途 |
|---|---|---|
| `intro_noise` | (D, 50, 55) | flow matching 采样噪声 |
| `intro_h_query` / `h_query_tokens` | (D, 2560) / (D, 8, 2560) | 8 个蒸馏查询位输出 → 感知观测 |
| `intro_h_image` / `h_lang` | (D, 2560) | 图像 / 语言 token 池化 |
| `intro_router_counts` | (D, 10, 36, 32) | 逐去噪步 × 36 层 × 32 专家负载 |
| `intro_router_entropy` | (D, 10, 36) | 逐层路由熵 |
| `intro_denoise_x` | (D, 10, 50, 55) | 完整去噪轨迹 |

全部只用模型前向可算的量，**不依赖仿真器特权状态**，因此可原样迁移到真机。
开销约 +21%（465ms vs 383ms/次决策）。

安装：
```bash
cd /path/to/lingbot-vla-v2
cp /path/to/AdpVLAFL/lingbot_patch/introspect.py lingbotvla/models/vla/lingbot_vla/
git apply /path/to/AdpVLAFL/lingbot_patch/instrumentation.patch
```

## 已确立的结论

**任务难度与时程无关。** 50 个任务全扫（每任务 12 episode，`demo_randomized`）后，
一半任务满分；步数上限最长的 `put_bottles_dustbin`(1700) 满分，而 400 步的
`click_alarmclock` 只有 41.7%。按时程猜难度是错的。

**执行可达性不是失败模式。** 手臂跟踪误差恒为 0——仿真器用 `set_arm_joints` 直接设定
关节位置，指令必然精确到达。该分支已从因子图删除。

**去噪速度场不衰减。** `‖v_t‖` 随步数上升而非下降，因为 flow matching 走的是近似直线；
"收敛度"对这个模型没有意义。

**值节点定义决定结论。** MoE 路由用「专家负载距离」筛选 AUC 0.50（零信号），
换成「路由熵」后 AUC 0.64。同一组件、同一批数据，测量方式错了就得出"该组件无关"的假阴性。

**bf16 精度下无法完全复现。** 固定噪声种子后残差仍有 1.3e-2，约等于 bf16 相对精度
(0.3%)，`torch.use_deterministic_algorithms` 无效；fp32 需 25GB 显存，4090 装不下。
因此干预实验采用三臂设计（基线 / 空白臂 / 干预），空白臂用于测量数值底噪。

**组件作用是任务特异的。** 分层模型（b1 按任务部分池化）LOO 显著优于池化
(elpd 差 12.2, dse 4.9)，且 `tau_routing` 的 94% HDI = [1.06, 3.77] 排除 0——
路由的作用强烈随任务变化。留出任务 AUC 仅 0.54，当前 5 组件抽象**不具备跨任务通用性**。

## 流程

```
setup/            环境与数据（一次性）
  ↓
collect/sweep_all_tasks.sh      全任务横扫，筛出有失败率的任务
collect/collect.sh              对难任务深度采集
  ↓
analysis/screen_reductions.py   筛选值节点的归约方式（按任务内 AUC）
analysis/extract_node_values.py 提取每组件每样本的观测值
analysis/fit_hierarchical.py    分层 noisy-OR + NUTS，输出组件效应与通用性 tau
  ↓
collect/intervene_exec_horizon.sh  干预验证（同场景同噪声，只改一个旋钮）
```
