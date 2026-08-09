# LingBot-VLA 2.0 完整流程（部署闭环 + 训练）

以本项目实测/代码确认为准（`lingbot_patch/` 所改动的上游代码即此流程的实现）。
图中 ⚑ 标注为本项目加入的扩展点。

## 部署闭环（每个决策）

```mermaid
flowchart TD
    subgraph SIM["仿真器 / 机器人（robotwin 环境）"]
        A[场景] --> B["观测快照<br/>3路相机 RGB（头/左腕/右腕）<br/>+ 14维关节角 + 语言指令"]
        EXEC["执行前 25 拍（exec_horizon）<br/>逐拍物理步进 + check_success"]
    end

    subgraph SRV["策略服务端（lingbotvla 环境，6B bf16 常驻）"]
        P1["① 预处理<br/>图像 resize 256×256 → 视觉 token<br/>指令分词；状态归一化（55 维规范空间）"]
        P2["② 前缀前向（每决策一次）<br/>Qwen3-VL 4B，36 层<br/>~198 视觉 + 24 蒸馏 query + ≤72 语言 token<br/>→ KV 缓存<br/>⚑ h_image/h_lang/h_query 埋点"]
        P3["③ 流匹配去噪 ×10 步<br/>动作专家 36 层 MoE（32 选 4）<br/>ε ~ N(0,I)，形状 50×55 ← 唯一随机输入<br/>t: 1.0→0.1，后缀 51 token 交叉注意前缀 KV<br/>欧拉步 x += −0.1·v_t<br/>⚑ noise_seed 钉死 ⚑ sampler=euler/vine<br/>⚑ 路由/去噪轨迹埋点 ⚑ LoRA 挂点"]
        P4["④ 反归一化（bounds_99）<br/>→ 50 拍 × 14 维关节目标<br/>⚑ sample_topn：N 抽签 + critic 择优"]
    end

    B -- "websocket (msgpack)" --> P1 --> P2 --> P3 --> P4
    P4 -- "websocket 返回" --> EXEC
    EXEC -- "新快照（无记忆，每决策独立）" --> B
```

- 一集 ≈ 4–12 个决策（快任务），直至成功或步数上限。
- 时延：官方 `--use_compile` ≈130ms/决策；本项目埋点配置（无 compile）≈0.5s。
- 策略**无记忆**：不看历史决策、不核对执行结果；每决策独立抽 ε。

## 训练流程（一次性，产出部署所用权重）

```mermaid
flowchart LR
    D["LeRobot 演示数据<br/>（多本体 → 55 维规范空间）"] --> N["归一化<br/>bounds_99 / meanstd"]
    N --> M["混合：抽 t~Beta(1.5,1)、抽 ε<br/>x_t = t·ε + (1−t)·动作"]
    M --> F["联合前向（与部署②③同构）<br/>前缀 = 图像+query+语言<br/>后缀 = 状态 + x_t"]
    F --> L1["流匹配回归<br/>u = ε − 动作（L1）"]
    F --> L2["蒸馏辅助损失<br/>当前/未来深度 + 未来视频<br/>（query token 读出 vs 冻结教师）"]
```

**要点**：成败信号不参与训练的任何环节（模仿学习）——这是本项目全部
负结果（RL 梯度、择优、定向抽签）的总根源；query token 的深度/动力学读出
在部署时是免费副产，被本项目用作状态特征。

## 本项目扩展点索引

| 扩展点 | 位置 | 用途 |
|---|---|---|
| introspect 埋点 | ②③ | 前缀读出、MoE 路由计数/熵、噪声、去噪轨迹 |
| noise_seed | ③ | 决策级噪声钉死（可复现/配对实验） |
| sampler=vine | ③ | VINE 终点预测采样器（arXiv 2607.10369） |
| --lora_path | ③ | 动作专家 LoRA 加载（RL 微调产物） |
| --critic_path + sample_topn | ④ | N 抽签 + 冻结裁判择优 |
| routing_bias | ③ | 路由分数定向偏置（阶段一矫正实验） |
