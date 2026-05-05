# CNN 蜡烛图案特定策略集成指南

## 概述

Trading_AgenticAI_GAF_CNN_DQN_Demo 已升级，现在在 DQN/QRL 决策中加入 **8 类蜡烛图案的特定策略建议**。

## 蜡烛图案到策略的映射

| 模式 | 建议动作 | 信心阈值 | 原因 |
|-----|--------|---------|------|
| **doji** | flat | 0.70 | 不确定 - 等待确认 |
| **hammer** | long | 0.60 | 看涨反转模式 |
| **hanging_man** | short | 0.60 | 看跌反转模式 |
| **shooting_star** | short | 0.70 | 强烈看跌信号 |
| **bullish_engulfing** | long | 0.50 | 强烈看涨反转 |
| **bearish_engulfing** | short | 0.50 | 强烈看跌反转 |
| **morning_star** | long | 0.60 | 看涨三柱线模式 |
| **evening_star** | short | 0.60 | 看跌三柱线模式 |
| **unknown** | flat | 0.00 | 模式不清楚 - 保持中性 |

## 工作流程

### 1. CNN 模式识别
```
OHLC 数据 → GAF 编码 → CNN 分类 → 获得 9 类概率
↓
提取最高概率 → 识别模式和信心
```

### 2. DQN/QRL 决策 + 模式校验
```
状态向量（12 个返回 + 9 个模式概率 + 1 个信心）
↓
DQN 神经网络 → 3 个 Q 值 (short/flat/long)
↓
选择最高 Q 值 → DQN 动作
↓
+ 检查 CNN 模式建议 → 检测冲突
↓
输出：最终动作 + 模式信息 + 对齐状态
```

### 3. 输出结构

每个策略建议现在包含：

```json
{
  "action": "flat",  // DQN 选择的动作
  "q_values": {      // 3 个 Q 值
    "short": 0.093,
    "flat": 0.119,
    "long": 0.068
  },
  "source": "dqn_only",  // 信息来源
  "pattern_info": {      // 新增：模式特定信息
    "pattern": "doji",   // 识别的蜡烛图案
    "confidence": 0.709, // CNN 的信心分数 (0-1)
    "suggested_action": "flat",  // 模式建议的动作
    "is_reliable": true,         // 信心是否超过阈值
    "reason": "Indecision - wait for confirmation"
  },
  "alignment": "pattern_and_dqn_agree"  // 动作对齐状态
}
```

## 冲突检测

当 DQN 和模式建议冲突时（模式信心高，但建议不同），输出会包含：

```json
{
  "conflict_detected": true,
  "conflict_action": "long",
  "conflict_note": "Pattern hammer (confidence=0.75) suggests long, but DQN chose flat"
}
```

## 使用示例

### 读取并分析最新的策略建议

```python
import json
from pathlib import Path

# 加载最新的策略上下文
context = json.loads(Path("results/agentic_context.json").read_text())

# 获取最新的 DQN 策略
strategy = context["latest_dqn_strategy"]

# 检查模式信息
if "pattern_info" in strategy:
    pattern = strategy["pattern_info"]
    print(f"识别的模式: {pattern['pattern']}")
    print(f"信心分数: {pattern['confidence']:.3f}")
    print(f"建议动作: {pattern['suggested_action']}")
    print(f"是否可靠: {pattern['is_reliable']}")
    print(f"原因: {pattern['reason']}")

# 检查冲突
if "conflict_detected" in strategy and strategy["conflict_detected"]:
    print(f"⚠️ 冲突检测: {strategy['conflict_note']}")
else:
    print(f"✅ 对齐状态: {strategy.get('alignment', 'unknown')}")
```

## 修改策略阈值

要调整模式特定的信心阈值或建议动作，编辑 `StrategyAgent` 类中的 `pattern_to_action` 字典：

```python
pattern_to_action = {
    "hammer": {
        "action": "long",
        "confidence_threshold": 0.60,  # 修改这里
        "reason": "Bullish reversal pattern"
    },
    # ... 其他模式
}
```

## CNN 信号表参考

生成的 `results/cnn_signal_table.csv` 包含所有训练和测试样本的模式识别结果：

- `cnn_pattern`：识别的模式名称
- `cnn_signal`：模式的偏见 (bearish/neutral/bullish)
- `cnn_confidence`：识别的信心分数
- `cnn_bearish`、`cnn_neutral`、`cnn_bullish`：三个偏见类别的概率

## 核心函数修改

### StrategyAgent.recommend_action()
现在接受两个可选参数：
- `cnn_pattern`: 识别的蜡烛图案名称
- `cnn_confidence`: CNN 的信心分数

### InterfaceAgent.build_context()
现在自动从 CNN 信号中提取模式信息并传递给策略代理。

## 下一步改进

1. **加权融合**：根据模式信心调整 Q 值
2. **历史学习**：跟踪特定模式的历史胜率
3. **多时间框架**：在不同 timeframe 上识别模式并融合决策
4. **强化学习反馈**：根据模式预测的准确性动态调整权重

## 测试

运行演示脚本以查看最新的策略和模式建议：

```bash
python Trading_AgenticAI_GAF_CNN_DQN_Demo.py
```

检查生成的文件：
- `results/agentic_context.json` - 完整的上下文和策略
- `results/cnn_signal_table.csv/html` - CNN 信号详细表
- `results/decision_explanations.csv/html` - 决策解释

---

**最后一个推荐**（来自最新运行）：
- CNN 识别: doji (信心: 0.709)
- DQN 动作: flat
- 风险决策: approve
- **最终**: FLAT (保持观望)
