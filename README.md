# Agentic AI FX Trading Demo: GAF-CNN + DQN

This repository contains a teaching demo for an Agentic AI foreign exchange trading workflow. It converts OHLC candlestick windows into Gramian Angular Field (GAF) images, uses a CNN to classify candlestick patterns, feeds the CNN output into a compact DQN/QRL trading policy, and wraps the full workflow with agent-style roles for signal analysis, strategy selection, risk review, and explanation.

The project is designed for classroom use with University of Washington students.

> Disclaimer: This project is for education and research only. It is not financial advice, and it does not connect to a live brokerage or execute real trades.

## What This Demo Does

The main pipeline is:

```text
OHLC market data
-> 10-bar candlestick window
-> CULR representation
-> GAF image, shape 10 x 10 x 4
-> CNN candlestick pattern classifier
-> DQN/QRL state vector
-> short / flat / long action
-> backtest and risk review
-> Agentic AI explanation layer
```

The CNN predicts 9 classes:

```text
doji, hammer, hanging_man, shooting_star,
bullish_engulfing, bearish_engulfing,
morning_star, evening_star, unknown
```

The DQN policy uses:

- 12 recent returns
- 9 CNN candlestick pattern probabilities
- 1 CNN confidence score

The output action is one of:

```text
short, flat, long
```

## Repository Contents

| Path | Description |
|---|---|
| `Trading_AgenticAI_GAF_CNN_DQN_Demo.py` | Main Colab-ready demo script. Downloads FX data, builds GAF images, loads the CNN, trains the DQN, runs a backtest, and generates an agentic strategy recommendation. |
| `cnn_model_10bar.h5` | Pretrained CNN model used by default. Keep this file in the project root if you do not want to retrain the CNN. |
| `label8_eurusd_10bar_1500_500_val200_gaf_culr.pkl` | Preprocessed GAF-CNN training dataset for retraining the CNN. |
| `train_cnn_model.py` | Script for retraining the CNN model. It saves the updated model as `cnn_model_10bar.h5`. |
| `check_strategy.py` | Helper script for inspecting `results/agentic_context.json` if you modify the main demo to save output files. |
| `requirements-colab.txt` | Python package requirements for Colab or local execution. |
| `PATTERN_STRATEGY_GUIDE.md` | Notes on mapping CNN candlestick patterns to strategy recommendations. |
| `OPTIMIZATION_GUIDE.md` | Notes on DQN training and execution-parameter optimization. |
| `FinancialVision-master/` | Reference implementation material for GAF-CNN candlestick classification and deep reinforcement learning for FX trading. |

## Setup

You can run the project in Google Colab or in a local Python environment. Python 3.10 is recommended.

### Local Setup on Windows PowerShell

```powershell
cd C:\UW_AgenticAI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-colab.txt
```

### Local Setup on macOS or Linux

```bash
cd /path/to/UW_AgenticAI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-colab.txt
```

## Run the Demo with the Pretrained Model

If you do not want to retrain the CNN, keep this file in the project root:

```text
cnn_model_10bar.h5
```

Then run:

```bash
python Trading_AgenticAI_GAF_CNN_DQN_Demo.py
```

The script will:

1. Download recent `EURUSD=X` hourly OHLC data from Yahoo Finance.
2. Fall back to synthetic OHLC data if the download fails.
3. convert 10-bar candlestick windows into `10 x 10 x 4` GAF images.
4. Load `cnn_model_10bar.h5`.
5. Predict 9-class candlestick pattern probabilities.
6. Build DQN state vectors from recent returns, CNN probabilities, and CNN confidence.
7. Train a compact DQN policy.
8. Search execution parameters such as holding period and Q-value margins.
9. Backtest the selected policy.
10. Produce an agentic strategy recommendation and risk review.

The default market settings are near the top of `Trading_AgenticAI_GAF_CNN_DQN_Demo.py`:

```python
TICKER = "EURUSD=X"
PERIOD = "730d"
INTERVAL = "1h"
WINDOW = 10
HORIZON = 8
COST = 0.00002
```

Students can try other Yahoo Finance FX tickers, such as:

```python
TICKER = "GBPUSD=X"
TICKER = "AUDUSD=X"
```

## Use the Model in a Trading Strategy

The main script creates a structured strategy context near the end:

```python
agentic_context, agentic_llm_report = interface_agent.advise(
    raw,
    states[-1],
    summary,
    decision_explanations,
)
```

The most important fields are:

```python
strategy = agentic_context["latest_dqn_strategy"]
risk = agentic_context["latest_risk_review"]
final = agentic_context["final_recommendation"]

print(strategy["action"])        # short, flat, or long
print(strategy["q_values"])      # DQN Q-values for short / flat / long
print(strategy.get("pattern_info"))
print(risk["decision"])          # approve, reduce, or block
print(risk["suggested_position_size"])
print(final)
```

A simple downstream position-sizing rule could look like this:

```python
if risk["decision"] == "block":
    target_position = 0.0
elif strategy["action"] == "long":
    target_position = +risk["suggested_position_size"]
elif strategy["action"] == "short":
    target_position = -risk["suggested_position_size"]
else:
    target_position = 0.0
```

This converts the model recommendation into a target position:

```text
+1.0  full long
+0.5  reduced long
 0.0  flat
-0.5  reduced short
-1.0  full short
```

The current project does not place live orders. To connect it to a real trading system, students would need to add brokerage integration, order validation, position limits, stop-loss/take-profit rules, slippage assumptions, logging, and failure handling.

## CNN Pattern Strategy Rules

`StrategyAgent.pattern_to_action` maps each CNN candlestick pattern to a teaching recommendation:

| CNN Pattern | Suggested Action | Confidence Threshold | Interpretation |
|---|---:|---:|---|
| `doji` | `flat` | `0.70` | Indecision; wait for confirmation. |
| `hammer` | `long` | `0.60` | Bullish reversal pattern. |
| `hanging_man` | `short` | `0.60` | Bearish reversal pattern. |
| `shooting_star` | `short` | `0.70` | Strong bearish signal. |
| `bullish_engulfing` | `long` | `0.50` | Strong bullish reversal. |
| `bearish_engulfing` | `short` | `0.50` | Strong bearish reversal. |
| `morning_star` | `long` | `0.60` | Bullish three-candle pattern. |
| `evening_star` | `short` | `0.60` | Bearish three-candle pattern. |
| `unknown` | `flat` | `0.00` | Pattern is unclear; stay neutral. |

The DQN still chooses the final action from Q-values. The pattern rule is used for explanation, reliability checks, and conflict detection. If a high-confidence CNN pattern disagrees with the DQN action, `latest_dqn_strategy` includes:

```python
strategy["conflict_detected"]
strategy["conflict_note"]
```

## Retrain the CNN Model

Retraining is optional. The repository already includes `cnn_model_10bar.h5`.

To retrain the CNN, make sure the preprocessed dataset is present:

```text
label8_eurusd_10bar_1500_500_val200_gaf_culr.pkl
```

Then run:

```bash
python train_cnn_model.py
```

The training script uses:

```python
PARAMS["classes"] = 9
PARAMS["lr"] = 0.01
PARAMS["epochs"] = 50
PARAMS["batch_size"] = 64
```

After training, the script saves the model to:

```text
cnn_model_10bar.h5
```

Because the main demo loads `cnn_model_10bar.h5` from the project root, the retrained model can be used immediately:

```bash
python Trading_AgenticAI_GAF_CNN_DQN_Demo.py
```

## Tune the DQN and Execution Rules

The DQN training call is:

```python
train_dqn(states, rewards_returns, episodes=15, gamma=0.98, epsilon=0.30, max_steps=300)
```

Execution-parameter search is controlled by:

```python
MIN_HOLD_GRID = [12, 24, 48, 72, 120, 168]
SWITCH_MARGIN_GRID = [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.010, 0.015]
ENTER_MARGIN_GRID = [0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.010]
```

General teaching intuition:

- Higher `min_hold_bars` means fewer trades and longer holding periods.
- Higher `switch_margin` means the model needs stronger evidence before switching positions.
- Higher `enter_margin` means the model needs stronger evidence before leaving `flat`.
- Lower thresholds usually create more trades, but may increase transaction costs and drawdown.

## Optional Gemini Explanation Layer

`GeminiExplanationAgent` is optional. If `GEMINI_API_KEY` is not set, the script uses a deterministic fallback report and continues running.

To enable Gemini:

```bash
export GEMINI_API_KEY="your_api_key"
export GEMINI_MODEL="gemini-2.5-flash"
python Trading_AgenticAI_GAF_CNN_DQN_Demo.py
```

On Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="your_api_key"
$env:GEMINI_MODEL="gemini-2.5-flash"
python Trading_AgenticAI_GAF_CNN_DQN_Demo.py
```

## Notes for GitHub Upload

This folder is not necessarily initialized as a Git repository yet. To create one:

```bash
git init
git add .
git commit -m "Add GAF-CNN DQN trading demo"
```

The dataset file is about 75 MB, which is under GitHub's 100 MB single-file limit, but it is still large. If future datasets or models exceed 100 MB, use Git LFS:

```bash
git lfs install
git lfs track "*.h5" "*.pkl"
git add .gitattributes
```

The included `.gitignore` excludes common local files:

```text
.venv/
__pycache__/
.ipynb_checkpoints/
.vscode/
jupyter_stdout.log
jupyter_stderr.log
results/
```

## Suggested Student Exercises

1. Change `TICKER` and compare EUR/USD, GBP/USD, and AUD/USD.
2. Change `WINDOW` and inspect how the GAF images change.
3. Adjust CNN confidence thresholds and observe how many signals are gated to `unknown`.
4. Tune DQN training parameters and compare annualized return, drawdown, and trade count.
5. Modify `StrategyAgent.pattern_to_action` and test how rule-based pattern advice interacts with DQN actions.
6. Add code to save `agentic_context` as JSON for later strategy analysis.

## References

This teaching demo builds on the reference material included in `FinancialVision-master/`:

- Encoding Candlesticks as Images for Patterns Classification Using Convolutional Neural Networks
- Deep Reinforcement Learning for Foreign Exchange Trading

When reusing this project in reports or assignments, please cite the original papers and clearly state that this repository is an educational adaptation.
