# Agentic AI FX Trading Demo: GAF-CNN + DQN

本專案是一個教學用的外匯交易策略範例，將 OHLC K 線資料轉成 GAF 影像，用 CNN 辨識 8 類 K 線型態加上 `unknown`，再把 CNN 機率、近期報酬與信心分數輸入 DQN/QRL，產生 `short`、`flat`、`long` 交易動作。最後以 Agentic AI 角色封裝成訊號分析、策略建議、風險檢查與文字解釋。

> 注意：本專案僅供課堂展示與研究，不是投資建議，也沒有連接真實下單系統。

## 專案內容

| 檔案 / 資料夾 | 說明 |
|---|---|
| `Trading_AgenticAI_GAF_CNN_DQN_Demo.py` | 主要 demo。下載 EUR/USD OHLC 資料，產生 GAF，載入 CNN，訓練 DQN，回測並輸出策略建議。 |
| `cnn_model_10bar.h5` | 已訓練好的 CNN 模型。主程式預設直接讀取這個檔案。 |
| `label8_eurusd_10bar_1500_500_val200_gaf_culr.pkl` | CNN 重新訓練資料，包含 train/val/test 的 GAF 影像與標籤。 |
| `train_cnn_model.py` | 重新訓練 CNN 的腳本。訓練完成後會覆蓋輸出 `cnn_model_10bar.h5`。 |
| `check_strategy.py` | 讀取 `results/agentic_context.json` 的輔助檢查腳本。若使用目前 Colab-ready 版本，主程式預設不會寫出 `results/`，需自行加入輸出儲存。 |
| `requirements-colab.txt` | Colab / Python 環境依賴。 |
| `PATTERN_STRATEGY_GUIDE.md` | CNN 型態到交易建議的對照說明。 |
| `OPTIMIZATION_GUIDE.md` | DQN 訓練與策略參數優化建議。 |
| `FinancialVision-master/` | 參考來源程式，包含 GAF-CNN 與外匯 DRL 範例。 |

## 環境安裝

建議使用 Python 3.10 或 Colab。Windows PowerShell 範例：

```powershell
cd C:\UW_AgenticAI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-colab.txt
```

如果本機執行時缺少 Plotly，請另外安裝：

```powershell
python -m pip install plotly
```

## 直接使用已訓練模型

如果不想重新訓練 CNN，只要保留根目錄的 `cnn_model_10bar.h5`，直接執行主程式即可：

```powershell
python Trading_AgenticAI_GAF_CNN_DQN_Demo.py
```

主流程會做以下事情：

1. 從 Yahoo Finance 下載 `EURUSD=X` 的 1 小時 OHLC 資料。
2. 若下載失敗，改用合成 OHLC 資料讓 demo 繼續執行。
3. 將每個 10 根 K 線視窗轉成 `10 x 10 x 4` 的 GAF-CNN 輸入。
4. 載入 `cnn_model_10bar.h5`，輸出 9 類 K 線型態機率。
5. 用 `12` 個近期報酬、`9` 個 CNN 型態機率、`1` 個 CNN 信心分數組成 DQN state。
6. 訓練 DQN，搜尋 `min_hold_bars`、`switch_margin`、`enter_margin` 等執行參數。
7. 回測最佳參數，產生 `short / flat / long` 動作、績效摘要與 Agentic AI 風險審查。

預設交易標的與資料週期在主程式上方可調整：

```python
TICKER = "EURUSD=X"
PERIOD = "730d"
INTERVAL = "1h"
WINDOW = 10
HORIZON = 8
COST = 0.00002
```

## 在專案中使用策略建議

主程式最後會建立：

```python
agentic_context, agentic_llm_report = interface_agent.advise(
    raw,
    states[-1],
    summary,
    decision_explanations,
)
```

其中 `agentic_context` 是策略系統最適合被其他程式接上的結構化輸出，重要欄位如下：

```python
strategy = agentic_context["latest_dqn_strategy"]
risk = agentic_context["latest_risk_review"]
final = agentic_context["final_recommendation"]

print(strategy["action"])        # short / flat / long
print(strategy["q_values"])      # DQN 對三個動作的 Q-value
print(strategy.get("pattern_info"))
print(risk["decision"])          # approve / reduce / block
print(risk["suggested_position_size"])
print(final)
```

實際交易策略可以用這個邏輯接入自己的系統：

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

目前系統只輸出建議，不會自動下單。要接交易 API 時，請另外加入下單、倉位限制、停損停利、滑價、交易時段與錯誤重試機制。

## CNN 型態與策略對照

`StrategyAgent.pattern_to_action` 內建下列規則：

| CNN 型態 | 型態建議 | 預設信心門檻 |
|---|---:|---:|
| `doji` | `flat` | `0.70` |
| `hammer` | `long` | `0.60` |
| `hanging_man` | `short` | `0.60` |
| `shooting_star` | `short` | `0.70` |
| `bullish_engulfing` | `long` | `0.50` |
| `bearish_engulfing` | `short` | `0.50` |
| `morning_star` | `long` | `0.60` |
| `evening_star` | `short` | `0.60` |
| `unknown` | `flat` | `0.00` |

DQN 最終仍會依 Q-value 選擇動作；CNN 型態規則主要用於補充說明、信心判斷與衝突偵測。如果 CNN 高信心建議與 DQN 動作不同，`latest_dqn_strategy` 會包含 `conflict_detected` 與 `conflict_note`。

## 重新訓練 CNN 模型

若要重新訓練 K 線型態 CNN，確認根目錄存在：

```text
label8_eurusd_10bar_1500_500_val200_gaf_culr.pkl
```

然後執行：

```powershell
python train_cnn_model.py
```

訓練腳本設定：

```python
PARAMS["classes"] = 9
PARAMS["lr"] = 0.01
PARAMS["epochs"] = 50
PARAMS["batch_size"] = 64
```

輸出模型會存成根目錄的：

```text
cnn_model_10bar.h5
```

因此重新訓練後，不需要改主程式，直接再次執行：

```powershell
python Trading_AgenticAI_GAF_CNN_DQN_Demo.py
```

## 調整 DQN 與交易參數

DQN 訓練在 `train_dqn()`：

```python
train_dqn(states, rewards_returns, episodes=15, gamma=0.98, epsilon=0.30, max_steps=300)
```

參數搜尋在 `MIN_HOLD_GRID`、`SWITCH_MARGIN_GRID`、`ENTER_MARGIN_GRID`：

```python
MIN_HOLD_GRID = [12, 24, 48, 72, 120, 168]
SWITCH_MARGIN_GRID = [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.010, 0.015]
ENTER_MARGIN_GRID = [0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.010]
```

想要更保守，可提高 `min_hold_bars`、`switch_margin`、`enter_margin`，讓策略少換倉。想要更多交易機會，可降低這些門檻，但通常會增加交易成本與回撤風險。

## Gemini 解釋層

`GeminiExplanationAgent` 是選用功能。若沒有設定 API key，程式會產生 deterministic fallback report，不會中斷。

若要啟用 Gemini：

```powershell
$env:GEMINI_API_KEY="your_api_key"
$env:GEMINI_MODEL="gemini-2.5-flash"
python Trading_AgenticAI_GAF_CNN_DQN_Demo.py
```

## GitHub 上傳建議

初始化 Git：

```powershell
git init
git add README.md requirements-colab.txt *.py *.md cnn_model_10bar.h5 label8_eurusd_10bar_1500_500_val200_gaf_culr.pkl FinancialVision-master
git commit -m "Add GAF-CNN DQN trading demo"
```

資料檔 `label8_eurusd_10bar_1500_500_val200_gaf_culr.pkl` 約 75 MB，低於 GitHub 單檔 100 MB 限制，但仍偏大。若之後模型或資料超過 100 MB，請改用 Git LFS：

```powershell
git lfs install
git lfs track "*.h5" "*.pkl"
git add .gitattributes
```

建議不要上傳：

```text
.venv/
__pycache__/
.ipynb_checkpoints/
jupyter_stdout.log
jupyter_stderr.log
```

## 參考來源

本專案整合並改寫自 `FinancialVision-master` 中的兩個方向：

- Encoding Candlesticks as Images for Patterns Classification Using Convolutional Neural Networks
- Deep Reinforcement Learning for Foreign Exchange Trading

請保留原始引用與授權資訊，並在 README 或報告中說明本專案是教學改作版本。
