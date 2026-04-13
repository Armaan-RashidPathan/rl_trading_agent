# DRL Trading Agent with Sentiment Analysis — Academic Project Guide

This document is written to help a professor (or any new reader) understand, run, and evaluate the project in this repository. The content below is based on the code and artifacts present in this workspace.

---

## 1. Project Overview

### Project title
- **DRL Trading Agent with Sentiment Analysis**

### Problem statement
- **Goal**: Build and evaluate a deep reinforcement learning (DRL) trading agent that can trade a single equity (default: **AAPL**) using **market technical indicators** and **news-derived sentiment**, and compare it against simpler baselines.

### Why this problem matters
- **Financial markets are noisy and non-stationary**: simple heuristics can break when regimes change (e.g., high-volatility periods).
- **Text signals may add value**: market-moving information often arrives via news; sentiment features provide an additional information channel beyond price history.
- **Academic relevance**: this project combines **time-series feature engineering**, **NLP sentiment modeling**, **RL environment design**, and **empirical evaluation** with metrics and plots.

### Main objective of the project
- **Train two PPO agents** and compare them on a held-out test period:
  - **Sentiment agent**: uses technical + regime + sentiment features and a custom **cross-modal attention** fusion layer.
  - **Baseline agent**: uses only price/technical/regime features (no sentiment).
- Produce a **results package** (metrics + plots) and an **explainability report** via attention-weight extraction.

### Key features
- **Custom trading environment** with:
  - Macro-actions and **action commitment** (hierarchical-like behavior).
  - Transaction costs and portfolio tracking.
- **Feature pipeline**:
  - Yahoo Finance price data (`yfinance`).
  - News data from Kaggle (if present) + generated sample headlines for gaps.
  - Technical indicators + regime features.
  - Daily sentiment features from headlines (VADER or FinBERT).
  - Final combined feature matrix used by RL.
- **DRL training** using Stable-Baselines3 PPO with a custom feature extractor.
- **Evaluation** with portfolio/returns/drawdown plots and metrics tables.
- **Explainability**: extraction and visualization of attention weights over time and by action type.

---

## 2. System Architecture

### High-level explanation of how the system works
1. **Data acquisition**: download prices; load/assemble news.
2. **Feature engineering**: compute technical indicators + market regime features.
3. **Sentiment extraction**: compute daily sentiment features from aligned news headlines.
4. **Feature combination**: merge technical/regime + sentiment into one feature matrix.
5. **RL training**:
   - The environment emits a **30-dimensional state**:
     - **24 market features** (19 technical+regime + 5 sentiment)
     - **6 portfolio/commitment features** (position/balance + commitment state)
   - PPO learns a policy over **24 discrete actions** (6 macro actions × 4 commitment lengths).
6. **Evaluation & reporting**: compare sentiment agent vs baseline vs buy-and-hold and generate plots.
7. **Explainability**: run the trained policy and read attention weights from the model to interpret when/why sentiment mattered.

### Main components/modules
- **Data layer**: `src/data/`
  - `data_loader.py`: price/news loading and news-to-trading-day alignment.
  - `feature_engineering.py`: technical + regime features.
  - `sentiment_extractor.py`: VADER/FinBERT sentiment + daily sentiment features.
  - `feature_combiner.py`: merges all features into `data/processed/combined_features.csv`.
  - `data_pipeline.py`: thin loader used by training/evaluation scripts (expects precomputed combined features).
- **RL/Model layer**: `src/models/`
  - `trading_env.py`: Gymnasium environment.
  - `attention.py`: cross-modal attention fusion + attention weight storage for explainability.
  - `ppo_agent.py`: SB3 PPO wrappers (sentiment agent + baseline agent) and custom feature extractor.
  - `integration_test.py`: end-to-end wiring test.
- **Training**: `src/training/`
  - `trainer.py`: `TrainingPipeline` orchestrating data load → train → evaluate → save.
  - `run_experiment.py`: CLI wrapper for quick vs full experiments.
- **Evaluation**: `src/evaluation/evaluator.py`
  - Loads saved models, re-evaluates, and generates plots in `results/plots/`.
- **Explainability**: `src/explainability/attention_viz.py`
  - Extracts attention weights from the trained sentiment agent and generates explainability plots.

### Technologies, frameworks, and languages used
- **Language**: Python
- **RL**: `stable-baselines3` (PPO), `gymnasium`
- **Deep learning**: `torch`
- **NLP**: `transformers` (FinBERT), `nltk` (VADER)
- **Data**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Config**: `pyyaml`

### How components interact (dataflow)
- `src/data/*` produce:
  - `data/raw/prices.csv`
  - `data/raw/news.csv`
  - `data/processed/technical_features.csv`
  - `data/processed/sentiment_scores.csv`
  - `data/processed/combined_features.csv`
- `src/training/trainer.py` consumes `combined_features.csv` + `prices.csv`, trains agents, and saves:
  - models to `models/saved/`
  - metrics to `results/training_results.json`
- `src/evaluation/evaluator.py` consumes saved models + data and writes plots to `results/plots/`.
- `src/explainability/attention_viz.py` consumes saved sentiment model + data and writes explainability plots to `results/plots/`.

---

## 3. Folder Structure Explanation

### Repository layout (important paths)
- **`configs/`**
  - `config.yaml`: central configuration (ticker/date range, hyperparameters, paths).
- **`data/`**
  - `raw/`: raw inputs (`prices.csv`, `news.csv`, Kaggle files).
  - `processed/`: engineered outputs used for training (`technical_features.csv`, `sentiment_scores.csv`, `combined_features.csv`).
- **`models/`**
  - `saved/`: SB3 `.zip` model checkpoints (e.g., `sentiment_agent.zip`, `baseline_agent.zip`).
- **`results/`**
  - `training_results.json`: summary metrics from the training pipeline.
  - `plots/`: generated PNG figures (portfolio comparisons, drawdowns, attention plots, etc.).
- **`src/`**
  - `data/`: dataset creation pipeline.
  - `models/`: environment, attention module, agents.
  - `training/`: training orchestration and experiment runner.
  - `evaluation/`: evaluation + plotting.
  - `explainability/`: attention extraction + interpretability plots.
- **`requirements.txt`**: Python dependency list.

---

## 4. Implementation Details

### Core algorithms / logic used
- **RL algorithm**: Proximal Policy Optimization (PPO) from Stable-Baselines3.
- **State design** (in `TradingEnvironment`):
  - 24 engineered market features (technical/regime + sentiment).
  - 6 portfolio/commitment features to support action persistence and interpretability.
- **Action design**:
  - 6 macro-actions (e.g., `EXIT`, `HOLD`, `TREND_FOLLOW`, `MEAN_REVERT`) × 4 commitment durations `[1, 3, 5, 10]` → **24 discrete actions**.
  - If commitment is active, the environment overrides the new action and continues the committed macro-action.
- **Reward**:
  - Portfolio return signal scaled (percent-style) + drawdown-aware risk penalty.
- **Cross-modal attention** (in `src/models/attention.py`):
  - Splits the 24 market features into:
    - **19 price/technical/regime** features
    - **5 sentiment** features
  - Learns two scalar attention weights per step:
    - Price → Sentiment
    - Sentiment → Price
  - Stores these weights for later explainability plots.

### Important functions/classes (what they do)
- **`src/training/trainer.py::TrainingPipeline`**
  - Loads data, splits train/test, trains sentiment & baseline agents, evaluates both, compares to buy-and-hold, saves models and `results/training_results.json`.
- **`src/models/trading_env.py::TradingEnvironment`**
  - Implements a Gymnasium-compatible trading simulator with commitment mechanics and portfolio accounting.
- **`src/models/ppo_agent.py::HierarchicalPPOAgent`**
  - Wraps SB3 PPO and builds a policy using `AttentionFeatureExtractor` (sentiment version) and supports `train/evaluate/save/load`.
- **`src/models/ppo_agent.py::BaselinePPOAgent`**
  - Same interface but strips sentiment features (price-only) and uses a smaller policy (no attention fusion).
- **`src/evaluation/evaluator.py::TradingEvaluator`**
  - Generates the report plots used in the presentation.
- **`src/explainability/attention_viz.py::AttentionExplainer`**
  - Extracts attention weights from the trained policy network and generates interpretability plots + a text report.

### Design decisions (why they were made)
- **Macro-actions + commitment**: approximates hierarchical behavior without implementing a full hierarchical RL algorithm; reduces churn and makes behavior easier to interpret.
- **Regime features**: helps the policy condition behavior on market regimes (bull/bear/high-volatility).
- **Two-agent comparison**: isolates the incremental value of sentiment features by comparing to a price-only baseline.
- **Attention-based fusion**: provides both representational power and an interpretable signal for “when sentiment mattered.”
- **Look-ahead avoidance**:
  - News alignment uses *previous day news* to reduce leakage risk.
  - Rolling normalization utilities are implemented (see `FeatureNormalizer`) to avoid using future statistics (note: the combined feature CSV in this repo is already produced).

---

## 5. Setup Instructions (Run From Scratch)

### Prerequisites
- **Python**: 3.10+ recommended (this repo includes a local `venv/` suggesting Python 3.11 was used).
- **System**: Windows/macOS/Linux (commands below include Windows PowerShell examples).
- **Disk/Network**:
  - FinBERT first run may download ~500MB from Hugging Face.
- **Optional**: a Kaggle news CSV file at `data/raw/raw_analyst_ratings.csv` (project can still run using generated sample news).

### Installation steps (recommended clean setup)
1. Create and activate a virtual environment:

```powershell
cd C:\Users\dihsa\rl_trading_agent
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

### Environment setup
- The project uses `configs/config.yaml` for paths and parameters. Default ticker is **AAPL**.
- If you change ticker/date range, re-run the data pipeline to regenerate raw/processed files.

---

## 6. How to Run the Project

### Option A — Use existing processed data and saved models (fastest)
This repo already contains:
- `data/processed/combined_features.csv`
- saved models in `models/saved/`
- existing plots in `results/plots/`

You can regenerate plots (evaluation report) from the saved models:

```powershell
cd C:\Users\dihsa\rl_trading_agent
python -m src.evaluation.evaluator
```

You can regenerate explainability plots (attention analysis):

```powershell
python -m src.explainability.attention_viz
```

### Option B — Rebuild data pipeline (reproducibility run)
1. Download prices + build/align news:

```powershell
python -m src.data.data_loader
```

2. Create technical + regime features:

```powershell
python -m src.data.feature_engineering
```

3. Create sentiment features:
- Fast (VADER):

```powershell
python -m src.data.sentiment_extractor
```

- Accurate but slow (FinBERT):

```powershell
python -m src.data.sentiment_extractor --finbert
```

4. Combine features:

```powershell
python -m src.data.feature_combiner
```

### Option C — Train the agents (re-run experiments)
- Quick experiment (~10 minutes, 50k timesteps):

```powershell
python -m src.training.run_experiment --mode quick
```

- Full experiment (~45 minutes, 200k timesteps):

```powershell
python -m src.training.run_experiment --mode full
```

### How to verify it’s working
- After training: confirm these outputs exist/updated:
  - **Models**: `models/saved/sentiment_agent.zip`, `models/saved/baseline_agent.zip`
  - **Metrics**: `results/training_results.json`
- After evaluation:
  - PNG plots appear in `results/plots/` (e.g., `portfolio_comparison.png`, `metrics_table.png`)
- After explainability:
  - Attention plots appear in `results/plots/` (e.g., `attention_timeline.png`, `attention_by_action.png`)

---

## 7. Demonstration Guide (For Presentation)

Below is a step-by-step demo script you can follow live.

### Demo setup (before class)
- Ensure the environment is activated and dependencies are installed.
- Ensure `models/saved/` exists and contains `sentiment_agent.zip` and `baseline_agent.zip`.
- Close unnecessary programs to keep the demo responsive.

### Demo script (what to say + what to show)
1. **Intro (30–45 seconds)**
   - Say: “This project trains a PPO-based trading agent that combines technical indicators with news sentiment. I compare a sentiment-aware agent to a price-only baseline and buy-and-hold.”
   - Show: `configs/config.yaml` (ticker, timesteps, action space description in your own words).

2. **Show the pipeline artifacts (30 seconds)**
   - Show folder tree:
     - `data/raw/` and `data/processed/`
     - `models/saved/`
     - `results/plots/`

3. **Generate evaluation plots live (1 minute)**
   - Run:

```powershell
python -m src.evaluation.evaluator
```

   - Show in `results/plots/`:
     - `portfolio_comparison.png`
     - `drawdown_comparison.png`
     - `metrics_table.png`
   - Say: “These plots compare absolute performance and risk via drawdowns and Sharpe ratio.”

4. **Run explainability (1–2 minutes)**
   - Run:

```powershell
python -m src.explainability.attention_viz
```

   - Show:
     - `attention_timeline.png` (when attention spikes, what actions occur)
     - `attention_correlation_heatmap.png` (correlations between attention and features)
   - Say: “Attention weights are extracted from the model’s fusion layer and serve as an interpretability signal for how much sentiment influenced the representation.”

5. **(Optional) Quick training run (if time permits)**
   - Run:

```powershell
python -m src.training.run_experiment --mode quick
```

   - Say: “This retrains both agents for a short run and writes a fresh `results/training_results.json`.”

### Example inputs to try
- Change ticker/date range in `configs/config.yaml` (then re-run the full data pipeline).
- Toggle sentiment extractor mode:
  - VADER for speed
  - FinBERT for accuracy

### Expected outputs
- Console logs for each stage (data loading, training progress, evaluation summary).
- Updated plots in `results/plots/`.

---

## 8. Example Output / Results

### Sample metrics (from this repository)
The file `results/training_results.json` includes an example run with:
- **Sentiment agent**:
  - Mean return: **~35.13%**
  - Sharpe ratio: **~1.57**
  - Max drawdown: **~15.20%**
- **Baseline agent / Buy & Hold**:
  - Mean return: **~32.02%**
  - Sharpe ratio: **~1.23**
  - Max drawdown: **~16.61%**

Interpretation (academic framing):
- The sentiment agent shows **higher return and higher Sharpe** than the baseline on the same split, suggesting the sentiment channel contributed additional predictive signal in that run.
- Drawdown is slightly lower for the sentiment agent, indicating improved risk profile in this sample.

### Figures produced by the project
Open the PNGs in `results/plots/` (already present in this repo), including:
- `portfolio_comparison.png`: portfolio value trajectories
- `returns_distribution.png`: distribution of daily returns
- `drawdown_comparison.png`: drawdowns over time
- `metrics_table.png`: side-by-side metrics table
- `attention_timeline.png`: attention weights + price + actions
- `attention_by_action.png`: how attention varies with action types
- `attention_correlation_heatmap.png`: correlation among attention and features

---

## 9. Progress & Development Process (Suggested Narrative)

Use this as a “research story” for an academic audience.

### Initial idea
- Combine **technical indicators** and **financial news sentiment** into an RL trading agent, then evaluate whether sentiment measurably improves outcomes.

### Major milestones
- **Week 1–2**: Data ingestion + feature engineering (prices, indicators, regimes).
- **Week 2–3**: News alignment + daily sentiment features (VADER/FinBERT), feature merging.
- **Week 3**: Trading environment with macro-actions and commitment; integration test (`src/models/integration_test.py`).
- **Week 4**: PPO training (sentiment vs baseline), evaluation plots, and explainability module.

### Challenges faced (typical for this domain)
- **Look-ahead bias**: solved by aligning news from the *previous day* to each trading day.
- **Sparse/uneven news coverage**: solved by combining real Kaggle headlines (when available) with generated sample headlines to avoid empty days.
- **Non-stationarity**: addressed by regime features and a reward that includes drawdown penalty.
- **Interpretability of DRL**: addressed by attention-weight extraction and visualization.

---

## 10. Limitations

- **Single-asset focus**: default configuration is one ticker (AAPL).
- **Backtest realism**:
  - simplified position model (cash vs holding, all-in trades)
  - no slippage model beyond transaction cost
  - no shorting, leverage, partial position sizing
- **Sentiment data quality**:
  - can depend on availability/quality of Kaggle headlines
  - sample headline generation is synthetic and may not reflect real-world news dynamics
- **Evaluation protocol**:
  - project uses a simple train/test split in training pipeline; walk-forward scaffolding exists but is not the main training loop in `TrainingPipeline`.

---

## 11. Future Improvements

- **Multi-asset portfolio** and risk constraints (position sizing, leverage limits).
- **More realistic execution model** (slippage, spreads, market impact).
- **Walk-forward training loop** (train/val/test folds as primary experiment design).
- **Richer NLP**:
  - entity-specific sentiment
  - time-decay weighting of headlines
  - source credibility weighting
- **Better policy architecture**:
  - sequence models over time (e.g., temporal attention / transformers for observations)
  - distributional RL or risk-sensitive objectives

---

## 12. 5-Minute Presentation Version

- **Problem**: Trading solely from prices is hard because markets are noisy and react to information. News sentiment can provide additional signal.
- **Approach**: Build a PPO trading agent in a custom Gymnasium environment. The state includes technical indicators, regime features, and sentiment signals. A cross-modal attention layer fuses price and sentiment and produces interpretable attention weights.
- **Baselines**: Train a price-only PPO baseline and compare both to buy-and-hold.
- **Results**: The repo includes an example where sentiment improves returns and Sharpe relative to baseline, and the evaluation module generates plots for portfolio performance and drawdowns.
- **Explainability**: Attention weights are extracted during evaluation to show when the model relies more on sentiment and how that correlates with volatility or actions.
- **Limitations/Future work**: expand to multi-asset, more realistic trading constraints, and stronger evaluation protocols (walk-forward).

---

## 13. Commands Cheat Sheet

### Install

```powershell
cd C:\Users\dihsa\rl_trading_agent
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Build data (from scratch)

```powershell
python -m src.data.data_loader
python -m src.data.feature_engineering
python -m src.data.sentiment_extractor
python -m src.data.feature_combiner
```

FinBERT sentiment (slow):

```powershell
python -m src.data.sentiment_extractor --finbert
```

### Train

```powershell
python -m src.training.run_experiment --mode quick
python -m src.training.run_experiment --mode full
```

### Evaluate (plots/metrics)

```powershell
python -m src.evaluation.evaluator
```

### Explainability (attention plots)

```powershell
python -m src.explainability.attention_viz
```

### Smoke tests / integration checks

```powershell
python -m src.models.integration_test
python -m src.models.trading_env
python -m src.models.attention
python -m src.models.ppo_agent
python -m src.data.data_pipeline
```


