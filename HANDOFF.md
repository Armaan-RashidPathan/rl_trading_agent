# RL Trading Agent — Project Handoff Report

Use this document to onboard another assistant (e.g. Claude) or to resume work with full context.

**Repository path:** `c:\Users\dihsa\rl_trading_agent`  
**Related doc:** `PROFESSOR_GUIDE.md` (architecture, setup, and run instructions in depth).

---

## Agent transcripts (search note)

A search of Cursor agent transcripts for this kind of “status + directories” report did not find older conversations on the same topic—only the session that requested this file.

---

## Project summary

**Name:** DRL Trading Agent with Sentiment Analysis  

**Goal:** Train and compare **two PPO agents** (sentiment + technical vs. baseline without sentiment) on a single equity (default **AAPL**), with evaluation plots and attention-based explainability.

### What is implemented end-to-end

| Area | Status |
|------|--------|
| Data: prices, news alignment, technical + regime features | Done (`src/data/`) |
| Sentiment: VADER / FinBERT, daily sentiment features | Done |
| Combined feature matrix | `data/processed/combined_features.csv` |
| Gymnasium **trading environment** (macro-actions, commitment, costs) | Done (`src/models/trading_env.py`) |
| **Cross-modal attention** fusion + weight storage | Done (`src/models/attention.py`) |
| **PPO** (Stable-Baselines3): sentiment agent + baseline | Done (`src/models/ppo_agent.py`) |
| Training pipeline + experiment runner | Done (`src/training/`) |
| Evaluation: portfolio / drawdown / metrics plots | Done (`src/evaluation/evaluator.py`) → `results/plots/` |
| Explainability: attention visualization | Done (`src/explainability/attention_viz.py`) |
| Saved checkpoints | `models/saved/sentiment_agent.zip`, `baseline_agent.zip` |
| Metrics snapshot | `results/training_results.json` |

### Git milestones (recent)

`Initial commit` → Week 2 (features + FinBERT) → Week 3 → trading env → cross-modal attention → stock buy/sell/hold testing and results saved.

### Example metrics (from `results/training_results.json` in repo)

Train 1143 days, test 286 days; sentiment vs baseline vs buy-and-hold recorded (e.g. sentiment Sharpe ~1.57 vs baseline ~1.23 in that snapshot).

### Empty / placeholder

- `notebooks/` — no tracked files yet  
- `tests/` — no tracked files yet  

---

## Directory tree (meaningful paths)

`venv/` is omitted below (local Python environment; recreate with `requirements.txt`, do not copy thousands of vendor files).

```
rl_trading_agent/
├── .gitignore
├── requirements.txt
├── PROFESSOR_GUIDE.md          # full academic/run guide
├── HANDOFF.md                  # this file
├── kaggle.py                   # Kaggle-related helper (repo root)
├── configs/
│   └── config.yaml             # ticker, dates, hyperparams, paths
├── data/
│   ├── raw/                    # prices, news, optional Kaggle CSVs
│   └── processed/              # technical_features, sentiment_scores, combined_features
├── models/
│   └── saved/                  # sentiment_agent.zip, baseline_agent.zip
├── notebooks/                  # (empty)
├── results/
│   ├── training_results.json
│   ├── plots/                  # portfolio, drawdown, metrics, returns, etc.
│   ├── figures/
│   ├── metrics/
│   └── models/
├── src/
│   ├── data/                   # loader, features, sentiment, combiner, pipeline
│   ├── models/                 # env, attention, PPO agents, integration_test
│   ├── training/               # trainer, run_experiment
│   ├── evaluation/             # evaluator + plots
│   └── explainability/         # attention_viz
├── tests/                      # (empty)
└── venv/                       # local virtualenv (reinstall deps; do not commit)
```

---

## Quick commands

- **Regenerate evaluation plots** (uses saved models + data):

  ```powershell
  cd C:\Users\dihsa\rl_trading_agent
  python -m src.evaluation.evaluator
  ```

- **Explainability (attention):** see `PROFESSOR_GUIDE.md` for the exact `attention_viz` command and outputs.

- **Full training / experiments:** `configs/config.yaml` plus `src/training/trainer.py` and `src/training/run_experiment.py`.

---

## Git status note

At the time this file was added, the working tree may still have local changes (e.g. plots, `evaluator.py`, untracked `PROFESSOR_GUIDE.md` or `src/explainability/`). Commit when ready.
