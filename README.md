# btc-quant-prob: Probabilistic Bitcoin Price Forecasting

A robust Python repository for probabilistic Bitcoin (BTC-USD) price forecasting, focused on uncertainty quantification, rigorous backtesting, and automated optimization. The system is engineered for data science research, educational exploration, and developing advanced trading strategies — not for financial advice or live trading.

**Disclaimer**: This project is for research and educational purposes only. It is not investment advice. Financial markets are uncertain; past performance does not guarantee future results.

---

## 🏛️ Architecture Overview

The system is built modularly, with clear divisions between key processes:

1. **Data Pipeline (`/data`)**  
   Ingests, validates, and preprocesses raw OHLCV (Open/High/Low/Close/Volume) data.

2. **Feature Engineering (`/features`)**  
   Constructs features including technical indicators, statistical metrics, volatility ratios, "halving" event counters, and custom “alpha” features with a scikit-learn pipeline.

3. **Modeling (`/models`)**  
   Implements several probabilistic models:
   - LightGBM Quantile Regression for fast, direct quantile forecasts.
   - Bayesian LSTM (with MC Dropout) for deep temporal uncertainty modeling (PyTorch).
   - Gaussian Process for Bayesian baselines.
   - Ensemble models combining various approaches by performance-weighted stacking.

4. **Training & Hyperparameter Tuning (`/train`, `/autotune`)**  
   Unified scripts for model training, leveraging Optuna for automatic hyperparameter search.

5. **Backtesting (`/backtest`)**  
   Rigorous walk-forward validation with trading simulation, including risk management logic.

6. **Evaluation (`/eval`)**  
   Tools for forecast quality measurement, such as CRPS, PICP, Winkler Score, calibration plots, and post-hoc recalibration.

7. **Deployment & Monitoring (`/deploy`)**  
   FastAPI serving for predictions and a performance drift monitor.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Git

### Installation

1. Clone the repository:

    ```bash
    git clone <repository_url>
    cd btc-quant-prob
    ```

2. (Recommended) Create and activate a virtual environment, then install dependencies:

    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. Download Data:  
   Get [Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data/data) from Kaggle and place `bitcoin_historical_data.csv` inside `btc-quant-prob/raw_data/`.

   _Tip: For quick tests, a mock data generator is also included in the codebase._

---

## ⚙️ Running the System

Basic workflow:

1. **Activate your virtual environment:**  
   ```bash
   source venv/bin/activate
   ```
   You should see `(venv)` in your terminal prompt.

2. **Train a Model:**  
   ```bash
   python -m train.train --horizon 180 --model_name lgbm_quantile
   ```
   Trains a LightGBM quantile regression model for 180-day predictions, saving the trained model in `artifacts/`.

3. **Generate Walk-Forward Backtest Predictions:**  
   ```bash
   python -m backtest.walkforward --horizon 180 --model_name lgbm_quantile
   ```
   Stores predictions in `artifacts/backtest/`.

4. **Evaluate and Visualize:**  
   Open and execute the notebook `notebooks/02_traineval.ipynb` for analysis and visualization.

**Summary Workflow:**
1. Training (`train.train`) → produces model
2. Backtesting (`backtest.walkforward`) → produces historical results
3. Notebook evaluation (`notebooks/02_traineval.ipynb`) → visualization and analysis

---

## 🧑‍🔬 Advanced Usage & Suggested Improvements

### Phase 1: System Optimization
- Tune hyperparameters for each model (see `autotune/iterate.py`, using Optuna).
- Engineer new, original features ("alpha"). Ideas:  
    - Short/long-term volatility ratios
    - Feature interactions (e.g., RSI × volatility)
    - Halving effects (`add_halving_features`)
- Try more models or ensembles (`models/gp_baseline.py`, `models/ensemble.py`).
- Refine backtester: Add stop-loss, volatility-based position sizing, and "high-conviction" trading signals in `backtest/simulate.py`.

### Phase 2: Expand Knowledge Base
- Deepen knowledge in probability/statistics (CRPS, Bayesian methods), time-series analysis (ARIMA, GARCH), linear algebra/calculus.
- Understand financial market concepts: EMH, market structure, options/futures.

### Phase 3: Build Portfolio & Community
- Document experiments methodically (“50 features tested, feature X improved Sharpe by 0.2…”).
- Join Kaggle time series competitions.
- Read quant research papers and replicate simple ideas.

---

## 📝 Example Past Experiment Log

- Applied Optuna hyperparameter tuning (50+ trials). Achieved low CRPS, but trading results poor with default median-based strategy (Sharpe < 0, max drawdown -100%).
- Added "halving" and new interaction features: model uncertainty prediction improved (CRPS = 0.11).
- Tried high-conviction, tail-based, and contrarian trading strategies. Only contrarian logic yielded sustainable returns (Sharpe ≈ 2), but drawdown remains problematic during regime shifts.
- Discovered that feature-rich models can forecast uncertainty, not direction — highlighting the challenge of trading with predictive signals.
- Plan: add volatility-based position sizing and trend regime filters (e.g., SMA-200) for risk management.

---

## 📚 References

- [mczielinski/bitcoin-historical-data (Kaggle)](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
- [CRPS: Continuous Ranked Probability Score](https://en.wikipedia.org/wiki/Scoring_rule#Continuous_ranked_probability_score)
- Quant blog reading: AQR, Renaissance, Two Sigma, etc.

---

## 📢 Contributing

Research ideas, feature pull requests, and reproducible experiment logs are welcome. Open issues, document your findings, and help make this project a clearer resource for quant-oriented probabilistic research!

---

## License

MIT (c) 2024-present  
Original author: [jwjooth](https://github.com/jwjooth)