# btc-quant-prob: Probabilistic Bitcoin Price Forecasting

A production-grade Python repository for probabilistic forecasting of Bitcoin (BTC-USD) price 6–12 months ahead. This system emphasizes uncertainty quantification, rigorous backtesting, and automated model improvement, designed to simulate a real-world quantitative research environment.

**Disclaimer:** This project is for research and educational purposes only. It is not investment advice and does not guarantee investment returns. Financial markets are inherently unpredictable, and past performance is not indicative of future results.

---

## 🏛️ Architecture Overview

The system is designed with a clear, modular architecture that separates concerns, from data ingestion to model deployment.

1.  **Data Pipeline (`/data`)**: Ingests, validates, and preprocesses raw OHLCV data.
2.  **Feature Engineering (`/features`)**: Creates a rich feature set from the OHLCV data, including momentum, volatility, and statistical indicators. A `scikit-learn` pipeline ensures transformations are reproducible.
3.  **Modeling (`/models`)**: Implements several probabilistic models:
    - **LightGBM Quantile Regressor**: Fast and powerful gradient boosting for direct quantile forecasting.
    - **Bayesian LSTM (MC Dropout)**: A deep learning approach using PyTorch to capture temporal patterns and model uncertainty.
    - **Gaussian Process**: A non-parametric Bayesian model serving as a strong baseline.
    - **Ensemble**: A stacking model that combines predictions from base models, weighted by their performance.
4.  **Training & Hyperparameter Tuning (`/train`)**: A unified training script manages model training, hyperparameter optimization with Optuna, and artifact saving.
5.  **Backtesting (`/backtest`)**: Implements a rigorous walk-forward validation methodology to simulate realistic model performance over time. It includes a simple trading simulator to evaluate economic outcomes.
6.  **Evaluation (`/eval`)**: Provides a suite of tools for evaluating probabilistic forecasts, including CRPS, PICP, Winkler Score, and reliability diagrams. It also includes post-hoc calibration methods like Isotonic Regression and Conformal Prediction.
7.  **Auto-Tuning (`/autotune`)**: A self-improvement loop that runs experiments, analyzes model calibration, applies corrections, and suggests improvements.
8.  **Deployment & Monitoring (`/deploy`)**: Includes a FastAPI endpoint for serving predictions and a monitoring script to track model performance drift over time.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Git

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd btc-quant-prob
    ```

2.  **Create a virtual environment and install dependencies:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Download the data:**
    Download the Bitcoin historical data from Kaggle: [Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data/data). Place the 
    `bitcoin_historical_data.csv` file inside a newly created `btc-quant-prob/raw_data/` directory.

    _Note: The project includes a mock data generation function for quick testing if you don't have the CSV._

---

## ⚙️ How to Run the System

### 1. Open the Terminal in VS Code
.\.venv\Scripts\activate
`make sure your virtual environment is active. U should see (venv) at the beginning of your terminal`

### 2. Run the Training Script
python -m train.train --horizon 180 --model_name lgbm_quantile
`this command will train the model to predict how many days ahead. it'll read the train/config.json file and save the trained model into the artifacts/ folder`

### 3. Run the Walk-Forward Backtest
`once training is done,u need to generate historical predictions to evaluate the models performance`
python -m backtest.walkforward --horizon 180 --model_name lgbm_quantile
`itll create a .parquet file in the artifacts/backtest/ directory`

### 4. Analyze the Results in the Notebook
`now that u have backtest predictions, u can visualize everything`
`open the file notebooks/02_traineval.ipynb then run the celss one by one`

### summary of workflow
1.  traim (train.train) --> creates initial model
2.  backtest(backtest.walkforward) --> generates historical performance data
3.  evaluate(notebooks/02_TrainEval.ipynb) --> analyze the result visually

### NEXT TASKS
### fase 1: optimalkan sistem yang ada
1.  lakukan hyperparameter tuning
        model lgbm_quantile menggunakan parameter default ini hampir tidak pernah optimal
        gunakan skrip autotune/itreate.py --> menggunakan library Optuna utk secara otomatis mencari kombinasi parameter terbaik (learning_rate, num_leaves dll) yang menghasilkan performa terbaik (crps terendah atau sharpe ration tertinggi)
2.  lakukan feature engineering kreatif
        model hanya melihat harga dan volume dan itu tidak cukup, butuh sebuah fitur (alpha) yang unik
        mulailah berburu ide di notebook 01_EDA.ipynb dan coba fitur baru di features.py, seperti:
        2.1 rasio volatilitas --> rasio volatilitas jangka pendek (7 hari) terhadap jangka panjang (90 hari)
        2.2 interaksi fitur --> apakah rsi bekerja lebih baik saat volatilitas sedang tinggi? buat fitur rsi * realized_vol_30d
        2.3 fitur halving --> buat fitur "jumlah hari sejak halving terakhir"
3.  eksperimen dengan model lain
        lightgbm --> cepat dan bagus tapi bukan yang terbaik untuk data ini
        latih model lain yang sudah ada di folder models/ misalnya:
        3.1 gp_baseline.py (gaussian process) --> gp bagus utk menangkap ketidakpastian (uncertainty) tapi lebih lambat
        3.2 ensemble.py --> gabungkan prediksi dari lightgbm dan gp. seringkali, ensemble memberikan hasil yang lebih stabil
4.  perbaiki logika trading
        buka backtest/simulate.py --> perbaiki logikanya, misalnya:
        4.1 tambahkan stoploss --> jika posisi sudah rugi berapa persen, tutup paksa
        4.2 volatility sizing --> ambil posisi lebih kecil saat pasar sedang sangat fluktuatif
        4.3 hanya buka posisi jika "peluang" tinggi --> jgn masuk hanya karena prediksi positif, masuk jika prediksi sgt positif

### fase 2: perluas wawasan
1.  matematika & statistik --> ini tidak bisa ditawar. fokus pada:
    1.1 probabilitas dan statistik --> pahami semua metrik di eval/prob_metrics.py secara mendalam. pelajari ttg distribusi, hipotesis testing dan metode bayesian
    1.2 time series analysis --> pelajari model spt arima, garch --> membantu memahami dinamikas volatilitas
    1.3 aljabar linear dan kalkulus --> dasar dari semua machine learning
2. ilmu keuangan dan struktur pasar
    2.1 the efficient market hypothesis --> pahami mengapa memprediksi pasar itu sulit
    2.2 market microstructure --> pelajari ttg order book, slippage, bid-ask spread
    2.3 derivatif --. pelajari tentang options dan futures --> strategi quant yang canggih sebenarnya

### fase 3: bangun portofolio dan jaringan
1.  buat proyek ini menjadi sebuah portofolio
        dokumentasikan semua ekspreimen di README.md spt "saya mencoba 50 fitur baru, dan fitur x meningkatkan sharpe ratio sebesar 0.2 karena alasan y. ini adalah hasilnya" ini akan m enjadi isi portofolio yang sangat mengesankan
2.  ikut kompetisi --> coba ikuti kompetisi time series atau finansial di kaggle
3.  baca riset --> baca paper dari aqr, renaissance technologies, atau two sigma. coba replikasi ide-ide sederhana dari sana


### dokumentasi fase 1
### fase 1
`1.  aku sudah menggunakan library Optuna dan menggunakan parameter terbaik dengan uji percobaan sebanyak 50x tapi ditemukan hasil yang tidak memuaskan dimana value (minimilized crps): -0.799314 dengan sharpe ratio: 0.00 dan max drawdown: 0.00% --> secara teknis tidak mungkin terjadi`
`2. aku mengubah eval/prob_metrics.py nya menggunakan crps np.maximum utk memastikan hasilnya akan selalu positif. dan mengubah logika backtest/simulate.py agar hanya beli jika 95% yakin hasilnya positif dan short jika 95% yakin hasilnya negatif`
`3. logika tadingnya berubah menjadi beli jika median prediksi > 0 dan jual jika median predisi < 0`
`4. kemudian aku mengubah features/build_features.py dengan menambahkan fungsi baru yaitu add_halving_features yang menghitung "hari sejak halving" dan "hari menuju halving", dan fungsi add_feature_interactions yang membuat rasio volatilitas dan interaksi antar fitur.`
`5. lalu aku membuat fitur-fitur baru (alpha) dan menjalankan pipeline hyperparameter tuning (optuna + walk-forward) dengan hasil tuning (crps = 0.1122), trading (shapre = -0.67, mdd = -100.00%)`
**sharpe ration: -0.67 --> kehilangan uang**
**max drawdown: -100.00% --> bangkrut (margin call)**
**--- crpbs rendah tapi sharpe negatif ---**
`6. untuk mengevaluasi hasil tsb, aku memperbarui parameter terbaik dari log ke dalam train/config.json bagian models : lgbm_quantile`
`7. lalu aku melatih model lagi satu kali menggunakan parameter baru ini dan membuat file model.joblib dengan python -m train.train --horizon 180 --model_name lgbm_quantile`
`8. lalu aku menghapus file backtest lama di artifacts/backtest/lgbm_quantile_h180_predcitoins.parquet agar tidak jadi binggung, dan menjalankan backtest final sekarang menggunakan model yang sudah di-tuning dari config.json`
`9. setelah selesai backtest masuk ne notebooks 2 untuk melihat visualisasi hasil backtestnya`
**hasil yang fantastis karena model trial 11 pintar dalam memprediksi ketidakpastian (crps 0.11) tapi strategi tradingny (median > 0) sangat buruk DIJAMIN BANGKRUT LU**

## FASE EVALUASI KETIGA KALI
`1. perbaiki strategi trading`
ubah menjadi "high-conviction berdasarkan prediksi di ekor(tail) dengan mengubah backtest/simulate.py dan mengubah logika positions pada fungsi simulate_trading dan menjalankan filenya lalu menjalankan notebooks 2
**btw hasilnya hancur lebur ya ges ya**
**Total Return: -99.97%**
**Annualized Return: -64.12%**
**Annualized Volatility: 87.66%**
**Sharpe Ratio: -0.73**
**Max Drawdown: -99.99%**
utk mengevaluasi aku mengubah fungsi simulate_trading dan logika tadi dengan logika kontrarian dan memperolah hasil yang lebih baik (kurasa):
**Total Return: 236057.22%**
**Annualized Return: 167.61%**
**Annualized Volatility: 87.61%**
**Sharpe Ratio: 1.91** --> I GOT ALMOST SHARPE RATIO 2 OMG
**Max Drawdown: -72.17%**
hasil ini membuktikan bahwa `MODEL YANG DILATIH PADA FITUR HALVING DAN ALPHA LAINNYA, SECARA KONSISTEN DAN ANDAL SALAH` dalam memprediksi arah. Tapi masih ada masalah yang besar yaitu max drawdown nya -72.17%. kita akan berusaha untuk membuat max drawdownnya < 30%

`2. mengidentifikasi visual`
menjalankan notebooks 2 dan melihat grafik stratgey cumulative returns, apakah grafiknya naik mulus? apakah itu hanya 1-2 trade hoki? dan dimana drawdown -72% itu terjadi? apakah saat crash covid?
**drawdown besar itu terjadi sekitar awal 2021 setelah puncak besar, lalu terjadi lagi di pertengahan 2022. ini adalah periode bear markey yang ganas dimana strateginya tertipu dan terus melawan tren**

`3. me-management risk`
saatnya menerapkan volatility sizing --> teknik paling umum utk memperbaik drawdown --> saat volatilitas pasar sedang tinggi, ambil posisi lebih kecil
crash di 2021 dan 2022 memiliki volatilitas yang sangat tinggi karena mengambil resiko penuh (position = 1), ubah ini menjadi (position = 1 atau -1) menjadi dinamis berdasarkan volatilitas.
lalu menambahkan filter rezim yaitu SMA_200 (simple moving average 200 hari) 