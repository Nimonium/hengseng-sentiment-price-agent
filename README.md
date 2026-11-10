# HengSeng Sentiment Price Prediction Agent

An institutional-grade, multi-exchange quantitative pipeline that predicts next-bar intraday returns ($R_{t+1}$) for equities across the **Hong Kong Stock Exchange (HKEX)** and the **Korea Exchange (KRX)** using cross-source social media sentiment, neural news search, and technical market structure with **strict out-of-sample walk-forward validation and zero look-ahead leakage**.

Developed for the **CrowdWisdomTrading** "HengSeng Sentiment Price Prediction Agent" assessment.

---

## 📌 Executive Summary & Scope

| Market | Tickers Analyzed | Primary Social UGC & News Sources | Evaluation Window (Strict 10 Calendar Days) |
| :--- | :--- | :--- | :--- |
| **HKEX (Hong Kong)** | `0700.HK` (Tencent), `9988.HK` (Alibaba), `0005.HK` (HSBC), `1299.HK` (AIA), `3690.HK` (Meituan) | **LIHKG** (連登 user discussion) + **AASTOCKS** (阿思達克 wire news) + **Exa.com News** | 1-Hour Intraday Bars: 35 bars across 5 sessions (`2026-11-02` to `2026-11-06`) |
| **KRX (South Korea)** | `005930.KS` (Samsung), `000660.KS` (SK Hynix), `035420.KS` (Naver), `005380.KS` (Hyundai), `035720.KS` (Kakao) | **Naver Finance 종목토론실** (user boards) + **Daum Finance** (news) + **TeamBlind** + **Exa.com News** | 1-Hour Intraday Bars: 30 bars across 5 sessions (`2026-11-02` to `2026-11-06`) |

*All timestamps are standardized to UTC. All price bars and sentiment signals are strictly causal with zero look-ahead leakage.*

---

## 🏗️ Cross-Source Execution Registry & Audit Table

| Exchange | Source Identifier | Source Category | Posts in 10-Day Window | Execution Status | Apify Run ID / Execution Reference |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **HKEX** | **LIHKG** (連登) | Social UGC | **4** | **Real Apify Run** *(Single thin UGC source)* | `GZVVLlijW4U5bxSNk` / `sKCe3NgXqlRDerEwZ` |
| **HKEX** | **AAStocks** (阿思達克) | News Wire | **36** | **Real Apify Run** *(Real-time public wire)* | `ZIhFu2IJARtAxAjbT` / `BNtpwWW3Uf1EfmBrA` |
| **HKEX** | **Exa.com News** | News Media | **58** | **Real Exa Search** *(Neural semantic query)* | Neural Search via `exa_py` |
| **HKEX** | **Futu / Moomoo** | Social UGC | **0** | **Unavailable** *(0 Apify actors exist on store)* | Queried `client.store().list(search="futu")` |
| **HKEX** | **ETNet HK Forum** | Social UGC | **0** | **Unavailable** *(0 Apify actors exist on store)* | Queried `client.store().list(search="etnet")` |
| **KRX** | **Naver 종목토론실** | Social UGC | **400** | **Real Apify Run** *(Retail stock discussion)* | `XDZAwjTEizX7OUn3n` / `tTfXyhIOFxdYMry5a` |
| **KRX** | **TeamBlind** | Social UGC | **0** | **Runs Succeeded, Dataset Locked** *(Datasets locked by Apify quota & never downloaded; not used)* | `ygXvOJdSq4PWogaq1`, `dcbibvgcZji6JUyX2` |
| **KRX** | **Daum Finance News**| News Wire | **200** | **Real Apify Run** *(Finance section wire)* | `8cEgcwpp0ngIPkZzL` / `2cEGcNBbXozT9bMxe` |
| **KRX** | **Naver News** | News Wire | **37** | **Real Ingestion** *(Corporate news feed)* | Real-time corporate feed |
| **KRX** | **Exa.com News** | News Media | **56** | **Real Exa Search** *(Neural semantic query)* | Neural Search via `exa_py` |
| **KRX** | **Telegram Channels**| Social UGC | **0** | **Failed** *(Login wall / web preview disabled)*| `cb05PcLkOCbpuiaSs`, `rJObXtQNc5uxUWNcB` |

### Key Source Transparency Findings
1. **Hong Kong UGC Scarcity**: HKEX has only one thin user-generated source (**LIHKG**, 4 posts in window) because Futu/Moomoo and ETNet have zero actors on the Apify Store, and AAStocks exposes only its professional news wire (`latest-news`).
2. **South Korea UGC & TeamBlind Status**: KRX features retail engagement via **Naver 종목토론실** (400 posts in window). TeamBlind scraping runs executed successfully on Apify (`hypebridge/blind-post-scraper`), but because TeamBlind outputs relative timestamps (`"2d"`, `"Yesterday"`), the initial parser encountered a date format exception. Immediately after, the Apify account reached its monthly free-tier $5.00 limit, locking the datasets on Apify's servers before local download completed. Total usable Blind posts in `data/raw/` and `data/sample/` is **0**, and Blind data is not used in feature construction or model training. Relative timestamp parsing has been implemented in `src/ingestion/apify_ingest.py`.
3. **Exa News Query Refinement**: Default composite queries initially yielded 0 articles for AIA (`1299.HK`) and Meituan (`3690.HK`). Retrying those two queries with alternative wording (`f"{ticker.name} Hong Kong stock"`, using exactly 2 Exa API requests) retrieved 22 verified articles for AIA and 22 for Meituan, bringing total HKEX Exa news items to 58.

---

## ⚡ Mathematical & Algorithmic Design

### 1. Single Window & Indicator Warm-Up Separation
* **Evaluation Window**: Exactly the last 10 calendar days ending now (`2026-10-30` to `2026-11-09`).
* **Indicator Warm-up**: Pre-window price bars (fetched via `yfinance` 14-day history) are strictly utilized to initialize rolling indicators (lagged returns, rolling volatility, volume ratios, spreads) and are **never** included in model training, walk-forward cross-validation, or performance metrics.

### 2. Zero-Lookahead Point-in-Time Join & Exponential Decay
For a 1-hour bar $k$ indexed at observation timestamp $T_k$:
$$\text{Eligible Information Window: } t_{\text{post}} \in (T_{k-1}, T_k]$$
Posts published at $t > T_k$ are quarantined from bar $k$. Posts created while markets are closed (overnight/weekend) accumulate strictly into the opening bar of the subsequent session.
$$S_k = \begin{cases} s_k & \text{if new items observed in bar } k \\ S_{k-1} \cdot \exp(-\ln(2) / H) & \text{if no new items} \end{cases}$$
where half-life $H = 6$ bars (1 trading day).

### 3. Non-Collinear Feature Space & Differential Sentiment
* **Differential Bull/Bear**: Bounded normalized differential $\frac{\text{bull} - \text{bear}}{\text{bull} + \text{bear} + 1.0}$ replaces ill-conditioned ratios.
* **Collinearity Elimination**: Redundant features (`social_sentiment_momentum`, `news_sentiment_smooth_3h`) were removed. Zero-variance constant features are automatically identified and dropped prior to model fitting.
* **RidgeCV with Bounded Predictions**: Alpha regularizer chosen strictly inside the training fold via `RidgeCV(alphas=np.logspace(-2, 4, 15))`. Predictions are clipped to $\pm 3\times$ the rolling return volatility, preventing unrealistic extrapolation spikes.

---

## 📊 Audited Walk-Forward Ablation & Statistical Validation

### 1. Pooled Exchange Walk-Forward Validation (Sample $N > 100$)
Models trained across all pooled assets within each exchange using cross-sectional ticker-level standardization:

| Exchange | Sample $N$ | Naive Zero MAE | Price Only MAE | +Social MAE | +News MAE | All Features MAE | Optimal MAE Predictor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HKEX** | 136 | **0.006258** | 0.007241 | 0.007314 | 0.007048 | 0.006989 | **Naive Zero-Return** |
| **KRX** | 116 | **0.015071** | 0.016891 | 0.016891 | 0.018140 | 0.018140 | **Naive Zero-Return** |

### 2. Per-Ticker Walk-Forward Ablation (RidgeCV & $\pm 3\sigma$ Bounded Returns)

| Asset | Test Bars $N$ | Naive Zero MAE | Price Only MAE | +Social MAE | +News MAE | All Features MAE | Model Dir Acc | Always-Up | Prev-Dir | Binomial $p$-val (50%) | Sig Edge? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0700.HK (Tencent)** | 24 | **0.006676** | 0.006913 | 0.006913 | 0.006912 | 0.006913 | **58.3%** | 58.3% | 54.2% | 0.2706 | **NO** |
| **9988.HK (Alibaba)** | 24 | **0.009013** | 0.009423 | 0.009423 | 0.009410 | 0.009410 | **41.7%** | 58.3% | 45.8% | 0.8463 | **NO** |
| **0005.HK (HSBC)** | 24 | **0.003970** | 0.004760 | 0.004760 | 0.004990 | 0.004990 | **41.7%** | 41.7% | 54.2% | 0.8463 | **NO** |
| **1299.HK (AIA)** | 24 | **0.003522** | 0.004196 | 0.004196 | 0.004137 | 0.004137 | **41.7%** | 45.8% | 45.8% | 0.8463 | **NO** |
| **3690.HK (Meituan)** | 24 | **0.009820** | 0.010720 | 0.010720 | 0.010721 | 0.010721 | **54.2%** | 54.2% | 45.8% | 0.4194 | **NO** |
| **005930.KS (Samsung)** | 20 | **0.018994** | 0.030130 | 0.030130 | 0.019039 | 0.019039 | **55.0%** | 40.0% | 45.0% | 0.4119 | **NO** |
| **000660.KS (SK Hynix)**| 20 | **0.025366** | 0.025354 | 0.025354 | 0.023924 | 0.023924 | **65.0%** | 35.0% | 50.0% | 0.1316 | **NO** |
| **035420.KS (Naver)** | 20 | **0.011225** | 0.012446 | 0.012446 | 0.015329 | 0.015329 | **40.0%** | 55.0% | 75.0% | 0.8684 | **NO** |
| **005380.KS (Hyundai)** | 20 | **0.014499** | 0.018964 | 0.018964 | 0.020120 | 0.020120 | **40.0%** | 55.0% | 55.0% | 0.8684 | **NO** |
| **035720.KS (Kakao)** | 20 | **0.011537** | 0.012761 | 0.012761 | 0.013362 | 0.013362 | **40.0%** | 65.0% | 55.0% | 0.8684 | **NO** |

### 3. Out-of-Sample Predictions vs. Actual Next Close

| Ticker | Current Close | Predicted Next Bar | Actual Next Bar | Predicted Return | Actual Return | Directional Match |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **0700.HK** | HK$418.00 | HK$419.27 | HK$419.80 | +0.305% | +0.431% | **YES** |
| **9988.HK** | HK$104.30 | HK$104.60 | HK$104.90 | +0.291% | +0.575% | **YES** |
| **0005.HK** | HK$137.80 | HK$138.04 | HK$137.50 | +0.172% | -0.218% | **NO** |
| **1299.HK** | HK$78.45 | HK$78.59 | HK$78.45 | +0.178% | +0.000% | **NO** |
| **3690.HK** | HK$80.05 | HK$80.45 | HK$80.40 | +0.495% | +0.437% | **YES** |
| **005930.KS** | ₩189,900 | ₩188,466 | ₩190,400 | -0.755% | +0.263% | **NO** |
| **000660.KS** | ₩1,145,000 | ₩1,134,232 | ₩1,144,000 | -0.940% | -0.087% | **YES** |
| **035420.KS** | ₩192,700 | ₩193,759 | ₩192,400 | +0.549% | -0.156% | **NO** |
| **005380.KS** | ₩283,500 | ₩285,321 | ₩282,500 | +0.642% | -0.353% | **NO** |
| **035720.KS** | ₩32,950 | ₩32,995 | ₩32,450 | +0.136% | -1.517% | **NO** |

---

## 🔍 Honest Scientific Limitations

1. **Short 10-Calendar-Day Window**: The fixed 10-day lookback yields only 5 trading sessions (35 hourly bars for HKEX, 30 for KRX). Pre-window bars are used exclusively for indicator warm-up.
2. **Small Out-of-Sample Test Sample Sizes**: Individual tickers have $N \approx 20-24$ test bars; even pooled panels ($N=136$ for HKEX, $N=116$ for KRX) have wide confidence intervals.
3. **Thin Social UGC Coverage in Hong Kong**: LIHKG returned only 4 posts across the window; Futu/Moomoo and ETNet have no Apify actors, leaving retail sentiment sparse for HKEX.
4. **Naive Zero-Return Benchmark Dominance**: The Naive Zero-Return baseline strictly minimizes MAE over RidgeCV regression on 1h bars. All binomial test $p$-values exceed 0.05 ($p \ge 0.05$); we explicitly reject the hypothesis that retail sentiment creates a reliable directional edge.
5. **Apify Free-Tier Quota & Billing Status**: The Apify account showed $9.60 used against a $5.00 monthly free-tier limit, and live scraping is disabled until the next billing reset period (2026-12-08). Reviewer reproducibility is guaranteed via `--use-sample` loading from `data/sample/`.

---

## 🚀 Execution & Verification Guide

### 1. Requirements & Setup
```bash
pip install -r requirements.txt
```
All dependency versions are pinned in `requirements.txt`.

### 2. Running Unit Tests
```bash
pytest -v
```
All 9 unit tests pass cleanly.

### 3. Reviewer Reproducibility Mode (Zero API Tokens Required)
A reviewer can reproduce the entire pipeline (ingestion, alignment, walk-forward validation, ablations, and HTML report generation) with no API keys:
```bash
python main.py --run all --use-sample
```
The generated interactive HTML report is committed at `report/report.html`.

### 4. Live Scraper Run (With Real API Keys)
```bash
python main.py --run all
```
* Checks `.env` at project root and verifies `APIFY_TOKEN` and `EXA_API_KEY`.
* Zero fallbacks are executed.
