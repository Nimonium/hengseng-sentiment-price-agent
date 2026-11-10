# Apify Actors Live Verification & Execution Registry

This document records the Apify Actors used in the HengSeng Sentiment Price Prediction Agent, verified on the live Apify Store API (`https://api.apify.com/v2/acts/{id}` via `client.actor(id).get()`), their input schemas, actual live execution status, run IDs, and verified data availability.

---

## 1. Verified Apify Store Actors Metadata

| Social Source | Actor Name & Store ID | Owner | isPublic | Pricing Model | Verification Method |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LIHKG** | `claude_code_reviewer/lihkg-scraper` | `claude_code_reviewer` | `True` | `PAY_PER_EVENT` | Live `client.actor().get()` |
| **AAStocks** | `bovi/aastocks-scraper` | `bovi` | `True` | `PAY_PER_EVENT` | Live `client.actor().get()` |
| **Naver Discussion** | `hgservices/naver-stock-discussion-scraper` | `hgservices` | `True` | `PAY_PER_EVENT` | Live `client.actor().get()` |
| **Daum News** | `ardent_fork/daum-news` | `ardent_fork` | `True` | `PAY_PER_EVENT` | Live `client.actor().get()` |
| **TeamBlind (Blind)** | `hypebridge/blind-post-scraper` | `hypebridge` | `True` | `PAY_PER_EVENT` | Live `client.actor().get()` |

---

## 2. Live Actor Execution Log & Account Quota Status

### Account Quota & Billing Status
* **Apify Authenticated User**: `nimonium`
* **Account Usage Status**: The Apify console reports **$9.60 used against a $5.00 monthly free tier limit**.
* **Live Scraping Status**: Live actor scraping is disabled by Apify until the next monthly billing cycle resets on **2026-12-08** (`monthlyUsageCycle.endAt`).
* **Reproducibility Mode**: All pipeline execution for reviewers is fully reproducible offline via `--use-sample`, loading the committed datasets in `data/sample/` with zero tokens required.

### Live Run Registry & Usable Items in Dataset

| Platform | Actor ID | Apify Run ID | Run Status | Usable Items in Window | Usability / Inclusion Status | Scope Covered |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **LIHKG** (Test) | `claude_code_reviewer/lihkg-scraper` | `r4xU33xQfUJEvILws` | `SUCCEEDED` | - | Test run | 0700.HK keywords |
| **LIHKG** (Full) | `claude_code_reviewer/lihkg-scraper` | `GZVVLlijW4U5bxSNk` | `SUCCEEDED` | **4** | **Downloaded & Used** | All 5 HKEX Tickers |
| **AAStocks** (Test) | `bovi/aastocks-scraper` | `BNtpwWW3Uf1EfmBrA` | `SUCCEEDED` | - | Test run | Public news wire |
| **AAStocks** (Full) | `bovi/aastocks-scraper` | `ZIhFu2IJARtAxAjbT` | `SUCCEEDED` | **36** | **Downloaded & Used** | Real-time HKEX news wire |
| **Naver** (Test) | `hgservices/naver-stock-discussion-scraper` | `tTfXyhIOFxdYMry5a` | `SUCCEEDED` | - | Test run | 005930 (Samsung) |
| **Naver** (Full) | `hgservices/naver-stock-discussion-scraper` | `XDZAwjTEizX7OUn3n` | `SUCCEEDED` | **400** | **Downloaded & Used** | All 5 KRX Tickers (종목토론실) |
| **Daum** (Test) | `ardent_fork/daum-news` | `2cEGcNBbXozT9bMxe` | `SUCCEEDED` | - | Test run | Finance Section |
| **Daum** (Full) | `ardent_fork/daum-news` | `8cEgcwpp0ngIPkZzL` | `SUCCEEDED` | **200** | **Downloaded & Used** | Finance Section (KRX) |
| **Blind** (Test) | `hypebridge/blind-post-scraper` | `s1MUKPxf5iEIvqoek` | `SUCCEEDED` | 0 | Dataset locked; not used | Samsung search |
| **Blind** (Samsung) | `hypebridge/blind-post-scraper` | `ygXvOJdSq4PWogaq1` | `SUCCEEDED` | 0 | Dataset locked; not used | Samsung Electronics |
| **Blind** (SK Hynix)| `hypebridge/blind-post-scraper` | `hPs1ejWbCNS7SZmh4` | `SUCCEEDED` | 0 | Dataset locked; not used | SK Hynix Channel |
| **Blind** (Naver) | `hypebridge/blind-post-scraper` | `dcbibvgcZji6JUyX2` | `SUCCEEDED` | 0 | Dataset locked; not used | Naver Channel |
| **Blind** (Hyundai) | `hypebridge/blind-post-scraper` | `4dSOmfOrgx1Fyt7hT` | `SUCCEEDED` | 0 | Dataset locked; not used | Hyundai Motor Channel |
| **Blind** (Kakao) | `hypebridge/blind-post-scraper` | `yUnnBKn4axdAs5CB9` | `SUCCEEDED` | 0 | Dataset locked; not used | Kakao Channel |

> [!NOTE]
> **TeamBlind Post Data**: While the scraping runs succeeded on Apify's infrastructure, the datasets were locked on Apify's servers once the $5.00 quota limit was hit before local ingestion completed. Total usable Blind posts in `data/raw/` and `data/sample/` is **0**. No features or ablation models utilize Blind data.

---

## 3. Actor Input JSON & Parameters

### A. Hong Kong Exchange (HKEX)

#### 1. LIHKG Scraper (`claude_code_reviewer/lihkg-scraper`)
* **Mode**: `search`
* **Date Filter**: `dateFrom: "10d"`
* **Proxy Configuration**: `{"useApifyProxy": true}`
* **Exact Input JSON**:
  ```json
  {
    "mode": "search",
    "keywords": ["0700", "騰訊", "9988", "阿里巴巴", "0005", "滙豐", "1299", "友邦", "3690", "美團"],
    "maxItems": 50,
    "dateFrom": "10d",
    "proxyConfiguration": {
      "useApifyProxy": true
    }
  }
  ```
* **Output Fields**: `thread_id`, `title`, `content`, `author`, `created_at` (ISO 8601 UTC), `last_reply_at`, `like_count`, `reply_count`.

#### 2. AAStocks Scraper (`bovi/aastocks-scraper`)
* **Feed**: `latest-news` (Public real-time wire)
* **Language**: `tc` (Traditional Chinese)
* **Exact Input JSON**:
  ```json
  {
    "feed": "latest-news",
    "language": "tc",
    "maxResults": 150,
    "proxyConfiguration": {
      "useApifyProxy": true,
      "apifyProxyGroups": ["RESIDENTIAL"]
    }
  }
  ```
* **Output Fields**: `article_id`, `title`, `url`, `source`, `feed`, `language`.

---

### B. South Korea Exchange (KRX)

#### 1. Naver Stock Discussion Scraper (`hgservices/naver-stock-discussion-scraper`)
* **Stocks**: All 5 KRX stock codes (`005930`, `000660`, `035420`, `005380`, `035720`)
* **Date Filter**: `dateFrom: "2026-10-30"` (Start of 10-calendar-day window)
* **Max Results**: `200` posts per stock
* **Exact Input JSON**:
  ```json
  {
    "stocks": ["005930", "000660", "035420", "005380", "035720"],
    "dateFrom": "2026-10-30",
    "maxResults": 200
  }
  ```
* **Output Fields**: `postId`, `stockCode`, `stockName`, `title`, `text`, `writtenAt` (ISO 8601 KST), `views`, `likes`, `dislikes`, `author`.

#### 2. Daum Finance News Scraper (`ardent_fork/daum-news`)
* **Sections**: `["finance"]`
* **Max Items**: `100`
* **Fetch Articles**: `false`
* **Exact Input JSON**:
  ```json
  {
    "sections": ["finance"],
    "maxItems": 100,
    "fetchArticles": false
  }
  ```
* **Output Fields**: `id`, `url`, `title`, `thumbnail`, `publisher`, `section`.

#### 3. TeamBlind Post Scraper (`hypebridge/blind-post-scraper`)
* **Search Query**: Korean ticker local name (`searchQuery: "삼성전자"`)
* **Status**: Runs executed successfully, but datasets locked on Apify platform due to free-tier quota ceiling ($9.60 used vs $5.00 limit). Usable posts in repo: 0.

---

## 4. Apify Store Store-Wide Discovery & Feasibility Audit (Tested Actors)

In accordance with strict verification and zero-fabrication rules, candidate actors on the Apify Store were searched via `client.store().list()` and tested live with `maxItems=5`:

1. **Futu / Moomoo Community**:
   - `client.store().list(search="futu")` & `client.store().list(search="moomoo")`: Returned **0 actors**. No public Apify actor exists for Futu/Moomoo.
2. **ETNet HK Financial Forum**:
   - `client.store().list(search="etnet")`: Returned **0 actors**. No actor exists.
3. **AAStocks Forum Comments**:
   - `bovi/aastocks-scraper` is verified and public, but its author only built scrapers for the news/commentary wire (`latest-news`), not user comments.
4. **Telegram Channels**:
   - `truefetch/telegram-channel-message` (Run ID: `cb05PcLkOCbpuiaSs`): Tested live; returned 0 items because the public preview was not enabled for the channel.
   - `tri_angle/telegram-scraper` (Run ID: `rJObXtQNc5uxUWNcB`): Tested live; failed because financial channels required interactive login.
5. **TeamBlind (Blind)**:
   - `hypebridge/blind-post-scraper`: Succeeded on Apify, but datasets locked behind quota before local download could complete.
