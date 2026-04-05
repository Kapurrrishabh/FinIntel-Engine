# Scraper and Orchestration Handoff

This folder contains the stock and news scraping utilities that feed the ML pipeline. The current code is close to a working demo, but it still needs cleanup before it can be treated as a stable data layer for the orchestrator.

## What Exists Now

- [stock_scraper.py](stock_scraper.py) downloads OHLCV data from `yfinance` and adds a set of technical indicators.
- [news_scraper_et_twitter.py](news_scraper_et_twitter.py) fetches news from Economic Times and Twitter/X, stores results in SQLite, and applies sentiment analysis.
- [run_orchestrator.py](../../run_orchestrator.py) combines technical, sentiment, fundamental, and risk signals into a final decision.
- [../stock_scraper.py](../stock_scraper.py) is a duplicate stock scraper outside this folder and should be treated as a maintenance risk until one canonical location is chosen.

## What Is Still Left To Do

### 1. Unify the technical feature schema

The biggest gap is that the inference path and the model schema are not fully aligned.

- [run_orchestrator.py](../../run_orchestrator.py) builds technical features that do not exactly match [feature_columns.json](../models/technical_model/feature_columns.json).
- The model expects columns such as `atr_14`, `KCLe_20_2`, `KCBe_20_2`, `KCUe_20_2`, `efi`, `NVI_1`, `PVI`, `PVIe_255`, and `Log_Ret`.
- The current orchestrator feature builder creates some of these, but not all of them consistently, and some indicator names come back from `pandas_ta` under different names.
- The stock scraper under [stock_scraper.py](stock_scraper.py) also uses a different indicator set (`adr_14`, `KC_20_2`, `KCL_20_2`, `KCB_20_2`, `vpt`, `adx_14`), so it should not be assumed to be model-compatible as-is.

Recommended action:

1. Create one canonical technical feature builder.
2. Make both training and inference use that same builder.
3. Verify the final dataframe contains exactly the columns in [feature_columns.json](../models/technical_model/feature_columns.json) in the same order.
4. Add a smoke test that fails if any expected feature is missing or renamed.

### 2. Replace demo data in the orchestrator

The current orchestration flow is still demo-oriented.

- The stock symbol is hard-coded to `AAPL`.
- News sentiment is driven by a small hard-coded sample list instead of the news scraper output.
- Fundamental inputs are also hard-coded sample values.

Recommended action:

1. Accept the ticker as a command-line argument or config input.
2. Pull the news list from `NewsScraper` instead of using sample strings.
3. Feed real company fundamentals into `fundamental_analysis` instead of static placeholders.
4. Make the pipeline return a structured result for one or many tickers.

### 3. Decide on the canonical news source behavior

[news_scraper_et_twitter.py](news_scraper_et_twitter.py) is functional, but part of it is still a fallback/demo implementation.

- Twitter/X scraping falls back to generated sample tweets when live scraping fails.
- Economic Times parsing depends on HTML structure that may change.
- The scraper stores data in SQLite, but there is no documented policy for cleanup, deduplication beyond the unique constraint, or retention.

Recommended action:

1. Decide whether the sample tweet fallback is acceptable for production or only for offline demos.
2. Add stronger parsing and date normalization for Economic Times articles.
3. Standardize the schema returned by every news source so the sentiment layer can consume it directly.
4. Document how the SQLite cache should be initialized, migrated, and reused.

### 4. Wire sentiment analysis into the real pipeline

The sentiment code is mostly complete, but it is still isolated from the orchestrator.

- `SentimentAnalyzer` can analyze a dataframe, but `run_orchestrator.py` currently passes hard-coded example strings to its own sentiment helper.
- The orchestrator should consume articles from [news_scraper_et_twitter.py](news_scraper_et_twitter.py), extract the relevant text, and score that real content.

Recommended action:

1. Feed actual news rows into sentiment scoring.
2. Preserve source metadata and confidence scores in the orchestrator output.
3. Decide whether the FinBERT model or the fallback sentiment logic is the default when the model is unavailable.

### 5. Clean up runtime assumptions and dependencies

There are still environment assumptions that need to be made explicit.

- `run_orchestrator.py` depends on TensorFlow/Keras, Transformers, PyTorch, `pandas_ta`, `yfinance`, and `vaderSentiment`.
- The repo already lists those dependencies in [requirements.txt](../../requirements.txt), but the orchestrator still needs a verified runtime setup that actually loads them cleanly.
- The current editor analysis also reports a TensorFlow import resolution issue in [run_orchestrator.py](../../run_orchestrator.py), so the environment and import path should be validated in a real execution run.

Recommended action:

1. Confirm the correct Python environment is used for the orchestrator.
2. Add a startup check that prints missing optional dependencies clearly.
3. Separate hard requirements from fallback-only packages.

### 6. Add tests and smoke checks

There are no obvious tests around the scraper/orchestrator handoff yet.

Recommended minimum tests:

1. A feature-schema test that compares generated technical columns against [feature_columns.json](../models/technical_model/feature_columns.json).
2. A news-schema test that verifies each scraper returns the expected keys.
3. A pipeline smoke test that runs one ticker end to end with mocked network calls.
4. A fallback test for the sentiment and technical branches when model files are unavailable.

## Suggested Implementation Order

1. Fix the technical feature mismatch first.
2. Remove the hard-coded demo inputs from the orchestrator.
3. Decide whether the Twitter sample fallback stays or becomes test-only.
4. Normalize the news output schema.
5. Add the smoke tests.
6. Remove or repurpose the duplicate `../stock_scraper.py` file.

## Short Version

If you only do three things next, make them these:

1. Align feature engineering with [feature_columns.json](../models/technical_model/feature_columns.json).
2. Replace the hard-coded demo inputs in [run_orchestrator.py](../../run_orchestrator.py).
3. Decide on one canonical stock scraper file and remove the duplication.

