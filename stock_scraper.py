"""
Stock Data Scraper for FinIntel-Engine
Fetches OHLCV data from yfinance and adds technical indicators
Used by technical model for stock classification (Buy/Hold/Sell)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockScraper:
    """
    Fetches stock data from yfinance and computes technical indicators
    matching the 38 features used in the technical model.
    """
    
    # Technical features used in the GRU model (from your notebooks)
    TECHNICAL_FEATURES = [
        'rsi_5', 'rsi_10', 'rsi_14', 'rsi_15',                           # RSI
        'roc_10', 'mom_10',                                              # Rate of Change & Momentum
        'STOCHRSIk_14_14_3_3', 'STOCHRSId_14_14_3_3',                   # Stochastic RSI
        'cci_20', 'wr_14',                                               # CCI & Williams %R
        'KST_10_15_20_30_10_10_10_15', 'KSTs_9',                        # KST (Know Sure Thing)
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',               # MACD
        'sma_5', 'ema_5', 'sma_10', 'ema_10', 'sma_20', 'ema_20',       # Moving Averages
        'vwma_20',                                                       # Volume Weighted MA
        'BBL_20_2.0_2.0', 'BBM_20_2.0_2.0', 'BBU_20_2.0_2.0',          # Bollinger Bands
        'BBB_20_2.0_2.0', 'BBP_20_2.0_2.0',                             # BB Width & %B
        'KC_20_2', 'KCL_20_2', 'KCB_20_2', 'KCUe_20_2',                 # Keltner Channel
        'adr_14',                                                        # Average True Range
        'obv',                                                           # On Balance Volume
        'vpt',                                                           # Volume Price Trend
        'ad', 'adx_14'                                                   # Accumulation/Distribution
    ]
    
    STOCK_LIST = [
        # Tech stocks
        "AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "CRM", "CSCO", "IBM", "INTC", "INTU",
        "ORCL", "QCOM", "TXN", "AMD", "NOW", "PLTR", "MU", "PANW", "AMAT", "LRCX",
        
        # Internet/Media
        "GOOGL", "GOOG", "META", "DIS", "NFLX", "T", "VZ", "CMCSA", "TMUS", "CHTR",
        
        # Finance
        "JPM", "BAC", "WFC", "GS", "MS", "AXP", "V", "MA", "BRK-B", "BLK",
        "C", "COF", "SCHW", "MET", "PYPL",
        
        # Healthcare
        "JNJ", "UNH", "LLY", "ABBV", "PFE", "MRK", "TMO", "ABT", "DHR", "MDT",
        "GILD", "BMY", "ISRG", "AMGN", "CVS",
        
        # Retail/Consumer
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TGT", "BKNG",
        
        # Industrial
        "GM", "ABNB", "F", "BA", "CAT", "HON", "GE", "UNP", "UPS", "LMT",
        "RTX", "FDX", "DE", "MMM", "EMR", "GD",
        
        # Consumer Staples
        "PG", "KO", "PEP", "COST"
    ]
    
    def __init__(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        Initialize scraper with date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD). Defaults to 2 years ago.
            end_date: End date (YYYY-MM-DD). Defaults to today.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        
        self.start_date = start_date
        self.end_date = end_date
        self.data_cache = {}
        
        logger.info(f"Scraper initialized: {start_date} to {end_date}")
    
    def fetch_stock_data(self, ticker: str, progress: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a single stock.
        
        Args:
            ticker: Stock ticker symbol
            progress: Show download progress
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            logger.info(f"Fetching data for {ticker}...")
            data = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                progress=progress,
                repair=True
            )
            
            if data.empty:
                logger.warning(f"No data found for {ticker}")
                return None
            
            # Add ticker column for tracking
            data['Ticker'] = ticker
            self.data_cache[ticker] = data
            
            logger.info(f"Successfully fetched {len(data)} rows for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {str(e)}")
            return None
    
    def fetch_multiple_stocks(self, tickers: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks.
        
        Args:
            tickers: List of ticker symbols. Defaults to STOCK_LIST.
            
        Returns:
            Dictionary of {ticker: dataframe}
        """
        if tickers is None:
            tickers = self.STOCK_LIST
        
        data = {}
        for ticker in tickers:
            df = self.fetch_stock_data(ticker, progress=False)
            if df is not None:
                data[ticker] = df
        
        logger.info(f"Successfully fetched data for {len(data)}/{len(tickers)} stocks")
        return data
    
    @staticmethod
    def add_technical_features(df: pd.DataFrame, 
                              ticker: Optional[str] = None) -> pd.DataFrame:
        """
        Add all 38 technical indicators to OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            ticker: Stock ticker (for logging)
            
        Returns:
            DataFrame with technical indicators added
        """
        try:
            df = df.copy()
            
            # Ensure proper column names
            df.columns = df.columns.str.capitalize()
            
            # Calculate all technical indicators using pandas_ta
            
            # 1. RSI (Relative Strength Index)
            df['rsi_5'] = ta.rsi(df['Close'], length=5)
            df['rsi_10'] = ta.rsi(df['Close'], length=10)
            df['rsi_14'] = ta.rsi(df['Close'], length=14)
            df['rsi_15'] = ta.rsi(df['Close'], length=15)
            
            # 2. Rate of Change & Momentum
            df['roc_10'] = ta.roc(df['Close'], length=10)
            df['mom_10'] = ta.mom(df['Close'], length=10)
            
            # 3. Stochastic RSI
            stoch_rsi = ta.stochrsi(df['Close'], length=14, rsi_length=14, k=3, d=3)
            if stoch_rsi is not None:
                df['STOCHRSIk_14_14_3_3'] = stoch_rsi.iloc[:, 0]
                df['STOCHRSId_14_14_3_3'] = stoch_rsi.iloc[:, 1]
            
            # 4. CCI (Commodity Channel Index) & Williams %R
            df['cci_20'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
            df['wr_14'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
            
            # 5. KST (Know Sure Thing)
            kst = ta.kst(df['Close'], length=[10, 15, 20, 30], signal=9)
            if kst is not None:
                df['KST_10_15_20_30_10_10_10_15'] = kst.iloc[:, 0]
                df['KSTs_9'] = kst.iloc[:, 1]
            
            # 6. MACD (Moving Average Convergence Divergence)
            macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            if macd is not None:
                df['MACD_12_26_9'] = macd.iloc[:, 0]
                df['MACDh_12_26_9'] = macd.iloc[:, 2]
                df['MACDs_12_26_9'] = macd.iloc[:, 1]
            
            # 7. Moving Averages
            df['sma_5'] = ta.sma(df['Close'], length=5)
            df['ema_5'] = ta.ema(df['Close'], length=5)
            df['sma_10'] = ta.sma(df['Close'], length=10)
            df['ema_10'] = ta.ema(df['Close'], length=10)
            df['sma_20'] = ta.sma(df['Close'], length=20)
            df['ema_20'] = ta.ema(df['Close'], length=20)
            
            # 8. Volume Weighted Moving Average
            df['vwma_20'] = ta.vwma(df['Close'], df['Volume'], length=20)
            
            # 9. Bollinger Bands
            bb = ta.bbands(df['Close'], length=20, std=2.0)
            if bb is not None:
                df['BBL_20_2.0_2.0'] = bb.iloc[:, 0]
                df['BBM_20_2.0_2.0'] = bb.iloc[:, 1]
                df['BBU_20_2.0_2.0'] = bb.iloc[:, 2]
                df['BBB_20_2.0_2.0'] = bb.iloc[:, 3]
                df['BBP_20_2.0_2.0'] = bb.iloc[:, 4]
            
            # 10. Keltner Channels
            kc = ta.kc(df['High'], df['Low'], df['Close'], length=20, scalar=2)
            if kc is not None:
                df['KC_20_2'] = kc.iloc[:, 1]
                df['KCL_20_2'] = kc.iloc[:, 0]
                df['KCB_20_2'] = kc.iloc[:, 3]
                df['KCUe_20_2'] = kc.iloc[:, 2]
            
            # 11. Average True Range
            df['adr_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            
            # 12. Volume Indicators
            df['obv'] = ta.obv(df['Close'], df['Volume'])
            df['vpt'] = ta.vpt(df['Close'], df['Volume'])
            
            # 13. Accumulation/Distribution
            df['ad'] = ta.ad(df['High'], df['Low'], df['Close'], df['Volume'])
            df['adx_14'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)
            
            logger.info(f"Technical features added for {ticker or 'stock'}")
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical features: {str(e)}")
            return df
    
    def process_stock(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch and process a single stock (fetch + add features).
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with OHLCV + technical features
        """
        df = self.fetch_stock_data(ticker)
        if df is not None:
            df = self.add_technical_features(df, ticker)
        return df
    
    def process_multiple_stocks(self, tickers: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Process multiple stocks in batch.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary of processed dataframes
        """
        if tickers is None:
            tickers = self.STOCK_LIST
        
        processed_data = {}
        for ticker in tickers:
            df = self.process_stock(ticker)
            if df is not None:
                processed_data[ticker] = df
        
        return processed_data
    
    def save_to_csv(self, data: Dict[str, pd.DataFrame], output_dir: str = "stock_data"):
        """
        Save processed data to CSV files.
        
        Args:
            data: Dictionary of {ticker: dataframe}
            output_dir: Directory to save files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for ticker, df in data.items():
            filepath = os.path.join(output_dir, f"{ticker}_data.csv")
            df.to_csv(filepath)
            logger.info(f"Saved {ticker} to {filepath}")
    
    def get_cached_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get cached data for a ticker."""
        return self.data_cache.get(ticker)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    # Example 1: Fetch and process a single stock
    print("=" * 80)
    print("Example 1: Single Stock")
    print("=" * 80)
    
    scraper = StockScraper(
        start_date="2022-01-01",
        end_date="2024-12-31"
    )
    
    aapl_data = scraper.process_stock("AAPL")
    if aapl_data is not None:
        print(f"\nShape: {aapl_data.shape}")
        print(f"\nColumns: {list(aapl_data.columns)}")
        print(f"\nFirst few rows:\n{aapl_data.head()}")
    
    
    # Example 2: Fetch multiple stocks
    print("\n" + "=" * 80)
    print("Example 2: Multiple Stocks")
    print("=" * 80)
    
    stock_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    data = scraper.fetch_multiple_stocks(stock_list)
    print(f"\nFetched data for {len(data)} stocks")
    
    
    # Example 3: Add features and save
    print("\n" + "=" * 80)
    print("Example 3: Process and Save")
    print("=" * 80)
    
    processed = {}
    for ticker, df in data.items():
        processed[ticker] = scraper.add_technical_features(df, ticker)
    
    scraper.save_to_csv(processed, output_dir="backend/datasets/technical_datset/raw_data")
    
    
    # Example 4: Access OHLCV and technical features separately
    print("\n" + "=" * 80)
    print("Example 4: Feature Extraction")
    print("=" * 80)
    
    sample_df = processed["AAPL"]
    
    # OHLCV columns
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    ohlcv_data = sample_df[ohlcv_cols]
    
    # Technical features (38 features for model input)
    technical_cols = StockScraper.TECHNICAL_FEATURES
    technical_data = sample_df[technical_cols].dropna()
    
    print(f"\nOHLCV Data Shape: {ohlcv_data.shape}")
    print(f"Technical Features Shape: {technical_data.shape}")
    print(f"\nSample Technical Features:\n{technical_data.head()}")