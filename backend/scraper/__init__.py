"""Scraper package exports."""

from .fundamental_financial_scraper import get_financial_data, safe_get
from .news_scraper import (
	fetch_et_news,
	fetch_google_news,
	fetch_news_api,
	fetch_reddit_news,
	fetch_twitter_news,
	get_news,
)

__all__ = [
	"fetch_et_news",
	"fetch_google_news",
	"fetch_news_api",
	"fetch_reddit_news",
	"fetch_twitter_news",
	"get_news",
	"get_financial_data",
	"safe_get",
]
