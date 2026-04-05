
from typing import List


import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import pandas_ta as ta
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("orchestrator")

ROOT = Path(__file__).resolve().parents[0]
def run_sentiment_model(news_list: List[str]) -> dict:
    model_dir = ROOT / 'backend' / 'models' / 'sentiment_model' / 'sentiment_expert_model_v1'
    if not model_dir.exists():
        logger.warning('Sentiment model not found; using simple lexicon fallback')
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        an = SentimentIntensityAnalyzer()
        comps = [an.polarity_scores(t)['compound'] for t in news_list]
        avg = float(np.mean(comps)) if comps else 0.0
        score = (avg + 1.0) / 2.0
        sentiment = 'positive' if avg > 0.05 else ('negative' if avg < -0.05 else 'neutral')
        return {"sentiment_score": score, "sentiment": sentiment}

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    scores = []
    model.eval()
    for text in news_list:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=0).numpy()
            score = float(probs[2] * 1.0 + probs[1] * 0.5 + probs[0] * 0.0)
            scores.append(score)

    avg_score = float(np.mean(scores)) if scores else 0.5
    sentiment = 'positive' if avg_score > 0.6 else ('negative' if avg_score < 0.4 else 'neutral')
    return {"sentiment_score": avg_score, "sentiment": sentiment}
