from typing import List
import logging
from pathlib import Path

import numpy as np

# ⚠️ DO NOT import heavy libs at top-level
# import tensorflow
# import torch
# import transformers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("sentiment")

ROOT = Path(__file__).resolve().parents[2]  # go to project root

# Global cache (so model loads only once)
_tokenizer = None
_model = None


def get_model():
    global _tokenizer, _model

    if _model is not None and _tokenizer is not None:
        return _tokenizer, _model

    model_dir = ROOT / "backend" / "models" / "sentiment_model" / "sentiment_expert_model_v1"

    if not model_dir.exists():
        logger.warning("⚠️ Sentiment model not found — using fallback")
        return None, None

    logger.info("🔄 Loading HuggingFace model (first time)...")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    _tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    _model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    _model.eval()

    logger.info("✅ Model loaded successfully")

    return _tokenizer, _model


def run_sentiment_model(news_list: List[str]) -> dict:
    if not news_list:
        return {"sentiment_score": 0.5, "sentiment": "neutral"}

    tokenizer, model = get_model()

    # 🔹 Fallback if model not available
    if model is None or tokenizer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(t)["compound"] for t in news_list]

        avg = float(np.mean(scores)) if scores else 0.0
        score = (avg + 1.0) / 2.0

        sentiment = (
            "positive" if avg > 0.05 else
            "negative" if avg < -0.05 else
            "neutral"
        )

        return {"sentiment_score": score, "sentiment": sentiment}

    # 🔹 Use ML model
    import torch

    scores = []

    for text in news_list:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=0).cpu().numpy()

            score = float(probs[2] * 1.0 + probs[1] * 0.5 + probs[0] * 0.0)
            scores.append(score)

    avg_score = float(np.mean(scores)) if scores else 0.5

    sentiment = (
        "positive" if avg_score > 0.6 else
        "negative" if avg_score < 0.4 else
        "neutral"
    )

    return {
        "sentiment_score": avg_score,
        "sentiment": sentiment
    }