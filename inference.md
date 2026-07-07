# inference.md — predict.py Design and Usage

## predict.py — full implementation

```python
# predict.py
import torch
import os
from transformers import AutoTokenizer
from model import BFSISentimentModel, SENTIMENT_ID2LABEL, IMPACT_ID2LABEL
from drive_storage import download_model_from_drive

_model     = None
_tokenizer = None


def load_model_once(output_dir: str = "output"):
    """Loads the model once and caches it in memory.
    Downloads from Drive if not already local."""
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    if not os.path.exists(f"{output_dir}/lora_adapters"):
        print("Model not found locally. Downloading from Google Drive...")
        download_model_from_drive(output_dir)

    print("Loading model into memory...")
    _model     = BFSISentimentModel.load(output_dir)
    _tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    print("Model ready.\n")

    return _model, _tokenizer


def predict(text: str, output_dir: str = "output") -> dict:
    """
    Predicts sentiment and market impact for a single text string.

    Returns:
        {
            "text":                  original input text,
            "sentiment":             "positive" | "neutral" | "negative",
            "sentiment_confidence":  float between 0 and 1,
            "impact":                "bullish" | "neutral" | "bearish",
            "impact_confidence":     float between 0 and 1
        }
    """
    model, tokenizer = load_model_once(output_dir)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        s_logits, i_logits = model(
            input_ids      = inputs["input_ids"],
            attention_mask = inputs["attention_mask"]
        )

    s_probs = torch.softmax(s_logits, dim=1)[0]
    i_probs = torch.softmax(i_logits, dim=1)[0]

    return {
        "text":                 text,
        "sentiment":            SENTIMENT_ID2LABEL[s_probs.argmax().item()],
        "sentiment_confidence": round(s_probs.max().item(), 3),
        "impact":               IMPACT_ID2LABEL[i_probs.argmax().item()],
        "impact_confidence":    round(i_probs.max().item(), 3),
        "all_sentiment_probs":  {
            SENTIMENT_ID2LABEL[i]: round(p.item(), 3)
            for i, p in enumerate(s_probs)
        },
        "all_impact_probs": {
            IMPACT_ID2LABEL[i]: round(p.item(), 3)
            for i, p in enumerate(i_probs)
        }
    }


def predict_batch(texts: list, output_dir: str = "output") -> list:
    """Predicts sentiment and impact for a list of texts."""
    return [predict(text, output_dir) for text in texts]


def print_result(result: dict):
    """Pretty-prints a single prediction result."""
    print(f"\nText: {result['text'][:80]}{'...' if len(result['text']) > 80 else ''}")
    print(f"  Sentiment : {result['sentiment']:10s}  (confidence: {result['sentiment_confidence']})")
    print(f"  Impact    : {result['impact']:10s}  (confidence: {result['impact_confidence']})")
    print(f"  All sentiment probs: {result['all_sentiment_probs']}")
    print(f"  All impact probs:    {result['all_impact_probs']}")


if __name__ == "__main__":
    # Demo predictions on BFSI-relevant examples
    test_texts = [
        "JPMorgan Q3 earnings beat analyst expectations by 12%, driven by strong IB revenue",
        "Federal Reserve signals aggressive rate hikes weighing on bank net interest margins",
        "Goldman Sachs names new chief financial officer effective next quarter",
        "Wells Fargo faces regulatory action over compliance failures in consumer division",
        "BlackRock AUM surpasses $10 trillion milestone for the first time",
        "Citigroup announces restructuring with 5000 job cuts across global operations",
        "Morgan Stanley wealth management division posts record quarterly inflows",
        "Bank of America credit loss provisions rise amid commercial real estate concerns"
    ]

    print("BFSI Sentiment and Market Impact Predictions")
    print("=" * 60)

    for text in test_texts:
        result = predict(text)
        print_result(result)

    print("\nDone.")
```

## How to use predict.py from other scripts

Once you have a trained model, you can import the `predict` function directly
into any other Python script — for example, wiring it into your earlier
market research pipeline from this conversation:

```python
# In your market research news_agent.py or synthesis_agent.py
from predict import predict

def analyze_news_sentiment(headline: str) -> dict:
    """Runs local FinBERT prediction — zero API tokens, zero cost."""
    return predict(headline)

# Returns: {"sentiment": "positive", "confidence": 0.91, "impact": "bullish", ...}
```

This replaces any LLM-based sentiment reasoning in the research pipeline with
a deterministic, free, locally-running model — same principle as replacing
LLM calls with yfinance and pandas in the financial agent.
