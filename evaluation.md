# evaluation.md — Metrics, Evaluation, and What Good Performance Looks Like

## evaluate.py — full implementation

```python
# evaluate.py
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer
from model import BFSISentimentModel, SENTIMENT_ID2LABEL, IMPACT_ID2LABEL
from dataset_prep import prepare_dataset
from drive_storage import download_model_from_drive
import os

def evaluate(csv_path: str = "data/training_data.csv"):
    """Runs full evaluation on the test split and prints detailed metrics."""

    # Download model from Drive if not local
    if not os.path.exists("output/lora_adapters"):
        print("Downloading model from Drive...")
        download_model_from_drive()

    print("Loading model...")
    model     = BFSISentimentModel.load()
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    splits    = prepare_dataset(csv_path)
    test_ds   = splits["test"]

    all_sentiment_preds  = []
    all_sentiment_labels = []
    all_impact_preds     = []
    all_impact_labels    = []

    print("Running evaluation...")
    model.eval()

    with torch.no_grad():
        for i in range(len(test_ds)):
            item = test_ds[i]
            input_ids      = torch.tensor([item["input_ids"]])
            attention_mask = torch.tensor([item["attention_mask"]])

            s_logits, i_logits = model(input_ids, attention_mask)

            all_sentiment_preds.append(s_logits.argmax(1).item())
            all_sentiment_labels.append(item["sentiment_label"])
            all_impact_preds.append(i_logits.argmax(1).item())
            all_impact_labels.append(item["impact_label"])

    # Map numeric predictions back to label strings
    s_preds_str  = [SENTIMENT_ID2LABEL[p] for p in all_sentiment_preds]
    s_labels_str = [SENTIMENT_ID2LABEL[l] for l in all_sentiment_labels]
    i_preds_str  = [IMPACT_ID2LABEL[p]    for p in all_impact_preds]
    i_labels_str = [IMPACT_ID2LABEL[l]    for l in all_impact_labels]

    print("\n" + "="*60)
    print("SENTIMENT CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(s_labels_str, s_preds_str,
                                 target_names=["positive", "neutral", "negative"]))

    print("\n" + "="*60)
    print("MARKET IMPACT CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(i_labels_str, i_preds_str,
                                 target_names=["bullish", "neutral", "bearish"]))

    print("\nSentiment Confusion Matrix:")
    print(confusion_matrix(s_labels_str, s_preds_str))
    print("\nImpact Confusion Matrix:")
    print(confusion_matrix(i_labels_str, i_preds_str))


if __name__ == "__main__":
    evaluate()
```

## What good performance looks like

On Financial PhraseBank with 500+ examples, FinBERT fine-tuned with LoRA typically
achieves:

| Metric | Acceptable | Good | Excellent |
|---|---|---|---|
| Sentiment accuracy | 75% | 82% | 88%+ |
| Impact accuracy | 65% | 75% | 82%+ |
| Macro F1 sentiment | 0.70 | 0.78 | 0.85+ |

Impact accuracy is always lower than sentiment because "bullish/bearish" requires
inferring market consequence from sentiment, which is a harder task — two models
that agree on "negative" sentiment may disagree on whether it's "bearish" (already
priced in vs surprise).

If accuracy is below the "acceptable" threshold, the most common causes in order of
likelihood are:

1. Dataset too small (fewer than 200 examples) — collect more labeled data
2. Class imbalance — too many "neutral" examples drowning out positive/negative signal
3. Label inconsistency — the same type of text labeled differently by the labeler
4. Learning rate too high — try 1e-5 instead of 2e-5
