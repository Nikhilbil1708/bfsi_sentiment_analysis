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

    # sklearn orders classes alphabetically unless `labels=` is given
    # explicitly, but `target_names` is applied positionally -- without
    # `labels=` here, target_names silently attaches to the wrong rows.
    sentiment_order = ["positive", "neutral", "negative"]
    impact_order    = ["bullish", "neutral", "bearish"]

    print("\n" + "="*60)
    print("SENTIMENT CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(s_labels_str, s_preds_str,
                                 labels=sentiment_order, target_names=sentiment_order))

    print("\n" + "="*60)
    print("MARKET IMPACT CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(i_labels_str, i_preds_str,
                                 labels=impact_order, target_names=impact_order))

    print(f"\nSentiment Confusion Matrix (rows/cols order: {sentiment_order}):")
    print(confusion_matrix(s_labels_str, s_preds_str, labels=sentiment_order))
    print(f"\nImpact Confusion Matrix (rows/cols order: {impact_order}):")
    print(confusion_matrix(i_labels_str, i_preds_str, labels=impact_order))


if __name__ == "__main__":
    evaluate()
