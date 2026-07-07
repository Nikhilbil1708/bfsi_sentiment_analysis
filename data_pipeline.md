# data_pipeline.md — Dataset Sources, Labeling, and Preparation

## Label schema — strictly enforced

The training CSV must use exactly these values. Any other values cause a validation
error at training start — do not silently ignore bad labels.

```
sentiment_label: "positive" | "neutral" | "negative"
impact_label:    "bullish"  | "neutral" | "bearish"
```

## CSV format

```csv
text,sentiment_label,impact_label
"JPMorgan Q3 earnings beat analyst expectations by 12%",positive,bullish
"Fed signals aggressive rate hikes weighing on bank margins",negative,bearish
"Goldman Sachs names new chief financial officer",neutral,neutral
```

Rules:
- text column: any length, but truncated to 128 tokens at training time
- No header row variations — exactly these three column names
- UTF-8 encoding
- Minimum 100 examples to train, 300+ recommended, 500+ ideal for good generalization

## Data sources for BFSI sentiment (where to get labeled examples)

### Source 1 — Financial PhraseBank (use this first, already labeled)

Available on HuggingFace, created by financial domain experts, ~4,845 sentences.
Labels are positive/neutral/negative only — you add impact labels manually or
derive them programmatically (positive sentiment on earnings = bullish, etc.).

```python
from datasets import load_dataset
ds = load_dataset("takala/financial_phrasebank", "sentences_allagree")
```

### Source 2 — Your own BFSI examples (more domain-specific)

Collect from these sources and label manually using label_tool.py:
- News headlines about JPM, GS, MS, BAC, WFC, C, BLK (use Tavily or news API)
- Earnings call transcript snippets from your earlier market research project
- SEC 8-K filing key passages (major events section)
- Fed policy announcement headlines

### Source 3 — FiQA Sentiment dataset

```python
ds = load_dataset("financial_sentiment", "fiqa")
```

## dataset_prep.py — full implementation

```python
# dataset_prep.py
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer

SENTIMENT_LABELS = {"positive": 0, "neutral": 1, "negative": 2}
IMPACT_LABELS    = {"bullish":  0, "neutral": 1, "bearish":  2}
VALID_SENTIMENTS = set(SENTIMENT_LABELS.keys())
VALID_IMPACTS    = set(IMPACT_LABELS.keys())
MAX_LENGTH       = 128   # keep short for CPU speed — sufficient for headlines


def validate_csv(csv_path: str):
    """Validates the training CSV before any training starts.
    Raises ValueError with a clear message if anything is wrong."""
    df = pd.read_csv(csv_path)

    required_columns = {"text", "sentiment_label", "impact_label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    bad_sentiments = df[~df["sentiment_label"].isin(VALID_SENTIMENTS)]
    if not bad_sentiments.empty:
        raise ValueError(
            f"Invalid sentiment_label values found:\n"
            f"{bad_sentiments[['text', 'sentiment_label']].head(5)}\n"
            f"Valid values: {VALID_SENTIMENTS}"
        )

    bad_impacts = df[~df["impact_label"].isin(VALID_IMPACTS)]
    if not bad_impacts.empty:
        raise ValueError(
            f"Invalid impact_label values found:\n"
            f"{bad_impacts[['text', 'impact_label']].head(5)}\n"
            f"Valid values: {VALID_IMPACTS}"
        )

    if len(df) < 100:
        print(f"WARNING: Only {len(df)} examples found. 300+ recommended for good results.")

    print(f"CSV validated: {len(df)} examples, all labels valid")
    print(f"Sentiment distribution:\n{df['sentiment_label'].value_counts()}")
    print(f"Impact distribution:\n{df['impact_label'].value_counts()}")
    return df


def prepare_dataset(
    csv_path: str,
    model_name: str = "ProsusAI/finbert",
    test_size: float = 0.2
):
    """Loads, validates, tokenizes, and splits the dataset.
    Returns a dict with 'train' and 'test' splits."""
    df        = validate_csv(csv_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_and_encode(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )
        tokenized["sentiment_label"] = [
            SENTIMENT_LABELS[l] for l in examples["sentiment_label"]
        ]
        tokenized["impact_label"] = [
            IMPACT_LABELS[l] for l in examples["impact_label"]
        ]
        return tokenized

    dataset  = Dataset.from_pandas(df)
    tokenized = dataset.map(tokenize_and_encode, batched=True,
                            remove_columns=["text", "sentiment_label", "impact_label"])
    splits   = tokenized.train_test_split(test_size=test_size, seed=42)

    print(f"Train: {len(splits['train'])} examples")
    print(f"Test:  {len(splits['test'])} examples")
    return splits
```

## label_tool.py — simple CLI labeling helper

This lets the user label raw text examples from the terminal, building up training_data.csv
one example at a time without needing a full annotation platform.

```python
# label_tool.py
import csv
import os

OUTPUT_FILE = "data/training_data.csv"
SENTIMENT_OPTIONS = ["positive", "neutral", "negative"]
IMPACT_OPTIONS    = ["bullish",  "neutral", "bearish"]


def label_examples():
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.exists(OUTPUT_FILE)

    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["text", "sentiment_label", "impact_label"])

        print("BFSI Sentiment Labeling Tool")
        print("Type the text to label, then choose labels.")
        print("Press Ctrl+C to stop.\n")

        count = 0
        while True:
            try:
                text = input("Text: ").strip()
                if not text:
                    continue

                print(f"Sentiment options: {SENTIMENT_OPTIONS}")
                while True:
                    s = input("Sentiment (p/n/ne): ").strip().lower()
                    sentiment_map = {"p": "positive", "n": "neutral", "ne": "negative",
                                     "positive": "positive", "neutral": "neutral",
                                     "negative": "negative"}
                    if s in sentiment_map:
                        sentiment = sentiment_map[s]
                        break
                    print("Invalid. Use p (positive), n (neutral), ne (negative)")

                print(f"Impact options: {IMPACT_OPTIONS}")
                while True:
                    i = input("Impact (b/n/be): ").strip().lower()
                    impact_map = {"b": "bullish", "n": "neutral", "be": "bearish",
                                  "bullish": "bullish", "neutral": "neutral",
                                  "bearish": "bearish"}
                    if i in impact_map:
                        impact = impact_map[i]
                        break
                    print("Invalid. Use b (bullish), n (neutral), be (bearish)")

                writer.writerow([text, sentiment, impact])
                f.flush()
                count += 1
                print(f"Saved ({count} total). Next example:\n")

            except KeyboardInterrupt:
                print(f"\nDone. {count} examples saved to {OUTPUT_FILE}")
                break


if __name__ == "__main__":
    label_examples()
```
