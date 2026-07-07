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

    # Encode labels to ints before building the HF Dataset. Doing this inside
    # a batched .map() that also removes the same-named string columns causes
    # datasets to unify the output schema back to string dtype (silently
    # stringifying the int labels), so we avoid that column-name collision here.
    df = df.copy()
    df["sentiment_label"] = df["sentiment_label"].map(SENTIMENT_LABELS)
    df["impact_label"]    = df["impact_label"].map(IMPACT_LABELS)

    def tokenize_and_encode(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    dataset  = Dataset.from_pandas(df)
    tokenized = dataset.map(tokenize_and_encode, batched=True,
                            remove_columns=["text"])
    splits   = tokenized.train_test_split(test_size=test_size, seed=42)

    print(f"Train: {len(splits['train'])} examples")
    print(f"Test:  {len(splits['test'])} examples")
    return splits
