# training_config.md — Hyperparameters and Training Loop

## CPU-specific settings — rationale for every choice

```python
TRAINING_CONFIG = {
    "epochs":         3,       # 3 passes over the data — enough for fine-tuning a pre-trained model
    "batch_size":     8,       # keep low on CPU — larger batches don't speed up CPU training
    "learning_rate":  2e-5,    # standard for fine-tuning BERT-family models
    "weight_decay":   0.01,    # mild regularization — helps with small datasets
    "warmup_steps":   50,      # linear LR warmup to stabilize early training
    "max_grad_norm":  1.0,     # gradient clipping — prevents instability during LoRA training
    "loss_weights":   {        # weight sentiment slightly higher — it's the primary task
        "sentiment": 0.6,
        "impact":    0.4
    }
}
```

## train.py — full implementation

```python
# train.py
import torch
import torch.nn as nn
import os
import time
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from dotenv import load_dotenv
load_dotenv()

from model import BFSISentimentModel
from dataset_prep import prepare_dataset
from drive_storage import download_file, upload_model_to_drive

TRAINING_CONFIG = {
    "epochs":        3,
    "batch_size":    8,
    "learning_rate": 2e-5,
    "weight_decay":  0.01,
    "warmup_steps":  50,
    "max_grad_norm": 1.0,
    "sentiment_loss_weight": 0.6,
    "impact_loss_weight":    0.4,
}


def collate_fn(batch):
    return {
        "input_ids":       torch.tensor([x["input_ids"]       for x in batch]),
        "attention_mask":  torch.tensor([x["attention_mask"]  for x in batch]),
        "sentiment_label": torch.tensor([x["sentiment_label"] for x in batch]),
        "impact_label":    torch.tensor([x["impact_label"]    for x in batch]),
    }


def train_epoch(model, loader, optimizer, scheduler, criterion, config):
    model.train()
    total_loss  = 0
    total_steps = 0

    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()

        sentiment_logits, impact_logits = model(
            input_ids      = batch["input_ids"],
            attention_mask = batch["attention_mask"]
        )

        sentiment_loss = criterion(sentiment_logits, batch["sentiment_label"])
        impact_loss    = criterion(impact_logits,    batch["impact_label"])
        loss = (
            config["sentiment_loss_weight"] * sentiment_loss +
            config["impact_loss_weight"]    * impact_loss
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        optimizer.step()
        scheduler.step()

        total_loss  += loss.item()
        total_steps += 1

        if batch_idx % 5 == 0:
            elapsed = time.strftime("%H:%M:%S", time.gmtime())
            print(f"  [{elapsed}] Batch {batch_idx}/{len(loader)} — "
                  f"loss: {loss.item():.4f} "
                  f"(sent: {sentiment_loss.item():.4f}, "
                  f"impact: {impact_loss.item():.4f})")

    return total_loss / total_steps


def evaluate_epoch(model, loader):
    model.eval()
    sentiment_correct = impact_correct = total = 0

    with torch.no_grad():
        for batch in loader:
            s_logits, i_logits = model(
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"]
            )
            sentiment_correct += (s_logits.argmax(1) == batch["sentiment_label"]).sum().item()
            impact_correct    += (i_logits.argmax(1) == batch["impact_label"]).sum().item()
            total             += len(batch["sentiment_label"])

    return sentiment_correct / total, impact_correct / total


def train(csv_path: str = "data/training_data.csv"):
    print("=" * 60)
    print("BFSI Sentiment SLM — Training")
    print("=" * 60)

    # Step 1: Load dataset
    print("\nStep 1: Loading and preparing dataset...")
    splits       = prepare_dataset(csv_path)
    train_loader = DataLoader(splits["train"], batch_size=TRAINING_CONFIG["batch_size"],
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(splits["test"],  batch_size=TRAINING_CONFIG["batch_size"],
                              shuffle=False, collate_fn=collate_fn)

    # Step 2: Initialize model
    print("\nStep 2: Initializing model...")
    model = BFSISentimentModel()

    # Step 3: Setup optimizer, scheduler, loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"]
    )
    total_steps = len(train_loader) * TRAINING_CONFIG["epochs"]
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=TRAINING_CONFIG["warmup_steps"],
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    # Step 4: Training loop
    print(f"\nStep 3: Training for {TRAINING_CONFIG['epochs']} epochs...")
    print(f"  Batches per epoch: {len(train_loader)}")
    print(f"  Estimated time on CPU: {len(train_loader) * 4 * TRAINING_CONFIG['epochs'] // 60} min\n")

    best_sentiment_acc = 0.0

    for epoch in range(TRAINING_CONFIG["epochs"]):
        print(f"\nEpoch {epoch + 1}/{TRAINING_CONFIG['epochs']}")
        print("-" * 40)

        avg_loss       = train_epoch(model, train_loader, optimizer, scheduler,
                                     criterion, TRAINING_CONFIG)
        sentiment_acc, impact_acc = evaluate_epoch(model, val_loader)

        print(f"\n  Epoch {epoch+1} summary:")
        print(f"    Avg loss:          {avg_loss:.4f}")
        print(f"    Sentiment accuracy: {sentiment_acc:.3f} ({sentiment_acc*100:.1f}%)")
        print(f"    Impact accuracy:    {impact_acc:.3f} ({impact_acc*100:.1f}%)")

        if sentiment_acc > best_sentiment_acc:
            best_sentiment_acc = sentiment_acc
            model.save("output")
            print(f"    New best model saved (sentiment acc: {best_sentiment_acc:.3f})")

    # Step 5: Upload to Drive
    print("\nStep 4: Uploading trained model to Google Drive...")
    upload_model_to_drive()
    print("\nTraining complete.")
    print(f"Best sentiment accuracy: {best_sentiment_acc:.3f}")


if __name__ == "__main__":
    # Download training data from Drive if not already local
    if not os.path.exists("data/training_data.csv"):
        print("Downloading training data from Drive...")
        os.makedirs("data", exist_ok=True)
        download_file("training_data.csv", "data/training_data.csv")

    train("data/training_data.csv")
```

## Expected training time on CPU

| Dataset size | Epochs | Estimated time |
|---|---|---|
| 100 examples | 3 | ~10-15 minutes |
| 300 examples | 3 | ~30-45 minutes |
| 500 examples | 3 | ~60-90 minutes |

These are estimates for a modern laptop CPU (Intel i5/i7 or AMD Ryzen equivalent).
Actual time varies by CPU generation and background load.
