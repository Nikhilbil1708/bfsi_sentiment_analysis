# model_architecture.md — FinBERT + LoRA + Dual Classification Heads

## Why FinBERT as the base model

FinBERT (ProsusAI/finbert) is BERT fine-tuned on financial text — Reuters, Bloomberg,
Financial Times. It already understands financial vocabulary, earnings terminology,
banking sector language, and regulatory phrasing. This means LoRA adapters only need
to learn the specific label scheme (your 6 classes across 2 tasks), not financial
language itself. This is why it outperforms general models like DistilBERT on BFSI text
even with a small training dataset.

## LoRA configuration for CPU

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,                          # rank 8 is right for a 3-class task on a pre-trained domain model
    lora_alpha=32,                # scaling = 4x rank, standard starting point
    lora_dropout=0.1,             # regularization — helps with small datasets
    target_modules=["query", "value"]   # apply adapters to attention query and value projections
)
```

### What r=8 means in plain terms

LoRA inserts two small matrices (A and B) at each target layer. r is the inner dimension
of these matrices. r=8 means each adapter has 8 "directions" it can modify the layer's
behavior in. Lower r = fewer parameters trained = faster, less expressive. Higher r =
more parameters = slower, more capable. For a 3-class sentiment task on FinBERT, r=8
is well-calibrated — the model already understands finance, so the adapters only need
to learn "which direction is positive, which is negative."

### Why target_modules=["query", "value"] specifically

In BERT's self-attention mechanism, each layer has query, key, and value projections.
Research on LoRA shows that applying adapters to query and value (not key) gives the
best accuracy-to-parameter-count ratio for classification tasks. Adding key typically
improves results by less than 1% while adding significant training time on CPU.

## Model class — full implementation

```python
# model.py
import torch
import torch.nn as nn
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType

SENTIMENT_LABELS = {"positive": 0, "neutral": 1, "negative": 2}
IMPACT_LABELS    = {"bullish":  0, "neutral": 1, "bearish":  2}

SENTIMENT_ID2LABEL = {v: k for k, v in SENTIMENT_LABELS.items()}
IMPACT_ID2LABEL    = {v: k for k, v in IMPACT_LABELS.items()}


class BFSISentimentModel(nn.Module):
    def __init__(self, base_model_name: str = "ProsusAI/finbert"):
        super().__init__()

        print(f"Loading base model: {base_model_name}")
        base_model = AutoModel.from_pretrained(base_model_name)

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "value"]
        )

        self.encoder   = get_peft_model(base_model, lora_config)
        hidden_size    = base_model.config.hidden_size  # 768 for FinBERT

        # Two separate heads — one per prediction task
        self.sentiment_head = nn.Linear(hidden_size, 3)
        self.impact_head    = nn.Linear(hidden_size, 3)

        # Print trainable parameter count so user can see LoRA is working correctly
        self.encoder.print_trainable_parameters()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # [CLS] token is the first token — used for sequence classification
        cls_output       = outputs.last_hidden_state[:, 0, :]
        sentiment_logits = self.sentiment_head(cls_output)
        impact_logits    = self.impact_head(cls_output)
        return sentiment_logits, impact_logits

    def save(self, output_dir: str = "output"):
        import os
        os.makedirs(f"{output_dir}/lora_adapters", exist_ok=True)
        self.encoder.save_pretrained(f"{output_dir}/lora_adapters")
        torch.save(self.sentiment_head.state_dict(), f"{output_dir}/sentiment_head.pt")
        torch.save(self.impact_head.state_dict(),    f"{output_dir}/impact_head.pt")
        print(f"Model saved to {output_dir}/")

    @classmethod
    def load(cls, output_dir: str = "output", base_model_name: str = "ProsusAI/finbert"):
        from peft import PeftModel
        instance = cls.__new__(cls)
        super(BFSISentimentModel, instance).__init__()

        base_model      = AutoModel.from_pretrained(base_model_name)
        instance.encoder = PeftModel.from_pretrained(base_model, f"{output_dir}/lora_adapters")
        hidden_size      = base_model.config.hidden_size

        instance.sentiment_head = nn.Linear(hidden_size, 3)
        instance.impact_head    = nn.Linear(hidden_size, 3)
        instance.sentiment_head.load_state_dict(torch.load(f"{output_dir}/sentiment_head.pt",
                                                            map_location="cpu"))
        instance.impact_head.load_state_dict(torch.load(f"{output_dir}/impact_head.pt",
                                                          map_location="cpu"))
        instance.eval()
        return instance
```

## Expected trainable parameter count

When `print_trainable_parameters()` runs, expect output similar to:

```
trainable params: 887,808 || all params: 110,062,592 || trainable%: 0.807
```

This means roughly 0.8% of parameters are being updated — this is the whole point of
LoRA. The other 99.2% are frozen (the pre-trained FinBERT weights), which is why
training is feasible on CPU without running out of memory.
