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
