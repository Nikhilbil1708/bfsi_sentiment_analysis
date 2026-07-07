# CLAUDE.md — BFSI Sentiment SLM Fine-tuning Project

Read this file fully before writing any code. Then read every file listed
in "Read these files next" before starting any implementation.

## What this project does

Fine-tunes a small pre-trained financial language model (FinBERT, 110M parameters)
using LoRA adapters to classify text from US BFSI industry sources into:

- Sentiment: positive / neutral / negative
- Market impact: bullish / neutral / bearish

Both predictions are produced in a single inference call from one multi-head model.
All training and inference runs on the user's local CPU laptop — no GPU assumed.
Model weights and training data are stored on Google Drive, not locally.

## Hardware constraints — read this before choosing any approach

- No dedicated NVIDIA GPU — CPU only
- This means QLoRA 4-bit quantization is NOT available (requires CUDA)
- Use plain LoRA with a small base model instead
- Base model: ProsusAI/finbert (110M parameters — correct for CPU training)
- Do NOT suggest Llama, Mistral, GPT-2 medium, or any model above 500M parameters
- Batch size must stay at 8 or lower during training

## Storage — Google Drive, not local

Model weights and training data live on Google Drive. The user has 15GB free quota
which is more than enough (total project files are ~30MB). All Drive access goes
through the `drive_storage.py` helper. Do not store model weights only locally.

## Tech stack

- Python 3.x
- transformers (HuggingFace) — model loading and tokenization
- peft — LoRA adapter implementation
- torch — training loop
- datasets — dataset loading from HuggingFace hub
- google-api-python-client — Google Drive API access
- scikit-learn — evaluation metrics
- pandas — dataset preparation
- evaluate — HuggingFace evaluation library

## Project structure to create

```
bfsi-sentiment-slm/
├── data/
│   └── training_data.csv       ← downloaded from Drive before training
├── output/
│   ├── lora_adapters/          ← saved locally after training, uploaded to Drive
│   ├── sentiment_head.pt
│   └── impact_head.pt
├── drive_storage.py            ← Google Drive upload/download helper
├── dataset_prep.py             ← tokenization and dataset preparation
├── model.py                    ← FinBERT + LoRA + dual classification heads
├── train.py                    ← training loop with Drive upload at the end
├── evaluate.py                 ← evaluation metrics and confusion matrix
├── predict.py                  ← inference with Drive download if needed
├── setup_drive.py              ← one-time Drive authentication and folder setup
├── label_tool.py               ← simple CLI tool for labeling raw text examples
├── credentials.json            ← OAuth credentials (gitignored, user provides)
├── token.json                  ← auto-generated after first auth (gitignored)
├── requirements.txt
└── .gitignore
```

## Read these files next before writing any code

- `model_architecture.md` — FinBERT + LoRA config + dual head design
- `data_pipeline.md` — dataset sources, labeling schema, CSV format
- `training_config.md` — hyperparameters, training loop, CPU-specific settings
- `drive_integration.md` — Google Drive API setup and helper implementation
- `evaluation.md` — metrics, confusion matrix, what good performance looks like
- `inference.md` — predict.py design and how to use the trained model

## Build order — follow this exactly

1. `.gitignore` and `requirements.txt`
2. `drive_storage.py`
3. `setup_drive.py` (one-time auth script)
4. `dataset_prep.py`
5. `label_tool.py`
6. `model.py`
7. `train.py`
8. `evaluate.py`
9. `predict.py`

## Non-negotiable rules

Never suggest GPU-only approaches (QLoRA 4-bit, device_map="cuda", etc.)
without explicitly flagging that this requires hardware the user does not have.

Never store credentials in any file other than credentials.json and token.json,
and both must be in .gitignore.

All Drive file operations must go through drive_storage.py — no direct Drive
API calls scattered across other files.

Training data CSV must be validated for correct label values before training
starts — fail loudly with a clear error message if labels are wrong, rather
than silently training on corrupt data.
