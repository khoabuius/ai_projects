# Models

These model files are too large to store in GitHub (total ~3.4 GB, GitHub limit is 100 MB per file).

Download them from Hugging Face Hub:

- **nllb-200-distilled-600M** (2.3 GB) — https://huggingface.co/facebook/nllb-200-distilled-600M
- **distilbert-base-uncased-finetuned-sst-2-english** (255 MB) — https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
- **distillBart-6-6** (877 MB) — https://huggingface.co/NuppuCat/distillBart-6-6
- **roberta-base-squad2** (473 MB) — https://huggingface.co/deepset/roberta-base-squad2
  - Includes LoRA adapter and tokenizer
  - Downloaded via Kaggle: `kaggle datasets download -d umarzeinardhisiregar/deepset-roberta-base-squad2`

The Python scripts in this project will automatically download these models on first run.
