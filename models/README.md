# Models

These model files are too large to store in GitHub (total ~3.4 GB, GitHub limit is 100 MB per file).

Download them from Hugging Face Hub:

- **detr-resnet-50** (object detection) — https://huggingface.co/facebook/detr-resnet-50
  - Used by: `genAI/image-processing/image-detector.py`
  - Expected local path: `models/detr-resnet-50`
- **nllb-200-distilled-600M** (2.3 GB) — https://huggingface.co/facebook/nllb-200-distilled-600M
- **distilbert-base-uncased-finetuned-sst-2-english** (255 MB) — https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
- **distillBart-6-6** (877 MB) — https://huggingface.co/NuppuCat/distillBart-6-6
- **roberta-base-squad2** (473 MB) — https://huggingface.co/deepset/roberta-base-squad2
  - Includes LoRA adapter and tokenizer
  - Downloaded via Kaggle: `kaggle datasets download -d umarzeinardhisiregar/deepset-roberta-base-squad2`

Kaggle assets downloaded locally:

- **VITS text-to-speech notebook outputs** — https://www.kaggle.com/code/youssef19/building-text-to-speech-systems-using-vits-artst
  - Local path: `models/vits-artst`
  - Files downloaded:
    - `building-text-to-speech-systems-using-vits-artst.ipynb`
    - `building-text-to-speech-systems-using-vits-artst.log`
    - `narrated_text.mp3`
    - `speech.wav`
  - Note: this Kaggle page provides notebook source and generated audio outputs. It does not include full VITS model weight files.

- **VITS checkpoint bundle (Kaggle model)** — `kelvinnkt/vits-tts`
  - Local path: `models/vits-ljs`
  - Download commands used:
    - `kaggle models instances versions download kelvinnkt/vits-tts/pytorch/default/1 -p models/vits-ljs --untar`
    - `kaggle models instances versions download kelvinnkt/vits-tts/pytorch/conf/1 -p models/vits-ljs --untar`
  - Downloaded files:
    - `model_file.pth`
    - `config.json`
  - Compatibility note:
    - This bundle does not include the full Hugging Face model package files in this environment, so `pipeline("text-to-speech", model="../../models/vits-ljs")` may still require additional tokenizer/model-export files.

The Python scripts in this project will automatically download these models on first run.
