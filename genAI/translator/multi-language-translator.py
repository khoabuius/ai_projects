import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
import gradio as gr

model_path = "../../models/nllb-200-distilled-600M"

# Load the JSON data from the file
with open('../../Files/languages.json', 'r', encoding='utf-8') as file:
    language_list = json.load(file)

# Extract language names for dropdown
language_names = [item['Language'] for item in language_list]

# Load model once at startup (not per request)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

def get_language_code(language_name):
    """Given a language name, return the corresponding NLLB code."""
    language_name_lower = language_name.strip().lower()
    for item in language_list:
        if item['Language'].lower() == language_name_lower:
            return item['code']
    return None

def translate_language(text, source_lang, target_lang):
    source_lang_code = get_language_code(source_lang)
    target_lang_code = get_language_code(target_lang)

    tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang=source_lang_code)
    inputs = tokenizer(text, return_tensors="pt")

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang_code),
        max_new_tokens=inputs["input_ids"].shape[1]
    )
    result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return result

gr.close_all()
main = gr.Interface(
    fn=translate_language,
    inputs=[
        gr.Textbox(label="Input text to translate", lines=6),
        gr.Dropdown(language_names, label="Select source language", value="English"),
        gr.Dropdown(language_names, label="Select target language", value="Vietnamese")
    ],
    outputs=[gr.Textbox(label="Translated text", lines=6)],
    title="@GenAI Project 4: Languages Translator",
    description="THIS APPLICATION WILL TRANSLATE LANGUAGE THE INPUT TEXT TO TARGET LANGUAGE"
)
main.launch()