import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_path = '../../models/models--NuppuCat--distillBart-6-6/snapshots/0988db501efe4d2e176dc850a49555c8c1a5474f'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

text = '''No, there is no way to increase the maximum item size limit (400 KB) for DynamoDB. This limit is a hard constraint set by DynamoDB's design and cannot be changed.'''

def print_summary(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=30, num_beams=4)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

gr.close_all()
# main = gr.Interface(fn=print_summary, inputs=["text"], outputs=["text"])
main = gr.Interface(fn=print_summary, inputs=[gr.Textbox(label="Input text for summary", lines=6)], outputs=[gr.Textbox(label="Summarized text", lines=6)],
                    title="@GenAI Project 1: Text Summarization",
                    description="THIS APPLICATION WILL SUMMARIZE THE INPUT TEXT")
main.launch()