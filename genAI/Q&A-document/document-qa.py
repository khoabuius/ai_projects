
import os

import torch
import gradio as gr
import pandas as pd
from joblib.externals.loky.backend import context
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import matplotlib.pyplot as plt

# Resolve model path relative to this script's location (works from any working directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(SCRIPT_DIR, "../../models/roberta-base-squad2/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(SCRIPT_DIR, "../../models/roberta-base-squad2/tokenizer-deepset-roberta-base-squad2")
)


def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits)
    print(f'outputs: {outputs}, start_idx: {start_idx}, end_idx: {end_idx}')
    score = (outputs.start_logits[0, start_idx] + outputs.end_logits[0, end_idx]).item()
    answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return {"answer": answer, "score": round(score, 4), "start": start_idx.item(), "end": end_idx.item()}

question = "What is Mark"
context = "Mark Elliot Zuckerberg (/ˈzʌkərbɜːrɡ/; born May 14, 1984) is an American businessman and programmer who co-founded the social media service Facebook and its parent company Meta Platforms. He is its chairman, chief executive officer (CEO), and controlling shareholder.Zuckerberg briefly attended Harvard College, and launched Facebook there in February 2004 with his roommates Eduardo Saverin, Andrew McCollum, Dustin Moskovitz and Chris Hughes. Zuckerberg took the company public in May 2012 with majority shares. He became the world's youngest self-made billionaire[a] in 2008 at age 23 and has consistently ranked among the world's wealthiest people. According to Forbes, Zuckerberg's estimated net worth stood at US$220 billion as of December 2025"
answer = answer_question(question, context)
print(answer)

def _get_answer(file, question):
    with open(file, "r") as f:
        content = f.read()
    answer = answer_question(question, content)
    return "No answer" if not answer["answer"] else answer["answer"]

# Launch Gradio UI
gr.close_all()
main = gr.Interface(
    fn=_get_answer,
    inputs=[gr.File(label="Upload your document", file_types=[".txt"]), gr.Textbox(label="Input your questions:", lines=3)],
    outputs=[gr.Textbox(label="Answers question", lines=20)],
    title="@GenAI Project 4: Answer the question based on context",
    description="Upload a text file with content. The app will will answer the question based on context."
)
main.launch()