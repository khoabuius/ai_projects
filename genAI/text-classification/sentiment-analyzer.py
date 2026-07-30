import os
import torch
import gradio as gr
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# Resolve model path relative to this script's location (works from any working directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(SCRIPT_DIR, "../../models/distilbert-base-uncased-finetuned-sst-2-english")
analyzer = pipeline("text-classification", model=model_path)


def sentiment_analysis(file):
    # Read the uploaded file
    if file.name.endswith('.csv'):
        df = pd.read_csv(file.name)
    else:
        df = pd.read_excel(file.name)

    # Find the text column (try common names)
    text_col = None
    for col in ['review', 'Sentiments', 'text', 'comment', 'Review', 'Text', 'Comment']:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        text_col = df.columns[0]  # fallback to first column

    # Analyze each row
    results = []
    labels = []
    for text in df[text_col].tolist():
        if isinstance(text, str) and text.strip():
            label = analyzer(text)[0]['label']
            results.append(f"{label:>8}  {text}")
            labels.append(label)
        else:
            results.append(f"{'SKIP':>8}  (empty)")

    # Plot pie chart of sentiment distribution
    label_counts = pd.Series(labels).value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140,
           colors=['#4CAF50', '#F44336'] if 'POSITIVE' in label_counts.index else None)
    ax.set_title('Sentiment Distribution')
    ax.axis('equal')

    return "\n".join(results), fig


# Launch Gradio UI
gr.close_all()
main = gr.Interface(
    fn=sentiment_analysis,
    inputs=[gr.File(label="Upload CSV or Excel file", file_types=[".csv", ".xlsx"])],
    outputs=[gr.Textbox(label="Sentiment Analysis Results", lines=20), gr.Plot(label="Sentiment Review Counts")],
    title="@GenAI Project 3: Sentiment Analyzer",
    description="Upload a CSV or Excel file with a text column. The app will analyze sentiment for each row."
)
main.launch()