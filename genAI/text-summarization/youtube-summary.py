import gradio as gr
import re
from youtube_transcript_api import YouTubeTranscriptApi
print(dir(YouTubeTranscriptApi))
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_path = '../../models/models--NuppuCat--distillBart-6-6/snapshots/0988db501efe4d2e176dc850a49555c8c1a5474f'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def print_summary(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=30, num_beams=4)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def extract_video_id(url):
    """
    Extracts the YouTube video ID from a URL.
    """
    # Regex pattern to match various YouTube URL formats
    pattern = (
        r'(?:v=|\/|be\/|embed\/)([a-zA-Z0-9_-]{11})'
    )
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Could not extract video ID from URL.")

def get_youtube_transcript_from_url(url):
    try:
        video_id = extract_video_id(url)
        api = YouTubeTranscriptApi()              # instantiate
        transcript_list = api.list(video_id)
        transcript = transcript_list.find_transcript(['en'])
        transcript_data = transcript.fetch()
        transcriptRecord = ' '.join([entry.text for entry in transcript_data])  # .text not ['text']
        return print_summary(transcriptRecord)
    except Exception as e:
        return f"Error: {e}"

# Example usage:
# url = 'https://www.youtube.com/watch?v=l00VBUXl1Q4&list=PLhr0Ua8H1x-K7UMXXeSfjULEIBCE1FVd1&index=2'
# print(get_youtube_transcript_from_url(url))

gr.close_all()
# main = gr.Interface(fn=print_summary, inputs=["text"], outputs=["text"])
main = gr.Interface(fn=get_youtube_transcript_from_url, inputs=[gr.Textbox(label="Input youtube url for summary", lines=6)], outputs=[gr.Textbox(label="Summarized text", lines=6)],
                    title="@GenAI Project 2: Youtube Script Summarization",
                    description="THIS APPLICATION WILL SUMMARIZE THE YOUTUBE SCRIPT")
main.launch()