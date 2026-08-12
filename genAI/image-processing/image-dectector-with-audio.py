import os
import gradio as gr
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
from gtts import gTTS

model_path = "../../models/detr-resnet-50"
detector = pipeline("object-detection", model=model_path)

TTS_OUTPUT = "file_tune.mp3"

vet_image = Image.open("vegetables.png")
results = detector(vet_image, threshold=0.7)

print(results)


def highlight_boundaries(image: Image.Image, detections: list) -> Image.Image:
    """
    Highlights boundaries on the input PIL image based on detection results.

    Parameters:
    - image: PIL.Image object
    - detections: list of dicts or tuples with bounding box coordinates, e.g.,
                  [{'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2}, ...]
                  or [(x1, y1, x2, y2), ...]

    Returns:
    - PIL.Image object with highlighted boundaries
    """
    # Make a copy of the image to draw on
    image_with_boundaries = image.copy()
    draw = ImageDraw.Draw(image_with_boundaries)

    # Define line color and width
    outline_color = 'red'
    line_width = 3
    font = ImageFont.load_default()

    for detection in detections:
        if isinstance(detection, dict):
            box = detection.get('box', detection)
            xmin = box['xmin']
            ymin = box['ymin']
            xmax = box['xmax']
            ymax = box['ymax']
            label = detection.get('label', '')
            score = detection.get('score', 0)
        elif isinstance(detection, (list, tuple)):
            xmin, ymin, xmax, ymax = detection
            label, score = '', 0
        else:
            continue

        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            outline=outline_color,
            width=line_width
        )

        if label:
            text = f"{label} {score:.0%}"
            draw.text((xmin, ymin - 12), text, fill=outline_color, font=font)

    return image_with_boundaries

def detect_object(image):
    my_img = Image.fromarray(image)
    detect_results = detector(my_img, threshold=0.7)
    processed_image = highlight_boundaries(my_img, detect_results)
    summary_text = summarize_detections(detect_results)
    try:
        audio_file = generate_audio(summary_text)
    except Exception as e:
        print(f"Warning: audio generation failed: {e}")
        audio_file = None
    return processed_image, audio_file

def generate_audio(text):
    """Generate speech from text using gTTS (Google Text-to-Speech)."""
    tts = gTTS(text)
    tts.save(TTS_OUTPUT)
    return TTS_OUTPUT

def summarize_detections(detections):
    label_counts = {}
    for obj in detections:
        label = obj['label']
        label_counts[label] = label_counts.get(label, 0) + 1

    # Create a summary string with pluralization
    summary_parts = []
    for label, count in label_counts.items():
        if count > 1:
            label_str = f"{label}s"  # plural form
        else:
            label_str = label
        summary_parts.append(f"{count} {label_str} ")
    summary = ", ".join(summary_parts)
    return summary


    # Create a summary string
    summary_parts = [f"{count} {label} " for label, count in label_counts.items()]
    summary = ", ".join(summary_parts)
    return summary


# Launch Gradio UI
gr.close_all()
main = gr.Interface(
    fn=detect_object,
    inputs=[gr.Image(label="Select Image")],
    outputs=[gr.Image(label="Processed Image with Audio Recognition"), gr.Audio(label="Audio Recognition")],
    title="@GenAI Project 7: Object Detection with Audio Recognition",
    description="Upload an Image. The app will will detect image in the provided image input and audio recognition."
)
main.launch()