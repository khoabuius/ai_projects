import os
import gradio as gr
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline

model_path = "../../models/detr-resnet-50"
detector = pipeline("object-detection", model=model_path)

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
    return processed_image

# Launch Gradio UI
gr.close_all()
main = gr.Interface(
    fn=detect_object,
    inputs=[gr.Image(label="Select Image")],
    outputs=[gr.Image(label="Processed Image")],
    title="@GenAI Project 6: Object Detection",
    description="Upload an Image. The app will will detect image in the provided image input."
)
main.launch()