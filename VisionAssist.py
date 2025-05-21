import streamlit as st
import google.generativeai as genai
import numpy as np
from PIL import Image, ImageDraw
import imagehash
import difflib
import threading
import pytesseract
import pyttsx3
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from dotenv import load_dotenv
import cv2
import os
import uuid
import time
from io import BytesIO

load_dotenv()

with open(r"C:\Users\shind\Python_Coding\API\keys\API_key.txt") as f:
    api_key = f.read()

genai.configure(api_key=api_key)

st.set_page_config(page_title="Vision Assistance", layout="centered", page_icon="🌟")

@st.cache_resource
def initialize_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

detection_model = initialize_model()

def convert_image_to_bytes(file):
    try:
        image_bytes = file.getvalue()
        return [{"mime_type": file.type, "data": image_bytes}]
    except Exception as e:
        raise ValueError(f"Failed to process image: {e}")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text(image_file):
    try:
        image = Image.open(image_file)
        return pytesseract.image_to_string(image).strip() or "No text detected in the image."
    except Exception as e:
        raise RuntimeError(f"Error extracting text: {e}")

def speak_text(text):
    try:
        if "audio_engine" not in st.session_state:
            st.session_state.audio_engine = pyttsx3.init()
        st.session_state.audio_engine.say(text)
        st.session_state.audio_engine.runAndWait()
    except Exception as e:
        raise RuntimeError(f"Text-to-speech conversion failed: {e}")
    
def perform_object_detection(image, threshold=0.5, nms_threshold=0.5):
    try:
        img_np = np.array(image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        st.image(equalized, caption="Histogram Equalized Image", channels="GRAY")

        edges = cv2.Canny(equalized, 100, 200)
        st.image(edges, caption="Canny Edge Detection", channels="GRAY")

        preprocess = transforms.Compose([transforms.ToTensor()])
        image_tensor = preprocess(image)
        outputs = detection_model([image_tensor])[0]
        indices = torch.ops.torchvision.nms(outputs['boxes'], outputs['scores'], nms_threshold)
        return {k: v[indices] for k, v in outputs.items() if k in ['boxes', 'labels', 'scores']}
    except Exception as e:
        raise RuntimeError(f"Object detection failed: {e}")
    
def highlight_objects(image, detections, threshold=0.5):
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
        if score > threshold:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            class_name = COCO_CLASSES[label.item()]
            draw.text((x1, y1), f"{class_name} ({score:.2f})", fill="yellow")
    return image

COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog", 
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "road"
]

SAVE_DIR = "captured_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

def describe_scene(image_input):
    try:
        if isinstance(image_input, str):
            img = Image.open(image_input)
        else:
            img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))

        buf = BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        prompt = (
            "Describe the image in one sentence."
        )

        gemini_image = {"mime_type": "image/png", "data": img_bytes}
        response = genai.GenerativeModel("gemini-1.5-flash-latest").generate_content([prompt, gemini_image]).text
        return response.strip()
    except Exception as e:
        return f"Error describing scene: {e}"

def filter_similar_descriptions(descriptions, similarity_threshold=0.8):
    filtered = []
    for desc in descriptions:
        if not any(difflib.SequenceMatcher(None, desc, existing).ratio() > similarity_threshold for existing in filtered):
            filtered.append(desc)
    return filtered

def save_frame(frame, frame_count, output_dir="frames"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"frame_{frame_count:03}.jpg")
    cv2.imwrite(path, frame)
    return path

def process_video():
    """Capture video from phone's back camera and return at least 60 unique frames using FPS."""
    PHONE_STREAM_URL = "http://192.168.189.63:8080/video" 
    TARGET_FRAMES = 60
    FPS = 10  
    MAX_DURATION = TARGET_FRAMES / FPS + 5  

    st.info("Connecting to phone camera...")
    cap = cv2.VideoCapture(PHONE_STREAM_URL)  

    if not cap.isOpened():
        st.error("Could not access phone camera stream.")
        return []

    st.success("Connected to phone camera. Processing live video feed...")

    previous_hashes = []
    frame_count = 0
    captured_frames = []
    start_time = time.time()

    while time.time() - start_time < MAX_DURATION and frame_count < TARGET_FRAMES:
        ret, frame = cap.read()
        if not ret:
            continue

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        curr_hash = imagehash.phash(pil_img)

        if not any(abs(curr_hash - prev_hash) < 25 for prev_hash in previous_hashes):
            previous_hashes.append(curr_hash)
            captured_frames.append(frame)
            save_frame(frame, frame_count)
            frame_count += 1

        time.sleep(1 / FPS)

    cap.release()
    st.success(f"Captured and saved {frame_count} unique frames.")
    return captured_frames

st.title("Real Time Video Processing")

if st.button("Start Video Capture"):
    unique_frames = process_video()
    st.success(f"{len(unique_frames)} unique frames captured. Starting scene analysis...")

    all_descriptions = []

    st.markdown("### Captured Frames and Descriptions")

    for i in range(0, len(unique_frames), 5):
        cols = st.columns(5)
        for j in range(5):
            if i + j < len(unique_frames):
                frame = unique_frames[i + j]
                scene_description = describe_scene(frame)
                all_descriptions.append(scene_description)
                with cols[j]:
                    st.image(frame, channels="BGR", caption=f"Frame {i + j + 1}")
                    # st.write(f"Frame {i + j + 1} Description: {scene_description}")

    unique_descriptions = filter_similar_descriptions(all_descriptions, similarity_threshold=0.35)
    full_text = " ".join(unique_descriptions)
    st.write(full_text)

    try:
        st.info("Speaking all instructions together...")
        speak_text(full_text)
    except Exception as e:
        st.warning(f"Failed to speak combined description: {e}")


st.sidebar.title("Actions to Perform")
text_button = st.sidebar.button("Extract Text 📜")
scene_button = st.sidebar.button("Describe Image 🖋️")
detect_button = st.sidebar.button("Detect Objects 🚦")
assist_button = st.sidebar.button("Personal Assist")
audio_button = st.sidebar.button("Stop Audio 🔇")

st.title("Image Assistance")
uploaded_images = st.file_uploader("Upload multiple images:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

for uploaded_image in uploaded_images:
    st.image(uploaded_image, caption=f"Uploaded Image: {uploaded_image.name}", use_column_width=True)

    if text_button:
        with st.spinner("Extracting text..."):
            extracted_text = extract_text(uploaded_image)
            st.subheader("Extracted Text (OCR):")
            st.write(extracted_text)

    if scene_button:
        with st.spinner("Analyzing scene..."):
            prompt = (
                "Describe image in short (max = 20 words) and precisely."
            )
            image_bytes = convert_image_to_bytes(uploaded_image)
            response = genai.GenerativeModel("gemini-1.5-flash-latest").generate_content([prompt, image_bytes[0]]).text
            st.subheader("Scene Description:")
            st.write(response)
            speak_text(response)

    if detect_button:
        with st.spinner("Detecting objects..."):
                img = Image.open(uploaded_image)
                detections = perform_object_detection(img)
                annotated_image = highlight_objects(img.copy(), detections)
                st.image(annotated_image, caption="Objects Detected", use_column_width=True)

    if assist_button:
        with st.spinner("Providing assistance..."):
            assistance_prompt = (
                "Imagine your giving instructions to a visually impaired person. Describe the scene "
                "(max = 20 words) in a way that they can easily understand and follow."
            )
            image_bytes = convert_image_to_bytes(uploaded_image)
            assistance_response = genai.GenerativeModel("gemini-1.5-flash-latest").generate_content([assistance_prompt, image_bytes[0]]).text
            st.subheader("Assistance:")
            st.write(assistance_response)
            speak_text(assistance_response)


if audio_button:
    try:
        if "tts_engine" not in st.session_state:
            st.session_state.tts_engine = pyttsx3.init()

        st.session_state.tts_engine.stop()
        st.success("Audio playback stopped.")
    except Exception as e:
        st.error(f"Failed to stop the audio. Error: {e}")