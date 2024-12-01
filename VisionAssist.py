import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageDraw
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

# Loading the environment variables
load_dotenv()

# Loading API key
with open(r"C:\Users\shind\Python_Coding\API\keys\API_key.txt") as f:
    api_key = f.read()

genai.configure(api_key=api_key)

# Streamlit App Configuration
st.set_page_config(page_title="AI Vision Assistance", layout="centered", page_icon="🌟")

# Helper Function to Load Object Detection Model
@st.cache_resource
def initialize_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

detection_model = initialize_model()

# Convert Uploaded Image to Bytes
def convert_image_to_bytes(file):
    try:
        image_bytes = file.getvalue()
        return [{"mime_type": file.type, "data": image_bytes}]
    except Exception as e:
        raise ValueError(f"Failed to process image: {e}")

# Extract Text from Uploaded Image
def extract_text(image_file):
    try:
        image = Image.open(image_file)
        return pytesseract.image_to_string(image).strip() or "No text detected in the image."
    except Exception as e:
        raise RuntimeError(f"Error extracting text: {e}")

# Convert Text to Speech
def speak_text(text):
    try:
        if "audio_engine" not in st.session_state:
            st.session_state.audio_engine = pyttsx3.init()
        st.session_state.audio_engine.say(text)
        st.session_state.audio_engine.runAndWait()
    except Exception as e:
        raise RuntimeError(f"Text-to-speech conversion failed: {e}")

# Detect Objects in Image
def perform_object_detection(image, threshold=0.5, nms_threshold=0.5):
    try:
        preprocess = transforms.Compose([transforms.ToTensor()])
        image_tensor = preprocess(image)
        outputs = detection_model([image_tensor])[0]
        indices = torch.ops.torchvision.nms(outputs['boxes'], outputs['scores'], nms_threshold)
        return {k: v[indices] for k, v in outputs.items() if k in ['boxes', 'labels', 'scores']}
    except Exception as e:
        raise RuntimeError(f"Object detection failed: {e}")

# Draw Detected Objects on Image
def highlight_objects(image, detections, threshold=0.5):
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
        if score > threshold:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            class_name = COCO_CLASSES[label.item()]
            draw.text((x1, y1), f"{class_name} ({score:.2f})", fill="yellow")
    return image

# COCO Classes
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

# Directory to save captured frames
SAVE_DIR = "captured_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_frame(frame, frame_count):
    """Save the current frame as an image file."""
    filename = os.path.join(SAVE_DIR, f"frame_{frame_count}_{uuid.uuid4().hex}.png")
    cv2.imwrite(filename, frame)
    return filename

def describe_scene(image_path):
    """Describe the scene in the given image using an AI model."""
    try:
        # Alternatively, using PIL to open the image and convert it to bytes
        img = Image.open(image_path)

        prompt = (
                "Analyze the uploaded image and describe the scene and object in clear and"
                "simple language in very short to assist visually impaired users like crossing roads and detecting objects. Answer must include key"
                "details with highlights about the environment, objects, people, and actions present in the image."
            )
        # Assuming genai is pre-configured with your API key
        response = genai.GenerativeModel("gemini-1.5-pro").generate_content([prompt, img]).text
        return response.strip()
    except Exception as e:
        return f"Error describing scene: {e}"

def process_video():
    """Capture video, save frames, and describe scenes."""
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    frame_count = 0
    start_time = time.time()  # Start timer when capture begins
    pause_time = 15  # Time (in seconds) to capture video before pausing

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Save the current frame
        frame_path = save_frame(frame, frame_count)

        # Describe the scene (always process each frame)
        scene_description = describe_scene(frame_path)
        
        # Display the processed frame and description in Streamlit
        st.image(frame, channels="BGR", caption=f"Frame {frame_count}")
        st.write(f"Frame {frame_count} Description: {scene_description}")
        speak_text(scene_description)

        frame_count += 1
        
        # Break after pausing for the specified time
        if elapsed_time >= pause_time:  # Pause after 15 seconds
            st.write("Video capture paused after 15 seconds.")
            break  # Exit the loop after 15 seconds of capture

    cap.release()
    st.success("Video processing completed. All frames saved.")

# Ensure the detection model is initialized before using
detection_model = initialize_model()

# Streamlit interface
st.title("Real Time Video Assist for Visually Impaired")
if st.button("Start Video Capture"):
    process_video()


# Sidebar with Buttons
st.sidebar.title("Actions to Perform")
text_button = st.sidebar.button("Describe Image 📜")
scene_button = st.sidebar.button("Read Text from Image 🖋️")
detect_button = st.sidebar.button("Detect Objects 🚦")
assist_button = st.sidebar.button("Personal Assist")
audio_button = st.sidebar.button("Stop Audio 🔇")

# Center Layout for Upload Section
st.title("Image Assistance for Visually Impaired")
uploaded_images = st.file_uploader("Upload multiple images:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

for uploaded_image in uploaded_images:
    # Process each image
    st.image(uploaded_image, caption=f"Uploaded Image: {uploaded_image.name}", use_column_width=True)

    # Text Extraction
    if text_button:
        with st.spinner("Describing image..."):
            prompt = (
                "Analyze the uploaded image and describe the scene in clear and"
                "simple language to assist visually impaired users. Include key"
                "details about the environment, objects, people, and actions present in the image."
            )
            image_bytes = convert_image_to_bytes(uploaded_image)
            response = genai.GenerativeModel("gemini-1.5-pro").generate_content([prompt, image_bytes[0]]).text
            st.subheader("Image Description:")
            st.write(response)

    # Scene Description
    if scene_button:
        with st.spinner("Analyzing scene..."):
            prompt = (
                "Analyze the uploaded image and describe the scene and every object in clear and"
                "simple language to assist visually impaired users. Answer must include key"
                "details with highlights about the environment, objects, people, and actions present in the image."
            )
            image_bytes = convert_image_to_bytes(uploaded_image)
            response = genai.GenerativeModel("gemini-1.5-pro").generate_content([prompt, image_bytes[0]]).text
            st.subheader("Scene Description:")
            st.write(response)
            speak_text(response)

    # Object Detection
    if detect_button:
        with st.spinner("Detecting objects..."):
            img = Image.open(uploaded_image)
            detections = perform_object_detection(img)
            annotated_image = highlight_objects(img.copy(), detections)
            st.image(annotated_image, caption="Objects Detected", use_column_width=True)

    # Assistance
    if assist_button:
        with st.spinner("Providing assistance..."):
            assistance_prompt = (
                "You are a helpful AI assistant designed for Visually impaired people."
                "Analyze the uploaded image (Boxes represent any obstacles or objects)"
                "and identify obstacles (and give numerically distance from obstacles, if car give real name) or objects so you can assist them with, tasks like"
                "crossing roads, playing with animals, doing their daily tasks, telling environment" 
                "around them , recognizing objects or reading labels."
            )
            image_bytes = convert_image_to_bytes(uploaded_image)
            assistance_response = genai.GenerativeModel("gemini-1.5-pro").generate_content([assistance_prompt, image_bytes[0]]).text
            st.subheader("Assistance:")
            st.write(assistance_response)
            speak_text(assistance_response)


if audio_button:
    try:
        # Initialize TTS engine if not already initialized
        if "tts_engine" not in st.session_state:
            st.session_state.tts_engine = pyttsx3.init()
        
        # Stoping the audio playback
        st.session_state.tts_engine.stop()
        st.success("Audio playback stopped.")
    except Exception as e:
        st.error(f"Failed to stop the audio. Error: {e}")