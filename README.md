# AI Vision Assistance for Visually Impaired

## Description
This project is designed to assist visually impaired users by providing real-time image and video processing features using AI. The application can extract text from images, detect objects, describe scenes, and provide assistance in tasks like crossing roads and identifying objects, all powered by generative AI models and computer vision techniques.

Key features include:
- **Text Extraction from Images**: Using OCR (Optical Character Recognition) to read text from images.
- **Scene Description**: Describes the environment, objects, and actions in the image.
- **Object Detection**: Detects and highlights objects in an image using a pre-trained object detection model.
- **Real-Time Video Capture & Assistance**: Captures video from the webcam, processes frames, and provides assistance in real-time.

## Requirements
To run this project, you need the following dependencies:

- Python 3.7 or later
- Streamlit
- Google Generative AI SDK
- PyTesseract (OCR library)
- Pyttsx3 (Text-to-speech engine)
- Torch (for object detection model)
- OpenCV (for video capture)
- Pillow (for image processing)
- dotenv (for loading environment variables)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-vision-assistance.git
