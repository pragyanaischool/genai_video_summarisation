import os
import cv2
import streamlit as st
import httpx
from langchain_ollama.llms import OllamaLLM

videos_directory = 'videos/'
frames_directory = 'frames/'

# Make sure Ollama is running at this URL
OLLAMA_BASE_URL = "http://localhost:11434"

# Initialize model
model = OllamaLLM(model="gemma3:27b", base_url=OLLAMA_BASE_URL)

def check_ollama_server():
    try:
        response = httpx.get(OLLAMA_BASE_URL)
        return response.status_code == 200
    except httpx.ConnectError:
        return False

def upload_video(file):
    with open(os.path.join(videos_directory, file.name), "wb") as f:
        f.write(file.getbuffer())

def extract_frames(video_path, interval_seconds=5):
    for file in os.listdir(frames_directory):
        os.remove(os.path.join(frames_directory, file))

    video = cv2.VideoCapture(video_path)

    fps = int(video.get(cv2.CAP_PROP_FPS))
    frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    current_frame = 0
    frame_number = 1

    while current_frame <= frames_count:
        video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        success, frame = video.read()

        if not success:
            current_frame += fps * interval_seconds
            continue

        frame_path = os.path.join(frames_directory, f"frame_{frame_number:03d}.jpg")
        cv2.imwrite(frame_path, frame)

        current_frame += fps * interval_seconds
        frame_number += 1

    video.release()

def describe_video():
    images = [
        os.path.join(frames_directory, file)
        for file in sorted(os.listdir(frames_directory))
    ]

    model_with_images = model.bind(images=images)
    return model_with_images.invoke("Summarize the video content in a few sentences.")

# Streamlit interface
st.title("🎥 AI Video Summarizer")

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov", "mkv"],
    accept_multiple_files=False
)

if uploaded_file:
    if not check_ollama_server():
        st.error("❌ Cannot connect to Ollama server at `localhost:11434`. Please start it with `ollama serve`.")
    else:
        with st.spinner("Processing video..."):
            upload_video(uploaded_file)
            extract_frames(os.path.join(videos_directory, uploaded_file.name))
            summary = describe_video()

        st.markdown("### 📝 Summary:")
        st.markdown(summary)
