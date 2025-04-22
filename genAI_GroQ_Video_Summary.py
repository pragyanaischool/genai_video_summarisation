import os
import cv2
import streamlit as st
from langchain_groq import ChatGroq

videos_directory = 'videos/'
frames_directory = 'frames/'

# Initialize Groq model
model = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"  # You can also use llama3-70b or gemma
)

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
    descriptions = []
    for file in sorted(os.listdir(frames_directory)):
        frame_path = os.path.join(frames_directory, file)
        descriptions.append(f"Describe this frame: {frame_path}")
    prompt = "You are a helpful assistant. Summarize the video based on the following image filenames:\n" + "\n".join(descriptions)
    return model.invoke(prompt)

# Streamlit UI
st.title("🎥 AI Video Summarizer (Groq-powered)")

uploaded_file = st.file_uploader(
    "Upload a video to generate summary",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file:
    with st.spinner("Processing video..."):
        os.makedirs(videos_directory, exist_ok=True)
        os.makedirs(frames_directory, exist_ok=True)

        upload_video(uploaded_file)
        extract_frames(os.path.join(videos_directory, uploaded_file.name))
        summary = describe_video()

    st.markdown("### 📝 Summary:")
    st.markdown(summary)
