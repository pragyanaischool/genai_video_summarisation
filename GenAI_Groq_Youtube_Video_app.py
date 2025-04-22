import os
import cv2
import streamlit as st
from pytube import YouTube
from langchain_groq import ChatGroq

videos_directory = 'videos/'
frames_directory = 'frames/'

# Initialize Groq model
model = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

def download_youtube_video(youtube_url):
    os.makedirs(videos_directory, exist_ok=True)
    yt = YouTube(youtube_url)
    video_stream = yt.streams.filter(progressive=True, file_extension='mp4').get_highest_resolution()
    video_path = os.path.join(videos_directory, f"{yt.title}.mp4")
    video_stream.download(output_path=videos_directory, filename=f"{yt.title}.mp4")
    return video_path

def extract_frames(video_path, interval_seconds=5):
    os.makedirs(frames_directory, exist_ok=True)
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
st.title("📺 YouTube Video Summarizer (Groq-powered)")

youtube_url = st.text_input("Enter YouTube video URL")

if youtube_url:
    with st.spinner("Downloading and processing YouTube video..."):
        video_path = download_youtube_video(youtube_url)
        extract_frames(video_path)
        summary = describe_video()
    st.markdown("### 📝 Summary:")
    st.markdown(summary)

st.divider()

uploaded_file = st.file_uploader(
    "Or upload a local video to generate summary",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file:
    with st.spinner("Processing uploaded video..."):
        os.makedirs(videos_directory, exist_ok=True)
        os.makedirs(frames_directory, exist_ok=True)

        with open(os.path.join(videos_directory, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

        extract_frames(os.path.join(videos_directory, uploaded_file.name))
        summary = describe_video()

    st.markdown("### 📝 Summary:")
    st.markdown(summary)
