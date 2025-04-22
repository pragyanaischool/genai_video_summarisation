import os
import cv2
import streamlit as st
from pytube import YouTube
import subprocess
from langchain_groq import ChatGroq

# Directories
videos_directory = 'videos/'
frames_directory = 'frames/'
os.makedirs(videos_directory, exist_ok=True)
os.makedirs(frames_directory, exist_ok=True)

# Initialize Groq model
model = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)
'''
def download_youtube_video(youtube_url):
    os.makedirs(videos_directory, exist_ok=True)
    yt = YouTube(youtube_url)
    video_stream = yt.streams.filter(progressive=True, file_extension='mp4').get_highest_resolution()
    video_path = os.path.join(videos_directory, f"{yt.title}.mp4")
    video_stream.download(output_path=videos_directory, filename=f"{yt.title}.mp4")
    return video_path
'''
# Download YouTube video using yt-dlp
def download_youtube_video(youtube_url):
    result = subprocess.run(
        [
            "yt-dlp",
            "-f", "best[ext=mp4]",
            "-o", os.path.join(videos_directory, "%(title)s.%(ext)s"),
            youtube_url
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp error:\n{result.stderr}")

    downloaded_files = sorted(
        os.listdir(videos_directory),
        key=lambda x: os.path.getctime(os.path.join(videos_directory, x)),
        reverse=True
    )
    return os.path.join(videos_directory, downloaded_files[0])

# Extract frames from the video
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

# Describe video content using Groq
def describe_video():
    descriptions = []
    for file in sorted(os.listdir(frames_directory)):
        frame_path = os.path.join(frames_directory, file)
        descriptions.append(f"Describe this frame: {frame_path}")
    prompt = "You are a helpful assistant. Summarize the video based on the following image filenames:\n" + "\n".join(descriptions)
    return model.invoke(prompt)
    
# Streamlit UI
st.title("📺 YouTube Video Summarizer (Groq-powered)")

youtube_url = st.text_input("Paste a YouTube video URL to analyze:", placeholder="https://www.youtube.com/watch?v=83sdwFOL1r8")

if youtube_url:
    try:
        with st.spinner("Downloading and processing video..."):
            video_path = download_youtube_video(youtube_url)
            extract_frames(video_path)
            summary = describe_video()

        st.markdown("### 📝 Video Summary:")
        st.markdown(summary)

    except Exception as e:
        st.error(f"❌ Error: {e}")

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
