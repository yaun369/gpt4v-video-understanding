import cv2
import base64
import openai
import os
from dotenv import load_dotenv
import requests

load_dotenv()

def handle_video(video_path):
    video = cv2.VideoCapture(video_path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    print(len(base64Frames), "frames read.")
    return base64Frames


def run(base64Frames, steps=10):
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                "These are frames from a video that I want to upload. Generate a compelling description that I can upload along with the video.MUST BE ANSWERED IN CHINESE, THIS IS VERY IMPORTANT TO ME.",
                *map(lambda x: {"image": x, "resize": 768},
                     base64Frames[0::steps]),
            ],
        },
    ]
    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "api_key": os.environ["OPENAI_API_KEY"],
        "headers": {"Openai-Version": "2020-11-07"},
        "max_tokens": 500,
    }

    result = openai.ChatCompletion.create(**params)
    text = result.choices[0].message.content
    print(text)
    return text


if __name__ == "__main__":
    base64Frames = handle_video("data/cat.mp4")
    run(base64Frames, 30)
