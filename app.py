import cv2
import base64
import openai
import os
from dotenv import load_dotenv
import requests

load_dotenv()


def run_by_video(video_path):
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
    prompt = "这是一个视频的连续帧，视频的总时长是0:05，生成一个诙谐幽默的描述，作为这个视频的字幕，输出字幕格式，必须是中文，这对很重要"
    return run(base64Frames, prompt, 30)


def run_by_img(img_path):
    with open(img_path, "rb") as f:
        base64Frames = [base64.b64encode(f.read()).decode("utf-8")]
    prompt = "这是一张关于12星座的描述图片，理解图片内容，并且生成一个简洁的短视频脚本，必须是中文，这对很重要"
    return run(base64Frames, prompt, 1)


def run(base64Frames, prompt, steps=10):
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                prompt,
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
    # run_by_video("data/cat.mp4")
    run_by_img("data/2.png")
