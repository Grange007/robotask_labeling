import base64
import cv2
import numpy as np
from PIL import Image
import os

def reshape_frame_to_512(image, timestamp):
    resized_image = cv2.resize(image, (512, 512))
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 30)  # 时间戳的位置，左上角
    font_scale = 1
    font_color = (255, 255, 255)  # 白色
    line_type = 2

    # 在图像上添加时间戳
    cv2.putText(resized_image, str(timestamp), 
                position, 
                font, 
                font_scale,
                font_color,
                line_type)
    return resized_image

def image_to_frame(image_path):
    frame = cv2.imread(image_path)
    return frame

def frame_to_image(frame, image_path):
    cv2.imwrite(image_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return image_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_frame_cv(frame):
    _, buffer = cv2.imencode('.png', frame)  # 编码为JPEG格式
    return base64.b64encode(buffer.tobytes()).decode('utf-8')

def get_frames_accelerate(frames, max_frames=60, max_acc_rate=5):
    if len(frames)<=max_frames:
        return frames
    if len(frames)>=max_acc_rate * max_frames:
        frames = frames[0:max_acc_rate * max_frames]
    acc_rate = int(len(frames)/max_frames)
    _frames = []
    for idx, frame in enumerate(frames):
        if idx % acc_rate==0:
            _frames.append(frame)
    return _frames[:max_frames]

def extract_frames_from_video(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    frames = []
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / frame_rate

    print(f"Frame Rate: {frame_rate}, Frame Count: {frame_count}, Duration: {duration}")

    # 每秒抽取一帧
    for sec in range(int(duration) + 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    print(len(frames))
    return frames

# def encode_to_base64(images, reshape_func=reshape_frame_to_512):
#     res = []
#     for i, img in enumerate(images):
#         print('&&&', img)
#         if not isinstance(img, np.ndarray):
#             img = image_to_frame(img)
#         print('===', img)
#         res.append(encode_frame_cv(reshape_func(img, i)))
#     print(len(res))
#     return res

def extract_images_from_video(video_path, output_path, max_frames=60, max_acc_rate=5):
    frames = extract_frames_from_video(video_path)
    info = {
        "video_path": video_path,
        "image_dir": output_path,
        "duration": len(frames)
    }
    os.makedirs(output_path, exist_ok=True)
    extracted_frames = get_frames_accelerate(frames, max_frames=max_frames, max_acc_rate=max_acc_rate)
    for i, frame in enumerate(extracted_frames):
        print(frame.shape, '====')
        frame_to_image(frame, os.path.join(output_path, f'frame_{i}.png'))
        print(os.path.join(output_path, f'frame_{i}.png'))
        frame2 = image_to_frame(os.path.join(output_path, f'frame_{i}.png'))
        print(frame2.shape, '&&&&')
    return info

if __name__ == "__main__":
    import sys
    # extract_images_from_video(sys.argv[1], sys.argv[2])
    # extract_images_from_video('/Users/bowen/Downloads/ego/egoexo4d_videos_clip/uniandes_dance_021_11/clip_0.mp4', 'ego4d_videos_clip_frames2/373e7a26-987e-4f82-ac56-bfb1f2d4a7d9#clip_0')
    extract_images_from_video("../demo_video/0b1a2d76ae4b62d02eb823a7b1bca0b8.mp4", "../demo_frames", 30)

