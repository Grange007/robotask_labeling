# your_script_distributed.py

from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig
from lmdeploy.vl import load_image
import time
import json
import os
import argparse
import re
from tqdm import tqdm
from PIL import Image
import torch
import torch.distributed as dist  # <<<<<< NEW: Import torch.distributed
import logging
from datetime import datetime

import torchvision.transforms as T
from decord import VideoReader, cpu
from lmdeploy.vl.constants import IMAGE_TOKEN
import numpy as np
from lmdeploy.vl.utils import encode_image_base64
import gc
import cv2
def parse_args():
    parser = argparse.ArgumentParser(description='Process video dataset with multi-modal model')
    parser.add_argument('--dataset_path', type=str, default="/mnt/world_foundational_model/pd_data/0625_robomind_format.jsonl",
                        help='Path to the input dataset JSONL file')
    parser.add_argument('--output_path', type=str, default="/mnt/world_foundational_model/pd_data/0625_robomind_longtext.jsonl",
                        help='Path to save the output JSONL file')
    parser.add_argument('--model_path', type=str, default="/media/users/wd/hf/hf_models/InternVL3-78B",
                        help='Path to the model directory')
    parser.add_argument('--max_frames', type=int, default=12,
                        help='Number of frames to sample from each video')
    # parser.add_argument('--original_fps', type=float, default=30.0,
    #                     help='Original video frame rate (frames per second)')
    parser.add_argument('--target_fps', type=float, default=5.0,
                        help='Target frame rate for sampling (frames per second)')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing')
    parser.add_argument('--new_key', type=str, default="dense_lang",
                        help='New key to add to the output records')
    parser.add_argument('--resume_start', type=int, default=0,
                        help='Use flash attention for faster processing')
    parser.add_argument('--show_video', action='store_true',
                        help='Save sampled frames as video')
    parser.add_argument('--seg_path', type=str, default="",
                        help='Path to existing segmentation JSON file to inject steps start/end into prompt')
    return parser.parse_args()

# ... (INSTRUCTION_TEMPLATE, find_closest_aspect_ratio, resize_img, build_instruction functions remain the same)
INSTRUCTION_TEMPLATE = """
Role and Goal
You are an expert system for robotic motion analysis, specializing in understanding and annotating robot manipulation tasks from video frames. Your primary goal is to meticulously analyze a provided sequence of chronologically ordered frames and a pre-segmented list of sub-action frames to produce a structured JSON annotation. You must describe each pre-defined segment with high fidelity.

Core Task
 * Analyze Video Frames: Carefully examine the sequence of frames, focusing on the actions within each pre-defined segment.
 * Generate Structured Annotations: For each segment provided by the frame list, fill out a JSON object that details the action, manipulated object, and spatial information. The verb for the action must be selected from the Skill Library provided.
 * Create Natural Language Instructions: Generate a concise instruction for each sub-action using the appropriate Instruction Template.
 * Strictly Adhere to Output Format: The final output must be a single, valid JSON object containing a list of all identified segments.
Skill Library (Action Verbs)
You MUST select the verb from the 'Skill Name (English)' column of the table below. Do not use any other verbs.

| Skill Name (English) |
| :------------------- |
| Pick / Grasp         |
| Place                |
| Push                 |
| Pull / Drag          |
| Insert               |
| Divide               |
| Open                 |
| Close                |
| Turn on              |
| Turn off             |
| Rotate / Turn        |
| Flip                 |
| Fold                 |
| Unfold               |
| Unwind               |
| Drop / Release       |
| Flatten              |
| Shake                |
| Stretch              |
| Straighten           |
| Press / Click        |
| Scan                 |
| Swipe                |
| Stick / Apply        |
| Screw / Tighten      |
| Unscrew              |
| Drill                |
| Spread               |
| Wipe                 |
| Sweep / Brush        |
| Stir                 |
| Scoop                |
| Suction              |
| Peel                 |
| Tie                  |
| Knock                |
| Beat                 |
| Scratch              |
| Hang                 |
| Throw                |
| Squeeze              |
| Twist                |
| Pluck                |
| Catch                |
| Hand to              |
| Pour                 |
| Wave                 |
| Clap                 |
| Point                |
Instruction Templates
Template 1 (Object Movement Actions): For actions involving clear displacement of an object from a start point to an end point.
Verbs: Place, Push, Pull, Drag, Insert, Throw, Pour, Hand to.
Format: "[verb] the [object] from [start_location] to [end_location]"

Template 2 (In-Place & State-Change Actions): For actions performed on an object at a single, specific location.
Verbs: Pick, Grasp, Release, Drop, Open, Close, Turn on, Turn off, Rotate, Turn, Flip, Press, Click, Screw, Unscrew, Tighten, Squeeze, Twist, Hang, Fold, Unfold, Tie, Pluck, Knock, Beat, Flatten, Shake, Stretch, Straighten, Wave, Clap, Point, Divide.
Format: "[verb] the [object] at [location]"
Note: Use start_location for actions like Pick/Grasp and end_location for actions like Release/Drop. For most others, start_location and end_location are the same; use this location in the template.

Template 3 (Surface/Tool Actions): For actions performed upon a surface or with a tool.
Verbs: Scan, Swipe, Stick, Apply, Drill, Spread, Wipe, Sweep, Brush, Stir, Peel, Suction, Scoop, Scratch.
Format: "[verb] the [object] on/at [location]"
Note: Use the location where the action is being performed (typically end_location).

Output Format (JSON Only)
{
  "segments_of_sub_actions": [
    {
      "start_frame": "Start frame index from original video (integer)",
      "end_frame": "End frame index from original video (integer)",
      "validity": "",
      "action": {
        "verb": "Action verb selected from the Skill Library",
        "object": "The specific object being manipulated",
        "start_location": "Detailed description of the object's initial position",
        "end_location": "Detailed description of the object's final position",
        "remarks": ""
      },
      "sub_action_instruction": "A combined instruction string generated from the appropriate template"
    }
  ]
}

Annotation Rules & Logic
 * Frame Indices:
   * Use the frame number of the index of each frame in the entire input sequence.
   * A segment must have start_frame ≤ end_frame.
 * Spatial Awareness (start_location / end_location):
   * BE A SPATIAL DETECTIVE. You must precisely describe locations.
   * Use Landmarks: Describe locations relative to fixed, visible landmarks in the environment (e.g., "blue component tray", "left side of the main assembly jig", "slot 3B of the storage rack", "inside the CNC machine").
   * Avoid Vague Terms: Do not use ambiguous terms like "on the table", "near the robot", or "in the area" if a more specific description is possible. Quantify if you can (e.g., "top shelf" vs. "shelf").
   * Movement vs. In-Place: For movement actions, start_location and end_location must be different. For in-place actions, they should be the same and describe the location of the action.
Critical Constraints
 * Analyze ONLY the provided frames and the pre-segmented frame list. Do not hallucinate actions or objects not visible.
 * The verb MUST be a valid skill from the 'Skill Name (English)' column in the Skill Library.
 * Locations MUST be as specific and descriptive as possible, based on visual evidence.
 * sub_action_instruction MUST be generated using the specified templates.
 * The final output MUST be a single valid JSON object.
"""

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    该函数用于在给定的目标宽高比集合（target_ratios）中，找到与输入图片宽高比（aspect_ratio）最接近的一个宽高比。

    参数说明：
    - aspect_ratio: 输入图片的宽高比（宽/高）。
    - target_ratios: 目标宽高比的集合，每个元素为一个元组 (w, h)。
    - width: 输入图片的宽度。
    - height: 输入图片的高度。
    - image_size: 目标图片的基准尺寸（通常为正方形边长）。

    实现逻辑：
    1. 遍历所有目标宽高比，计算每个目标宽高比与输入图片宽高比的差值（绝对值）。
    2. 记录差值最小的目标宽高比，作为最优选择。
    3. 如果有多个目标宽高比与输入图片宽高比的差值相同，则优先选择面积更大的目标宽高比（面积大于0.5 * image_size^2 * w * h）。
    4. 返回最接近的目标宽高比。

    返回值：
    - best_ratio: 与输入图片宽高比最接近的目标宽高比元组 (w, h)。
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def resize_img(image, min_num=1, max_num=6, image_size=448):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    return resized_img

def extract_json_from_response(response_text: str) -> dict:
    """
    从AI响应中提取JSON部分，增强错误处理和修复功能
    
    Args:
        response_text: AI的完整响应文本
        
    Returns:
        dict: 提取的JSON对象，如果提取失败返回空字典
    """
    try:
        # 尝试找到JSON代码块
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response_text, re.DOTALL)
        
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        
        # 如果没有找到代码块，尝试直接查找JSON对象
        json_pattern = r'\{.*"segments_of_sub_actions".*\}'
        match = re.search(json_pattern, response_text, re.DOTALL)
        
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        
        # 如果还是没找到，尝试查找任何JSON对象
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        match = re.search(json_pattern, response_text, re.DOTALL)
        
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
            
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"JSON提取失败: {e}")
        print(f"原始响应: {response_text[:200]}...")
    
    return {}

def map_frames_to_original_video(extracted_json: dict, frame_mapping: dict) -> dict:
    """
    将AI回答中的帧索引映射回原视频的索引
    
    Args:
        extracted_json: AI提取的JSON结果
        frame_mapping: 帧映射字典 {采样后索引: 原视频索引}
    
    Returns:
        dict: 映射后的JSON结果
    """
    if not extracted_json or not frame_mapping:
        return extracted_json
    
    # 深拷贝原始JSON以避免修改原数据
    final_json = json.loads(json.dumps(extracted_json))
    
    # 检查是否有segments_of_sub_actions字段
    if "segments_of_sub_actions" in final_json:
        segments = final_json["segments_of_sub_actions"]
        
        for segment in segments:
            # 映射start_frame
            if "start_frame" in segment:
                sampled_start = segment["start_frame"]
                # 如果start_frame是字符串，尝试转换为整数
                if isinstance(sampled_start, str):
                    try:
                        sampled_start = int(sampled_start)
                    except ValueError:
                        continue
                
                # 映射到原视频索引：AI回答中的索引对应采样后的索引
                if sampled_start in frame_mapping:
                    segment["start_frame"] = frame_mapping[sampled_start]
                else:
                    # 如果找不到映射，保持原值
                    pass
            
            # 映射end_frame
            if "end_frame" in segment:
                sampled_end = segment["end_frame"]
                # 如果end_frame是字符串，尝试转换为整数
                if isinstance(sampled_end, str):
                    try:
                        sampled_end = int(sampled_end)
                    except ValueError:
                        continue
                
                # 映射到原视频索引：AI回答中的索引对应采样后的索引
                if sampled_end in frame_mapping:
                    segment["end_frame"] = frame_mapping[sampled_end]
                else:
                    # 如果找不到映射，保持原值
                    pass
    
    return final_json

def generate_video_filename(video_path: str, rank: int) -> str:
    """
    从视频路径生成复杂的文件名
    
    Args:
        video_path: 视频文件路径
        rank: 当前进程的rank
    
    Returns:
        str: 生成的文件名
    """
    # 从完整路径中提取信息
    path_parts = video_path.split('/')
    
    # 查找关键信息
    type_info = ""
    task_info = path_parts[-5]
    date_info = ""
    camera_info = ""
    episode_info = ""
    
    for i, part in enumerate(path_parts):
        if part.startswith('type'):
            type_info = part
        elif 'camera' in part.lower():
            camera_info = part
        elif part.startswith('episode_'):
            episode_info = part
        # elif 'franka_' in part.lower() or 'ur_' in part.lower():
        #     task_info = part
    
    # 构建文件名
    filename_parts = []
    if task_info:
        filename_parts.append(task_info)
    # if type_info:
    #     filename_parts.append(type_info)
    # if date_info:
    #     filename_parts.append(date_info)
    if camera_info:
        filename_parts.append(camera_info)
    if episode_info:
        filename_parts.append(episode_info)
    
    # 如果没有找到足够的信息，使用原始文件名
    if len(filename_parts) < 2:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        filename_parts = [video_name]
    
    filename = '_'.join(filename_parts) + f"_sampled_rank{rank}.mp4"
    return filename

def save_sampled_video(images: list, output_path: str, fps: float = 2.0, frame_mapping: dict = None, original_fps: float = None):
    """
    将采样后的图像保存为视频
    
    Args:
        images: 图像列表（PIL Image对象）
        output_path: 输出视频路径
        fps: 视频帧率
        frame_mapping: 帧映射字典，用于在视频上显示帧索引信息
        original_fps: 原始视频帧率，用于计算时间戳
    """
    if not images:
        print(f"Warning: No images to save for {output_path}")
        return
    
    # 获取图像尺寸
    width, height = images[0].size
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    try:
        for i, img in enumerate(images):
            # 将PIL图像转换为OpenCV格式
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # 在图像上添加帧索引信息
            if frame_mapping and i in frame_mapping:
                original_frame = frame_mapping[i]
                text1 = f"Frame {i} -> Original {original_frame}"
                cv2.putText(img_cv, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                
                # 添加时间戳信息
                if original_fps:
                    timestamp = original_frame / original_fps
                    text2 = f"Time: {timestamp:.2f}s"
                    cv2.putText(img_cv, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 0), 2)
            
            out.write(img_cv)
    except Exception as e:
        print(f"Error saving video {output_path}: {e}")
    finally:
        out.release()
    
    print(f"Sampled video saved to: {output_path}")

def build_instruction(text: str, segmentation_data=None) -> str:
    """
    构建指令prompt，支持添加已有的划分结果
    
    Args:
        text: 原始文本描述
        segmentation_data: 已有的划分结果数据，包含steps字段
    """
    # 基础指令模板
    base_instruction = INSTRUCTION_TEMPLATE
                
    # 在基础指令前添加预划分信息
    enhanced_instruction = f"""
{base_instruction}
This is the original rough language instruction of the entire robot's trajectory for reference: {text}
Your annotation:
"""
    return enhanced_instruction

def _build_preseg_context_from_steps(steps, include_descriptions=True):
    """
    将已有segmentation的steps转换为JSON文本块
    如果include_descriptions=True，包含完整的step信息（包括step_description）
    如果include_descriptions=False，仅包含start/end帧信息
    """
    processed_steps = []
    for step in steps or []:
        if isinstance(step, dict) and 'start_frame' in step and 'end_frame' in step:
            if include_descriptions:
                # 包含完整的step信息
                processed_step = {
                    'start_frame': step['start_frame'],
                    'end_frame': step['end_frame']
                }
                if 'step_description' in step:
                    processed_step['step_description'] = step['step_description']
                processed_steps.append(processed_step)
            else:
                # 仅包含帧信息
                processed_steps.append({
                    'start_frame': step['start_frame'],
                    'end_frame': step['end_frame']
                })
    
    if not processed_steps:
        return ""
    
    segments_text = json.dumps({"segments_of_sub_actions": processed_steps}, indent=2)
    return f"\nPre-segmented frame list:\n{segments_text}\n"

def _index_segmentation_file(seg_items):
    """为seg文件建立多键索引，便于通过video名模糊匹配。"""
    by_episode = {}
    by_id = {}
    by_video_base = {}
    for item in seg_items:
        if not isinstance(item, dict):
            continue
        ep = item.get('episode_name') or item.get('id')
        if ep:
            by_episode[ep] = item
            by_id[item.get('id', ep)] = item
        ref = (item.get('id') or item.get('episode_name') or '')
        base = os.path.basename(ref)
        if base:
            by_video_base[base] = item
        core = base.split('.mp4')[0] if '.mp4' in base else base
        if core:
            by_video_base[core] = item
    return {"by_episode": by_episode, "by_id": by_id, "by_video_base": by_video_base}

def _extract_episode_key_from_video_path(video_path):
    """
    从video_path中提取episode关键信息，用于匹配segmentation数据
    
    Args:
        video_path: 视频文件路径
    
    Returns:
        str: 提取的episode关键信息
    """
    # 从完整路径中提取文件名
    filename = os.path.basename(video_path)
    
    # 移除.mp4扩展名
    if filename.endswith('.mp4'):
        filename = filename[:-4]
    
    # 提取关键部分：从路径中提取任务名称和episode信息
    # 例如：/path/to/franka_fr3_dualArm-gripper-6cameras_2_find_out_packaging_tape_into_the_other_basket_20250507/videos/chunk-000/observation.rgb_images.camera_top/episode_000000.mp4
    # 应该提取：franka_fr3_dualArm-gripper-6cameras_2_find_out_packaging_tape_into_the_other_basket_20250507_observation.rgb_images.camera_top_episode_000000
    
    # 查找包含任务名称的目录
    path_parts = video_path.split('/')
    task_name = ""
    episode_part = ""
    
    # 查找包含任务名称的目录（通常包含日期和任务描述）
    for part in path_parts:
        if 'franka_' in part or 'ur_' in part:
            task_name = part
            break
    
    # 提取episode部分
    if 'episode_' in filename:
        episode_part = filename
    
    # 组合关键信息
    if task_name and episode_part:
        # 移除episode_前缀，只保留数字部分
        episode_num = episode_part.replace('episode_', '')
        key = f"{task_name}_observation.rgb_images.camera_top_episode_{episode_num}"
        return key
    
    return filename

def _extract_task_name_from_video_path(video_path):
    """
    从video_path中提取任务名称，用于更精确的匹配
    
    Args:
        video_path: 视频文件路径
    
    Returns:
        str: 提取的任务名称
    """
    path_parts = video_path.split('/')
    
    # 查找包含任务名称的目录
    for part in path_parts:
        if 'franka_' in part or 'ur_' in part:
            return part
    
    return ""

def _find_segmentation_for_record(record, video_path, seg_index):
    """根据record和video_path在索引中查找对应的segmentation条目。"""
    if not seg_index:
        return None
    
    # 从video_path提取关键信息
    episode_key = _extract_episode_key_from_video_path(video_path)
    task_name = _extract_task_name_from_video_path(video_path)
    base = os.path.basename(video_path)
    core = base.split('.mp4')[0] if '.mp4' in base else base
    
    # 1. 尝试精确匹配episode_key
    if episode_key and episode_key in seg_index['by_episode']:
        return seg_index['by_episode'][episode_key]
    
    # 2. 尝试在by_video_base中精确匹配
    for key in (base, core, episode_key):
        if key and key in seg_index['by_video_base']:
            return seg_index['by_video_base'][key]
    
    # 3. 基于任务名称的模糊匹配
    if task_name:
        for k, item in seg_index['by_video_base'].items():
            # 检查segmentation数据中的episode_name是否包含任务名称
            seg_episode_name = item.get('episode_name', '')
            if task_name in seg_episode_name:
                # 进一步检查是否包含episode信息
                if 'episode_000000' in seg_episode_name and 'episode_000000' in video_path:
                    return item
                elif 'episode_' in seg_episode_name and 'episode_' in video_path:
                    return item
    
    # 4. 基于episode_key的模糊匹配
    if episode_key:
        for k, item in seg_index['by_video_base'].items():
            seg_episode_name = item.get('episode_name', '')
            # 检查是否包含关键信息
            if episode_key in seg_episode_name or seg_episode_name in episode_key:
                return item
    
    # 5. 最后的模糊匹配：基于文件名
    if core:
        for k, item in seg_index['by_video_base'].items():
            seg_episode_name = item.get('episode_name', '')
            if core in seg_episode_name or seg_episode_name in core:
                return item
    
    return None

def load_model(MODEL_PATH):
    # Model loading configuration remains the same, lmdeploy will use the visible GPUs.
    # tp=4 means it will use 4 GPUs for tensor parallelism.
    pipe = pipeline(
        MODEL_PATH, 
        backend_config=TurbomindEngineConfig(
            # session_len=32768,  # 增加session长度以支持更多帧
            session_len=131072,  # 增加session长度以支持更多帧
            tp=4,
            # # 添加内存优化配置
            # max_batch_size=1,
            # max_input_len=131072,
            # max_output_len=512
        ), 
        chat_template_config=ChatTemplateConfig(model_name='internvl2_5')
    )
    return pipe

def sample_frames(video_path: str, original_fps: float, target_fps: float, max_frames: int):
    """
    根据真实采集频率和需求频率进行抽帧
    
    Args:
        video_path: 视频文件路径
        original_fps: 原始视频帧率
        target_fps: 目标采样帧率
        max_frames: 最大采样帧数
    
    Returns:
        tuple: (imgs, frame_mapping) - 采样后的图像列表和帧索引映射字典
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    duration = total_frames / original_fps  # 视频总时长（秒）
    
    # 计算采样间隔（帧数）
    sampling_interval = original_fps / target_fps
    
    # 计算实际需要的帧数（基于目标帧率和视频时长）
    target_total_frames = int(duration * target_fps)
    
    # 使用较小的值：用户指定的帧数或基于目标帧率计算的帧数
    actual_num_frames = min(max_frames, target_total_frames)
    
    # 生成采样索引
    if actual_num_frames == 1:
        # 如果只需要一帧，取中间帧
        indices = [total_frames // 2]
    else:
        # 均匀采样
        indices = np.linspace(0, total_frames - 1, num=actual_num_frames, dtype=int)
    
    imgs = []
    frame_mapping = {}  # 映射字典：{采样后索引: 原视频索引}
    
    for i, frame_index in enumerate(indices):
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = resize_img(img)
        img = img.resize((480, 480))
        imgs.append(img)
        frame_mapping[i] = int(frame_index)  # 记录映射关系
    
    return imgs, frame_mapping

def setup_logging(rank, output_dir):
    """
    设置日志记录器，为每个rank创建独立的日志文件
    
    Args:
        rank: 当前进程的rank
        output_dir: 输出目录路径
    
    Returns:
        logger: 配置好的日志记录器
    """
    # 创建日志目录
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名
    log_filename = f"processing_timing_rank{rank}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # 配置日志记录器
    logger = logging.getLogger(f"rank_{rank}")
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器（仅rank 0）
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
    
    # 设置文件处理器格式
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

def setup_distributed():
    """
    Initializes the distributed process group. 
    初始化分布式进程组，并返回当前进程的rank和总进程数world_size。

    该函数使用NCCL后端初始化分布式环境，适用于多GPU/多节点分布式推理或训练。
    torchrun会自动设置LOCAL_RANK，lmdeploy/PyTorch会根据CUDA_VISIBLE_DEVICES自动分配设备，
    因此无需手动调用torch.cuda.set_device。
    
    返回:
        rank (int): 当前进程的rank编号（在所有进程中的唯一标识）。
        world_size (int): 总的进程数（即参与分布式的所有进程数量）。
    """
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # torchrun sets LOCAL_RANK automatically. lmdeploy/PyTorch will use it.
    # We don't need to call torch.cuda.set_device here because lmdeploy handles device management
    # based on CUDA_VISIBLE_DEVICES.
    return rank, world_size

def main():
    args = parse_args()
    rank_from_env = int(os.environ['RANK'])
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
    if rank_from_env % 2 == 0:
        # Even-ranked processes (0, 2, ...) are assigned GPUs 0, 1, 2, 3
        os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    else:
        # Odd-ranked processes (1, 3, ...) are assigned GPUs 4, 5, 6, 7
        os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"
    # <<<<<< NEW: Setup distributed environment
    rank, world_size = setup_distributed()

    DATASET_PATH = args.dataset_path
    OUTPUT_PATH = args.output_path
    MAX_FRAMES = args.max_frames
    # ORIGINAL_FPS = args.original_fps
    TARGET_FPS = args.target_fps
    MAX_NEW_TOKENS = args.max_new_tokens
    BATCH_SIZE = args.batch_size
    NEW_KEY = args.new_key
    MODEL_PATH = args.model_path
    RESUME_STRART = args.resume_start
    SHOW_VIDEO = args.show_video

    # <<<<<< MODIFIED: Each rank writes to its own file to avoid conflicts
    output_dir = os.path.dirname(OUTPUT_PATH)
    output_filename = os.path.basename(OUTPUT_PATH)
    base, ext = os.path.splitext(output_filename)
    # Example: pd_02_tmp.jsonl -> pd_02_tmp_rank0.jsonl
    rank_output_path = os.path.join(output_dir, f"{base}_rank{rank}{ext}")
    
    # 设置日志记录器
    logger = setup_logging(rank, output_dir)
    
    if rank == 0:
        print(f"Starting distributed run with {world_size} processes.")
        logger.info(f"Starting distributed run with {world_size} processes.")
    print(f"[Rank {rank}] Model Path: {MODEL_PATH}")
    print(f"[Rank {rank}] Output will be saved to: {rank_output_path}")
    logger.info(f"Model Path: {MODEL_PATH}")
    logger.info(f"Output will be saved to: {rank_output_path}")

    pipe = load_model(MODEL_PATH)
    generation_config = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=2)
    
    with open(DATASET_PATH, "r") as f:
        all_records = [json.loads(line) for line in f if line.strip()]
        # all_records = all_records[:50]

    # 读取并索引已有分割结果（可选）
    seg_index = None
    if args.seg_path and os.path.exists(args.seg_path):
        try:
            seg_items = json.load(open(args.seg_path, 'r'))
            if isinstance(seg_items, list):
                seg_index = _index_segmentation_file(seg_items)
                if rank == 0:
                    print(f"Loaded {len(seg_items)} items from seg_path: {args.seg_path}")
            else:
                if rank == 0:
                    print(f"seg_path file is not a list: {args.seg_path}")
        except Exception as e:
            if rank == 0:
                print(f"Failed to load seg_path {args.seg_path}: {e}")

    records_for_this_rank = all_records[rank::world_size]
    total_records = len(all_records)
    num_records_this_rank = len(records_for_this_rank)
    
    if rank == 0:
        print(f"Total dataset has {total_records} records.")
        logger.info(f"Total dataset has {total_records} records.")
    print(f"[Rank {rank}] Processing {num_records_this_rank} records.")
    logger.info(f"Processing {num_records_this_rank} records on rank {rank}")

    output_f = open(rank_output_path, "w")

    # 初始化时间统计变量
    start = time.time()
    total_processing_time = 0
    record_times = []  # 存储每个record的处理时间
    batch_times = []   # 存储每个batch的处理时间
    try: 
        for i in tqdm(range(0, num_records_this_rank, BATCH_SIZE), desc=f"Rank {rank} Processing", disable=(rank!=0)):
            batch_start_time = time.time()
            batch_records = records_for_this_rank[i:i + BATCH_SIZE]
            data = []
            frame_mappings = []  # 存储每个记录的帧映射
        
            for record_idx, record in enumerate(batch_records):
                record_start_time = time.time()
                video_path = record.get("video_path", "")
                if video_path.startswith("/mnt/world_foundational_model"):
                    pass
                else:
                    video_path = video_path.replace("/mnt", "/mnt/world_foundational_model") # Adjust path if necessary
                text = record.get("lang", "")
                original_fps = record.get("tele_fps")
                
                # 记录视频处理开始时间
                video_processing_start = time.time()
                images, frame_mapping = sample_frames(video_path, original_fps, TARGET_FPS, MAX_FRAMES)
                video_processing_time = time.time() - video_processing_start
                frame_mappings.append(frame_mapping)  # 保存帧映射
                
                # 如果启用视频保存功能，保存采样后的视频
                if SHOW_VIDEO:
                    # 创建视频输出目录
                    video_output_dir = os.path.join(os.path.dirname(rank_output_path), "sampled_videos")
                    os.makedirs(video_output_dir, exist_ok=True)
                    
                    # 生成更复杂的视频文件名
                    filename = generate_video_filename(video_path, rank)
                    video_output_path = os.path.join(video_output_dir, filename)
                    
                    # 保存采样后的视频
                    save_sampled_video(images, video_output_path, TARGET_FPS, frame_mapping, original_fps)
                
                # 记录图像加载时间
                image_loading_start = time.time()
                images = [load_image(x) for x in images]
                image_loading_time = time.time() - image_loading_start
                
                prompt = ""
                for img_idx in range(len(images)):
                    prompt += f'Frame{img_idx + 1}: {IMAGE_TOKEN}\n'
                
                # 优先使用seg_path中的预分割结果（包含完整信息）
                if seg_index is not None:
                    matched = _find_segmentation_for_record(record, video_path, seg_index)
                    if matched:
                        steps = ((matched.get('segmentation') or {}).get('steps'))
                        task_summary = matched.get('task_summary', '')
                        
                        # 记录匹配成功的信息
                        video_name = os.path.basename(video_path)
                        logger.info(f"Found segmentation match for {video_name}: {matched.get('id', 'unknown_id')}")
                        
                        # 构建包含task_summary和完整step信息的上下文
                        preseg_context = ""
                        if task_summary:
                            preseg_context += f"\nTask Summary from Stage 1: {task_summary}\n"
                        
                        # 包含完整的step信息（包括step_description）
                        preseg_text = _build_preseg_context_from_steps(steps, include_descriptions=True)
                        preseg_context += preseg_text
                        
                        # 保持与本脚本的build_instruction组合方式一致
                        prompt += f"{preseg_context}{build_instruction(text)}"
                    else:
                        # 记录未找到匹配的信息
                        video_name = os.path.basename(video_path)
                        episode_key = _extract_episode_key_from_video_path(video_path)
                        task_name = _extract_task_name_from_video_path(video_path)
                        logger.warning(f"No segmentation match found for {video_name}")
                        logger.warning(f"  - episode_key: {episode_key}")
                        logger.warning(f"  - task_name: {task_name}")
                        logger.warning(f"  - video_path: {video_path}")
                        prompt += build_instruction(text)
                else:
                    prompt += build_instruction(text)
                content = [{'type': 'text', 'text': prompt}]
                for img in images:
                    content.append({'type': 'image_url', 'image_url': {'max_dynamic_patch': 1, 'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'}})
                """
                [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': 'Frame1: <image>\nFrame2: <image>\nFrame3: <image>\n\nYou are an expert...'},
                            {'type': 'image_url', 'image_url': {'max_dynamic_patch': 1, 'url': 'data:image/jpeg;base64,...'}},
                            {'type': 'image_url', 'image_url': {'max_dynamic_patch': 1, 'url': 'data:image/jpeg;base64,...'}},
                            {'type': 'image_url', 'image_url': {'max_dynamic_patch': 1, 'url': 'data:image/jpeg;base64,...'}}
                        ]
                    }
                ]
                """
                data.append(dict(role='user', content=content))
                # data.append({'prompt': prompt, 'images': images})

            # 记录模型推理时间
            inference_start = time.time()
            responses = pipe(data, gen_config=generation_config)
            inference_time = time.time() - inference_start
            
            if not isinstance(responses,list):
                responses = [responses]
            for record_idx, (record, response, frame_mapping) in enumerate(zip(batch_records, responses, frame_mappings)):
                # 计算单个record的总处理时间
                record_end_time = time.time()
                record_total_time = record_end_time - record_start_time
                record_times.append(record_total_time)
                
                # 记录详细的时间信息到record中
                record[f"{NEW_KEY}_timing"] = {
                    "video_processing_time": video_processing_time,
                    "image_loading_time": image_loading_time,
                    "inference_time": inference_time / len(batch_records),  # 平均推理时间
                    "total_record_time": record_total_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                record[f"{NEW_KEY}_original"] = response.text
                # record[NEW_KEY] = response.text
                # output_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                # 提取JSON并保存到新的key中
                extracted_json = extract_json_from_response(response.text)
                if extracted_json:
                    record[f"{NEW_KEY}_json"] = extracted_json
                else:
                    record[f"{NEW_KEY}_json"] = {}
                
                # 添加帧映射信息到记录中
                record[f"{NEW_KEY}_frame_mapping"] = frame_mapping
                
                # 如果保存了视频，添加视频路径信息
                if SHOW_VIDEO:
                    filename = generate_video_filename(record.get("video_path", ""), rank)
                    video_output_path = os.path.join("sampled_videos", filename)
                    record[f"{NEW_KEY}_sampled_video_path"] = video_output_path
                
                # 创建映射回原视频索引的JSON
                final_json = map_frames_to_original_video(extracted_json, frame_mapping)
                record[f"{NEW_KEY}_final_json"] = final_json
                
                # 记录到日志
                video_name = os.path.basename(record.get("video_path", "unknown"))
                logger.info(f"Record {i + record_idx + 1}/{num_records_this_rank} - {video_name}: "
                          f"Total={record_total_time:.2f}s, Video={video_processing_time:.2f}s, "
                          f"Image={image_loading_time:.2f}s, Inference={inference_time/len(batch_records):.2f}s")
                
                # 保存便于阅读的JSON格式
                output_f.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
                output_f.flush()
            
            # 记录batch处理时间
            batch_end_time = time.time()
            batch_total_time = batch_end_time - batch_start_time
            batch_times.append(batch_total_time)
            
            # 记录batch统计信息到日志
            logger.info(f"Batch {i//BATCH_SIZE + 1}: {len(batch_records)} records processed in {batch_total_time:.2f}s "
                      f"(avg {batch_total_time/len(batch_records):.2f}s per record)")
            
            # 清理内存
            del data, responses, frame_mappings
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        output_f.close()
    
    end = time.time()
    total_time = end - start
    
    # 计算统计信息
    if record_times:
        avg_record_time = sum(record_times) / len(record_times)
        min_record_time = min(record_times)
        max_record_time = max(record_times)
    else:
        avg_record_time = min_record_time = max_record_time = 0
    
    if batch_times:
        avg_batch_time = sum(batch_times) / len(batch_times)
        min_batch_time = min(batch_times)
        max_batch_time = max(batch_times)
    else:
        avg_batch_time = min_batch_time = max_batch_time = 0
    
    # 输出最终统计信息
    print(f"[Rank {rank}] Processing finished! Time taken: {total_time:.2f} seconds.")
    print(f"[Rank {rank}] Results saved to: {rank_output_path}")
    print(f"[Rank {rank}] Record timing stats: avg={avg_record_time:.2f}s, min={min_record_time:.2f}s, max={max_record_time:.2f}s")
    print(f"[Rank {rank}] Batch timing stats: avg={avg_batch_time:.2f}s, min={min_batch_time:.2f}s, max={max_batch_time:.2f}s")
    
    # 记录到日志
    logger.info(f"Processing finished! Total time: {total_time:.2f} seconds")
    logger.info(f"Results saved to: {rank_output_path}")
    logger.info(f"Record timing statistics:")
    logger.info(f"  - Total records processed: {len(record_times)}")
    logger.info(f"  - Average time per record: {avg_record_time:.2f}s")
    logger.info(f"  - Min time per record: {min_record_time:.2f}s")
    logger.info(f"  - Max time per record: {max_record_time:.2f}s")
    logger.info(f"Batch timing statistics:")
    logger.info(f"  - Total batches processed: {len(batch_times)}")
    logger.info(f"  - Average time per batch: {avg_batch_time:.2f}s")
    logger.info(f"  - Min time per batch: {min_batch_time:.2f}s")
    logger.info(f"  - Max time per batch: {max_batch_time:.2f}s")
    
    # 保存统计信息到JSON文件
    stats = {
        "rank": rank,
        "total_time": total_time,
        "total_records": len(record_times),
        "total_batches": len(batch_times),
        "record_timing": {
            "average": avg_record_time,
            "min": min_record_time,
            "max": max_record_time,
            "all_times": record_times
        },
        "batch_timing": {
            "average": avg_batch_time,
            "min": min_batch_time,
            "max": max_batch_time,
            "all_times": batch_times
        },
        "timestamp": datetime.now().isoformat()
    }
    
    stats_file = os.path.join(output_dir, "logs", f"timing_stats_rank{rank}.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Timing statistics saved to: {stats_file}")

if __name__ == "__main__":
    main()