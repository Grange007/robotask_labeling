from prompt_utils import *
from api_utils import *
import glob
import argparse
import json
import re
import asyncio
import time
import os
from prompt_manager import create_prompt_manager, get_available_versions

# InternVL2.5 相关导入
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64
from PIL import Image
import torch
import gc

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """InternVL2.5 图像预处理：找到最接近的宽高比"""
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
    """InternVL2.5 图像预处理：调整图像大小"""
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

def load_internvl_model(model_path):
    """加载InternVL2.5模型"""
    print(f"Loading InternVL2.5 model from: {model_path}")
    pipe = pipeline(
        model_path, 
        backend_config=TurbomindEngineConfig(
            session_len=32768,  # 支持长序列
            tp=1,  # 单GPU推理，可根据需要调整
        ), 
        chat_template_config=ChatTemplateConfig(model_name='internvl2_5')
    )
    return pipe

def format_segmentation_message_universal(path_to_frames, prompt_manager, embodiment=None, desc=None, **kwargs):
    """第一阶段：使用Gemini的消息格式化"""
    images = os.listdir(path_to_frames)
    images = sorted(images, key=lambda x: int(x[x.index('_')+1:x.index('.png')]))
    images = [os.path.join(path_to_frames, img) for img in images]
    
    if not desc:
        desc = ""
    
    prompt = prompt_manager.format_segmentation_prompt(desc)
    message = {
        "request_id": f"{path_to_frames}_segmentation", 
        "prompt": prompt, 
        "image_base64": [image_to_frame(img) for img in images]
    } 
    return message

def format_detailed_analysis_message_internvl(path_to_frames, segments, prompt_manager, task_summary="", embodiment=None, desc=None, include_step_descriptions=True, **kwargs):
    """第二阶段：使用InternVL2.5的消息格式化"""
    images = os.listdir(path_to_frames)
    images = sorted(images, key=lambda x: int(x[x.index('_')+1:x.index('.png')]))
    images = [os.path.join(path_to_frames, img) for img in images]
    
    if not desc:
        desc = ""
    
    # 获取基础prompt
    base_prompt = prompt_manager.get_detailed_prompt()
    
    # 根据参数决定是否包含step descriptions
    if include_step_descriptions:
        # 获取格式化的上下文（包含step descriptions）
        context = prompt_manager.format_detailed_prompt_context(segments, task_summary, desc)
    else:
        # 只传递task_summary，不传递segments的step descriptions
        filtered_segments = []
        for segment in segments:
            filtered_segment = {
                'start_frame': segment.get('start_frame'),
                'end_frame': segment.get('end_frame')
            }
            filtered_segments.append(filtered_segment)
        context = prompt_manager.format_detailed_prompt_context(filtered_segments, task_summary, desc)
    
    # 组合完整prompt
    full_prompt = f"{base_prompt}{context}"
    
    # 构建InternVL2.5格式的prompt
    prompt = ""
    for img_idx in range(len(images)):
        prompt += f'Frame{img_idx + 1}: {IMAGE_TOKEN}\n'
    
    prompt += full_prompt
    
    # 预处理图像
    processed_images = []
    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        img = resize_img(img)
        img = img.resize((480, 480))  # 统一调整到480x480
        processed_images.append(img)
    
    return {
        "prompt": prompt,
        "images": processed_images,
        "request_id": f"{path_to_frames}_detailed_internvl"
    }

async def request_internvl_model(message_data, internvl_pipe, generation_config):
    """使用InternVL2.5进行推理"""
    try:
        # 准备输入数据
        prompt = message_data["prompt"]
        images = message_data["images"]
        
        # 将图像转换为InternVL需要的格式
        loaded_images = [load_image(img) for img in images]
        
        # 构建内容
        content = [{'type': 'text', 'text': prompt}]
        for img in loaded_images:
            content.append({
                'type': 'image_url', 
                'image_url': {
                    'max_dynamic_patch': 1, 
                    'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'
                }
            })
        
        # 构建消息格式
        messages = [{'role': 'user', 'content': content}]
        
        # 调用模型
        response = internvl_pipe(messages, gen_config=generation_config)
        
        if isinstance(response, list):
            response = response[0]
        
        return {
            'id': message_data.get("request_id", "unknown"),
            'response': response.text
        }
        
    except Exception as e:
        print(f"InternVL推理出错: {e}")
        return {
            'id': message_data.get("request_id", "unknown"),
            'response': None,
            'error': str(e)
        }

async def process_hybrid_two_stage_pipeline(frame_paths, desc, embodiment, prompt_version="v1", result_file=None, seg_result_file=None, include_step_descriptions=True, internvl_model_path=None):
    """混合两阶段处理管道：Stage1用Gemini，Stage2用InternVL2.5"""
    
    # 记录总体开始时间
    total_start_time = time.time()
    
    prompt_manager = create_prompt_manager(prompt_version)
    version_info = prompt_manager.get_version_info()
    
    print(f"=== Hybrid Pipeline Configuration ===")
    print(f"Stage 1: Gemini (Segmentation)")
    print(f"Stage 2: InternVL2.5 (Detailed Analysis)")
    print(f"Using prompt version: {version_info['version']}")
    print(f"Output format: {version_info['output_format']}")
    print(f"Has task summary: {version_info['has_task_summary']}")
    print(f"Has step descriptions: {version_info['has_step_descriptions']}")
    print(f"Include step descriptions in stage 2: {include_step_descriptions}")
    print()
    
    # 加载InternVL2.5模型
    if not internvl_model_path:
        internvl_model_path = "/media/users/wd/hf/hf_models/InternVL3-78B"  # 默认路径
        print(f"Warning: Using default InternVL model path: {internvl_model_path}")
    
    internvl_pipe = load_internvl_model(internvl_model_path)
    generation_config = GenerationConfig(max_new_tokens=512, min_new_tokens=2)
    
    all_results = []
    seg_results_list = []  # 存储第一阶段结果
    time_records = []  # 存储时间记录
    
    for i, frame_path in enumerate(frame_paths):
        # 记录单个episode开始时间
        episode_start_time = time.time()
        print(f'=== Processing {i+1}/{len(frame_paths)}: {frame_path} ===')
        
        episode_name = os.path.basename(frame_path)
        episode_desc = desc.get(episode_name, {}).get('desc', '')
        
        try:
            # 第一阶段：使用Gemini进行分割
            stage1_start_time = time.time()
            print("Stage 1: Segmentation (Gemini)...")
            seg_message = format_segmentation_message_universal(frame_path, prompt_manager, embodiment, episode_desc)
            seg_results = await request_model([seg_message])  # 使用原有的Gemini API
            stage1_end_time = time.time()
            stage1_duration = stage1_end_time - stage1_start_time
            
            if not seg_results or not seg_results[0].get('response'):
                print(f"Failed to get segmentation result for {frame_path}")
                continue
                
            # 解析分割结果
            segments, task_summary = prompt_manager.parse_segmentation_result(seg_results[0]['response'])
            if not segments:
                print(f"Failed to parse segments for {frame_path}")
                continue
                
            print(f"Found {len(segments)} segments")
            if task_summary:
                print(f"Task summary: {task_summary}")
            
            # 保存第一阶段结果
            seg_result = {
                'id': frame_path,
                'prompt_version': prompt_version,
                'segmentation_response': seg_results[0]['response'],
                'segments': segments,
                'task_summary': task_summary,
                'version_info': version_info
            }
            seg_results_list.append(seg_result)
            
            # 第二阶段：使用InternVL2.5进行详细分析
            stage2_start_time = time.time()
            print("Stage 2: Detailed analysis (InternVL2.5)...")
            detail_message = format_detailed_analysis_message_internvl(
                frame_path, segments, prompt_manager, task_summary, embodiment, episode_desc, include_step_descriptions
            )
            detail_results = await request_internvl_model(detail_message, internvl_pipe, generation_config)
            stage2_end_time = time.time()
            stage2_duration = stage2_end_time - stage2_start_time
            
            if detail_results and detail_results.get('response'):
                # 计算episode总时间
                episode_end_time = time.time()
                episode_duration = episode_end_time - episode_start_time
                
                # 记录时间信息
                time_record = {
                    'episode_id': frame_path,
                    'episode_name': episode_name,
                    'stage1_duration': stage1_duration,
                    'stage2_duration': stage2_duration,
                    'total_episode_duration': episode_duration,
                    'stage1_model': 'gemini',
                    'stage2_model': 'internvl2.5',
                    'status': 'success',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(episode_start_time))
                }
                time_records.append(time_record)
                
                # 保存两阶段的结果
                result = {
                    'id': frame_path,
                    'prompt_version': prompt_version,
                    'segmentation_response': seg_results[0]['response'],
                    'detailed_response': detail_results['response'],
                    'segments': segments,
                    'task_summary': task_summary,
                    'version_info': version_info,
                    'model_info': {
                        'stage1_model': 'gemini',
                        'stage2_model': 'internvl2.5'
                    }
                }
                all_results.append(result)
                
                # 打印时间统计
                print(f"Successfully processed {frame_path}")
                print(f"  Stage 1 (Gemini): {stage1_duration:.2f}s")
                print(f"  Stage 2 (InternVL2.5): {stage2_duration:.2f}s")
                print(f"  Total episode time: {episode_duration:.2f}s")
            else:
                episode_end_time = time.time()
                episode_duration = episode_end_time - episode_start_time
                
                # 记录时间信息（失败情况）
                time_record = {
                    'episode_id': frame_path,
                    'episode_name': episode_name,
                    'stage1_duration': stage1_duration,
                    'stage2_duration': None,
                    'total_episode_duration': episode_duration,
                    'stage1_model': 'gemini',
                    'stage2_model': 'internvl2.5',
                    'status': 'failed_stage2',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(episode_start_time))
                }
                time_records.append(time_record)
                
                print(f"Failed to get detailed analysis for {frame_path}")
                print(f"  Stage 1 (Gemini): {stage1_duration:.2f}s")
                print(f"  Total episode time: {episode_duration:.2f}s")
                
        except Exception as e:
            episode_end_time = time.time()
            episode_duration = episode_end_time - episode_start_time
            
            # 记录时间信息（异常情况）
            time_record = {
                'episode_id': frame_path,
                'episode_name': episode_name,
                'stage1_duration': stage1_duration if 'stage1_duration' in locals() else None,
                'stage2_duration': None,
                'total_episode_duration': episode_duration,
                'stage1_model': 'gemini',
                'stage2_model': 'internvl2.5',
                'status': 'error',
                'error_message': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(episode_start_time))
            }
            time_records.append(time_record)
            
            print(f"Error processing {frame_path}: {e}")
            print(f"  Episode time: {episode_duration:.2f}s")
            continue
    
    # 计算总体运行时间
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # 创建总体时间统计
    overall_stats = {
        'total_episodes': len(frame_paths),
        'successful_episodes': len(all_results),
        'failed_episodes': len(frame_paths) - len(all_results),
        'total_processing_time': total_duration,
        'average_time_per_episode': total_duration / len(frame_paths) if len(frame_paths) > 0 else 0,
        'start_timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time)),
        'end_timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_end_time)),
        'prompt_version': prompt_version,
        'pipeline_type': 'hybrid_gemini_internvl'
    }
    
    print(f"\n=== Processing Summary ===")
    print(f"Total episodes processed: {len(all_results)}")
    print(f"Total processing time: {total_duration:.2f}s")
    if len(all_results) > 0:
        print(f"Average time per episode: {total_duration/len(frame_paths):.2f}s")
    
    # 保存第一阶段结果到单独文件
    if seg_result_file and seg_results_list:
        os.makedirs(os.path.dirname(seg_result_file), exist_ok=True)
        with open(seg_result_file, 'w', encoding='utf-8') as f:
            json.dump(seg_results_list, f, ensure_ascii=False, indent=2)
        print(f"Segmentation results saved to {seg_result_file}")
    
    # 保存时间记录到单独文件
    if result_file and time_records:
        time_report_file = result_file.replace('.json', '_time_report.json')
        os.makedirs(os.path.dirname(time_report_file), exist_ok=True)
        
        time_report = {
            'overall_statistics': overall_stats,
            'episode_details': time_records
        }
        
        with open(time_report_file, 'w', encoding='utf-8') as f:
            json.dump(time_report, f, ensure_ascii=False, indent=2)
        print(f"Time report saved to {time_report_file}")
    
    # 清理模型资源
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Hybrid Two-stage RTX planning pipeline (Gemini + InternVL2.5)")
    parser.add_argument('--root', type=str, required=True, help="视频抽帧文件路径")
    parser.add_argument('--embodiment', type=str, default="franka-dual_arm")
    parser.add_argument('--json', type=str, required=True, help="描述文件路径")
    parser.add_argument('--output', type=str, required=True, help="结果文件路径")
    parser.add_argument('--prompt-version', type=str, default="v1", 
                       choices=get_available_versions(),
                       help="Prompt版本 (v1: task_summary+steps, v2: segments_only)")
    parser.add_argument('--internvl-model-path', type=str, 
                       default="/media/users/wd/hf/hf_models/InternVL3-78B",
                       help="InternVL2.5模型路径")
    parser.add_argument('--verbose', action='store_true', help="是否调试，调试阶段不调用api")
    parser.add_argument('--rerun', action='store_true', help="是否重跑，默认加载已有结果")
    parser.add_argument('--max', type=int, default=0, help="最大请求数，默认是0，代表不限制")
    parser.add_argument('--include-step-descriptions', action='store_true', default=True, 
                       help="是否在第二阶段包含第一阶段的step descriptions（默认包含）")
    parser.add_argument('--exclude-step-descriptions', action='store_true', 
                       help="不在第二阶段包含第一阶段的step descriptions")
    args = parser.parse_args()

    # 记录main函数开始时间
    main_start_time = time.time()
    
    # 处理step descriptions包含选项
    include_step_descriptions = True  # 默认包含
    if args.exclude_step_descriptions:
        include_step_descriptions = False
    
    root_dir = args.root
    desc_file = args.json
    result_file = args.output
    prompt_version = args.prompt_version

    desc = parse_rtx_desc(desc_file)
    frame_paths = glob.glob(f"{root_dir}/*episode_*")
    # choose only the frames with description
    frame_paths = [p for p in frame_paths if os.path.basename(p) in desc]
    
    print(f"=== Hybrid Pipeline Setup ===")
    print(f"Available prompt versions: {get_available_versions()}")
    print(f"Using prompt version: {prompt_version}")
    print(f"Include step descriptions in stage 2: {include_step_descriptions}")
    print(f"InternVL model path: {args.internvl_model_path}")
    print()
    
    print(frame_paths)
    print('Total:', len(frame_paths))

    # 处理断点续传
    has_results = []
    if not args.rerun:
        try:
            has_results = [res for res in json.load(open(result_file)) if res]
        except:
            has_results = []

    exists = [res['id'] for res in has_results]
    frame_paths = [p for p in frame_paths if p not in exists]
    print('Remain:', len(frame_paths))
    
    frame_paths = sorted(frame_paths)

    if args.max > 0:
        frame_paths = frame_paths[:args.max]
        print('Processing paths:', len(frame_paths))
    
    if not args.verbose and frame_paths:
        # 生成第一阶段结果文件路径
        seg_result_file = result_file.replace('.json', '_segmentation.json')
        
        # 运行混合两阶段管道
        results = asyncio.run(process_hybrid_two_stage_pipeline(
            frame_paths, desc, args.embodiment, prompt_version, result_file, seg_result_file, 
            include_step_descriptions, args.internvl_model_path))
        print(f"Processed {len(results)} episodes")
        
        # 打印Gemini token使用统计（仅第一阶段）
        print_token_usage_summary()
        
        # 保存token使用报告
        token_report_file = result_file.replace('.json', '_token_usage.json')
        os.makedirs(os.path.dirname(token_report_file), exist_ok=True)
        save_token_usage_report(token_report_file)
        
        # 显示成本估算
        get_estimated_cost()
    else:
        results = []

    # 合并结果
    all_results = has_results + results
    
    # 保存结果
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    json.dump(all_results, open(result_file, 'w'), ensure_ascii=False, indent=2)
    print(f"Results saved to {result_file}")
    
    # 计算并显示总体运行时间
    main_end_time = time.time()
    main_duration = main_end_time - main_start_time
    print(f"\n=== Overall Summary ===")
    print(f"Total main execution time: {main_duration:.2f}s ({main_duration/60:.1f} minutes)")

if __name__ == "__main__":
    main()
