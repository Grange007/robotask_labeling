from prompt_utils import *
from api_utils import *
import glob
import argparse
import json
import re
import asyncio
from prompt_manager import create_prompt_manager, get_available_versions

def format_segmentation_message_universal(path_to_frames, prompt_manager, embodiment=None, desc=None, **kwargs):
    """通用的第一阶段消息格式化"""
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

def format_detailed_analysis_message_universal(path_to_frames, segments, prompt_manager, task_summary="", embodiment=None, desc=None, **kwargs):
    """通用的第二阶段消息格式化"""
    images = os.listdir(path_to_frames)
    images = sorted(images, key=lambda x: int(x[x.index('_')+1:x.index('.png')]))
    images = [os.path.join(path_to_frames, img) for img in images]
    
    if not desc:
        desc = ""
    
    # 获取基础prompt
    base_prompt = prompt_manager.get_detailed_prompt()
    
    # 获取格式化的上下文
    context = prompt_manager.format_detailed_prompt_context(segments, task_summary, desc)
    
    # 组合完整prompt
    full_prompt = f"{base_prompt}{context}"
    
    message = {
        "request_id": f"{path_to_frames}_detailed", 
        "prompt": full_prompt, 
        "image_base64": [image_to_frame(img) for img in images]
    } 
    return message

async def process_universal_two_stage_pipeline(frame_paths, desc, embodiment, prompt_version="v1", result_file=None):
    """通用的两阶段处理管道"""
    prompt_manager = create_prompt_manager(prompt_version)
    version_info = prompt_manager.get_version_info()
    
    print(f"Using prompt version: {version_info['version']}")
    print(f"Output format: {version_info['output_format']}")
    print(f"Has task summary: {version_info['has_task_summary']}")
    print(f"Has step descriptions: {version_info['has_step_descriptions']}")
    print()
    
    all_results = []
    
    for i, frame_path in enumerate(frame_paths):
        print(f'=== Processing {i+1}/{len(frame_paths)}: {frame_path} ===')
        
        episode_name = os.path.basename(frame_path)
        episode_desc = desc.get(episode_name, {}).get('desc', '')
        
        try:
            # 第一阶段：分割
            print("Stage 1: Segmentation...")
            seg_message = format_segmentation_message_universal(frame_path, prompt_manager, embodiment, episode_desc)

            # prompt_store_path = os.path.join(os.path.pardir(result_file), f"{episode_name}_prompt.txt")
            # print(f"Storing prompt to {prompt_store_path}")
            # print(f"Prompt:\n{seg_message['prompt']}\n")
            # os.makedirs(os.path.dirname(prompt_store_path), exist_ok=True)
            # with open(prompt_store_path, 'w', encoding='utf-8') as f:
            #     f.write(seg_message['prompt'])

            seg_results = await request_model([seg_message])
            
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
            
            # 第二阶段：详细分析
            print("Stage 2: Detailed analysis...")
            detail_message = format_detailed_analysis_message_universal(
                frame_path, segments, prompt_manager, task_summary, embodiment, episode_desc
            )
            detail_results = await request_model([detail_message])
            
            if detail_results and detail_results[0].get('response'):
                # 保存两阶段的结果
                result = {
                    'id': frame_path,
                    'prompt_version': prompt_version,
                    'segmentation_response': seg_results[0]['response'],
                    'detailed_response': detail_results[0]['response'],
                    'segments': segments,
                    'task_summary': task_summary,
                    'version_info': version_info
                }
                all_results.append(result)
                print(f"Successfully processed {frame_path}")
            else:
                print(f"Failed to get detailed analysis for {frame_path}")
                
        except Exception as e:
            print(f"Error processing {frame_path}: {e}")
            continue
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Universal Two-stage RTX planning pipeline")
    parser.add_argument('--root', type=str, required=True, help="视频抽帧文件路径")
    parser.add_argument('--embodiment', type=str, default="franka-dual_arm")
    parser.add_argument('--json', type=str, required=True, help="描述文件路径")
    parser.add_argument('--output', type=str, required=True, help="结果文件路径")
    parser.add_argument('--prompt-version', type=str, default="v1", 
                       choices=get_available_versions(),
                       help="Prompt版本 (v1: task_summary+steps, v2: segments_only)")
    parser.add_argument('--verbose', action='store_true', help="是否调试，调试阶段不调用api")
    parser.add_argument('--rerun', action='store_true', help="是否重跑，默认加载已有结果")
    parser.add_argument('--max', type=int, default=0, help="最大请求数，默认是0，代表不限制")
    args = parser.parse_args()

    root_dir = args.root
    desc_file = args.json
    result_file = args.output
    prompt_version = args.prompt_version

    desc = parse_rtx_desc(desc_file)
    frame_paths = glob.glob(f"{root_dir}/*episode_*")
    
    # choose only the frames with description
    frame_paths = [p for p in frame_paths if os.path.basename(p) in desc]
    
    print(f"Available prompt versions: {get_available_versions()}")
    print(f"Using prompt version: {prompt_version}")
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
        # 运行通用两阶段管道
        results = asyncio.run(process_universal_two_stage_pipeline(frame_paths, desc, args.embodiment, prompt_version, result_file))
        print(f"Processed {len(results)} episodes")
        
        # 打印token使用统计
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
    json.dump(all_results, open(result_file, 'w'), ensure_ascii=False, indent=2)
    print(f"Results saved to {result_file}")

if __name__ == "__main__":
    main()
