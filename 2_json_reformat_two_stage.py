import json
import re
import os
import argparse

def clean_and_convert_response(response_text):
    """清理和转换response中的文本为正式JSON"""
    if not response_text:
        return None
        
    # 使用正则表达式去除 ```json 和 ``` 包裹的部分
    cleaned_text = re.sub(r'^```json|```$', '', response_text.strip(), flags=re.MULTILINE)
    try:
        # 将清理后的文本转换为 JSON 对象
        json_data = json.loads(cleaned_text)
        return json_data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Problematic text: {cleaned_text[:200]}...")
        return None

def process_two_stage_results(input_file, output_file):
    """处理两阶段结果文件"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    reformatted_results = []
    
    for item in data:
        if not isinstance(item, dict):
            continue
            
        result = {
            'id': item.get('id', ''),
            'episode_name': os.path.basename(item.get('id', '')),
            'task_summary': item.get('task_summary', ''),  # 添加第一阶段的任务总结
        }
        
        # 处理第一阶段结果（分割）
        seg_response = item.get('segmentation_response', '')
        if seg_response:
            seg_data = clean_and_convert_response(seg_response)
            if seg_data:
                result['segmentation'] = seg_data
            else:
                result['segmentation'] = {'error': 'Failed to parse segmentation response'}
        
        # 处理第二阶段结果（详细分析）
        detail_response = item.get('detailed_response', '')
        if detail_response:
            detail_data = clean_and_convert_response(detail_response)
            if detail_data:
                result['detailed_analysis'] = detail_data
            else:
                result['detailed_analysis'] = {'error': 'Failed to parse detailed analysis response'}
        
        # 保留原始segments信息
        if 'segments' in item:
            result['parsed_segments'] = item['segments']
        
        # 保留原始responses用于调试
        result['raw_responses'] = {
            'segmentation': seg_response,
            'detailed_analysis': detail_response
        }
        
        reformatted_results.append(result)

    # 保存格式化结果
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(reformatted_results, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved reformatted results to {output_file}")
        
        # 打印统计信息
        total_episodes = len(reformatted_results)
        successful_seg = sum(1 for r in reformatted_results if 'segmentation' in r and 'error' not in r['segmentation'])
        successful_detail = sum(1 for r in reformatted_results if 'detailed_analysis' in r and 'error' not in r['detailed_analysis'])
        
        print(f"Statistics:")
        print(f"  Total episodes: {total_episodes}")
        print(f"  Successful segmentations: {successful_seg}")
        print(f"  Successful detailed analyses: {successful_detail}")
        
    except Exception as e:
        print(f"Error saving output file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Reformat two-stage pipeline results")
    parser.add_argument('--input-json', type=str, required=True, help="输入的JSON文件路径")
    parser.add_argument('--output-json', type=str, help="输出的JSON文件路径")
    args = parser.parse_args()
    
    input_file = args.input_json
    
    # 如果没有指定输出文件，自动生成
    if args.output_json:
        output_file = args.output_json
    else:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_reformat.json"
    
    print(f"Processing {input_file} -> {output_file}")
    process_two_stage_results(input_file, output_file)

if __name__ == "__main__":
    main()
