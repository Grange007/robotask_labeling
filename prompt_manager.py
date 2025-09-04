# 统一的两阶段prompt管理系统
# 支持多种prompt格式的兼容性

import json
import re

class PromptManager:
    """管理不同版本的prompt并提供统一接口"""
    
    def __init__(self, prompt_version="v1"):
        self.prompt_version = prompt_version
        self._load_prompts()
    
    def _load_prompts(self):
        """根据版本加载对应的prompt"""
        if self.prompt_version == "v1":
            from prompts.planning_prompt_1_pick_and_place import SEGMENTATION_PROMPT, DETAILED_ANALYSIS_PROMPT
            self.segmentation_prompt = SEGMENTATION_PROMPT
            self.detailed_prompt = DETAILED_ANALYSIS_PROMPT
            self.output_format = "task_summary_steps"
        elif self.prompt_version == "v2":
            from prompts.planning_prompt_2 import SEGMENTATION_PROMPT, DETAILED_ANALYSIS_PROMPT
            self.segmentation_prompt = SEGMENTATION_PROMPT
            self.detailed_prompt = DETAILED_ANALYSIS_PROMPT
            self.output_format = "segments_only"
        else:
            raise ValueError(f"Unsupported prompt version: {self.prompt_version}")
    
    def format_segmentation_prompt(self, desc=""):
        """格式化第一阶段prompt"""
        if self.prompt_version == "v1":
            # v1需要填充desc参数
            return self.segmentation_prompt.format(desc=desc)
        else:
            # v2不需要desc参数
            return self.segmentation_prompt
    
    def parse_segmentation_result(self, response_text):
        """解析第一阶段结果，统一输出格式"""
        try:
            # 清理response中的markdown标记
            cleaned_text = re.sub(r'^```json|```$', '', response_text.strip(), flags=re.MULTILINE)
            segments_data = json.loads(cleaned_text)
            
            if self.output_format == "task_summary_steps":
                # v1格式：包含task_summary和steps
                steps = segments_data.get("steps", [])
                segments = []
                for step in steps:
                    segments.append({
                        "start_frame": step.get("start_frame"),
                        "end_frame": step.get("end_frame"),
                        "step_description": step.get("step_description", "")
                    })
                return segments, segments_data.get("task_summary", "")
                
            elif self.output_format == "segments_only":
                # v2格式：只包含segments_of_sub_actions
                segments_list = segments_data.get("segments_of_sub_actions", [])
                segments = []
                for segment in segments_list:
                    segments.append({
                        "start_frame": segment.get("start_frame"),
                        "end_frame": segment.get("end_frame"),
                        "step_description": ""  # v2不包含描述
                    })
                return segments, ""  # v2没有task_summary
                
        except json.JSONDecodeError as e:
            print(f"Error parsing segmentation result: {e}")
            return [], ""
    
    def format_detailed_prompt_context(self, segments, task_summary="", desc=""):
        """格式化第二阶段prompt的上下文信息"""
        # 将segments转换为第二阶段期望的格式
        segments_for_second_stage = []
        for segment in segments:
            segments_for_second_stage.append({
                "start_frame": segment["start_frame"],
                "end_frame": segment["end_frame"]            })
        
        segments_text = json.dumps({"segments_of_sub_actions": segments_for_second_stage}, indent=2)
        
        context = f"""
Pre-segmented frame list:
{segments_text}

Task Description: {desc}"""
        
        if task_summary:
            context += f"\nTask Summary from Stage 1: {task_summary}"
        
        if self.prompt_version == "v1":
            # v1有step描述，添加额外上下文
            context += "\n\nAdditional context from Stage 1 steps:"
            for i, segment in enumerate(segments):
                if segment.get("step_description"):
                    context += f"\nStep {i+1} (frames {segment['start_frame']}-{segment['end_frame']}): {segment['step_description']}"
        
        return context
    
    def get_detailed_prompt(self):
        """获取第二阶段prompt"""
        return self.detailed_prompt
    
    def get_version_info(self):
        """获取版本信息"""
        return {
            "version": self.prompt_version,
            "output_format": self.output_format,
            "has_task_summary": self.output_format == "task_summary_steps",
            "has_step_descriptions": self.output_format == "task_summary_steps"
        }

# 便捷函数
def create_prompt_manager(version="v1"):
    """创建prompt管理器"""
    return PromptManager(version)

def get_available_versions():
    """获取可用的prompt版本"""
    return ["v1", "v2"]
