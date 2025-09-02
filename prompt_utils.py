import json
from cv_utils import image_to_frame
import os

def format_planning_api_message(path_to_frames, embodiment=None, desc=None, **kwargs):
	images = os.listdir(path_to_frames)
	images = sorted(images, key=lambda x: int(x[x.index('_')+1:x.index('.png')]))
	images = [os.path.join(path_to_frames, img) for img in images]
	if not desc:
		desc = ""
	# print(desc)
	if embodiment == 'table_action_single_arm':
		from planning_prompt import PLANNING_TEMPLATE
	else:
		raise NotImplementedError(f"Embodiment {embodiment} not implemented")
	prompt = PLANNING_TEMPLATE.format(desc=desc)
	message = {"request_id": path_to_frames, "prompt": prompt, "image_base64": [image_to_frame(img) for img in images]} 
	return message

def format_two_stage_segmentation_message(path_to_frames, embodiment=None, desc=None, **kwargs):
	"""格式化两阶段处理的第一阶段消息（分割）"""
	images = os.listdir(path_to_frames)
	images = sorted(images, key=lambda x: int(x[x.index('_')+1:x.index('.png')]))
	images = [os.path.join(path_to_frames, img) for img in images]
	if not desc:
		desc = ""
	
	if embodiment == 'table_action_single_arm':
		from code.planning_prompt_1 import SEGMENTATION_PROMPT
		prompt = SEGMENTATION_PROMPT
	else:
		raise NotImplementedError(f"Embodiment {embodiment} not implemented for two-stage processing")
	
	message = {"request_id": f"{path_to_frames}_segmentation", "prompt": prompt, "image_base64": [image_to_frame(img) for img in images]} 
	return message

def format_two_stage_detailed_message(path_to_frames, segments, embodiment=None, desc=None, **kwargs):
	"""格式化两阶段处理的第二阶段消息（详细分析）"""
	images = os.listdir(path_to_frames)
	images = sorted(images, key=lambda x: int(x[x.index('_')+1:x.index('.png')]))
	images = [os.path.join(path_to_frames, img) for img in images]
	if not desc:
		desc = ""
	
	if embodiment == 'table_action_single_arm':
		from code.planning_prompt_1 import DETAILED_ANALYSIS_PROMPT
		# 构建包含分割信息的prompt
		segments_text = json.dumps(segments, indent=2)
		prompt = f"{DETAILED_ANALYSIS_PROMPT}\n\nPre-segmented frame list:\n{segments_text}\n\nTask Description: {desc}"
	else:
		raise NotImplementedError(f"Embodiment {embodiment} not implemented for two-stage processing")
	
	message = {"request_id": f"{path_to_frames}_detailed", "prompt": prompt, "image_base64": [image_to_frame(img) for img in images]} 
	return message

def format_api_message(path_to_frames, base_prompt, **kwargs):
	images = os.listdir(path_to_frames)
	images = sorted(images, key=lambda x: int(x[x.index('_')+1:x.index('.png')]))
	images = [os.path.join(path_to_frames, img) for img in images]
	try:
		prompt = base_prompt.format(**kwargs)
	except:
		raise AssertionError

	message = {"request_id": path_to_frames, "prompt": prompt, "image_base64": [image_to_frame(img) for img in images]} 
	return message

def parse_rtx_desc(path_to_desc):
	res = {}
	desc = json.load(open(path_to_desc))
	for video_id, desc_text in desc.items():
		res[video_id] = {'desc': desc_text}
	return res

def format_api_message_vl(request_id, prompt, image_path):
	message = {
		"request_id": request_id, 
		"prompt": prompt,
		"image_base64": [image_to_frame(image_path)],
	}
	return message
