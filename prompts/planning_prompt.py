PLANNING_TEMPLATE ="""You will analyze a video (represented by image frames) of a robotic arm performing a specific task, where the task is described as: ``{desc}``. Note that the referenced task summary might not accurate or complete. Your task is to identify the primary task during the video with the help of the referenced descrition, summarize the task and rewrite the description, extract the necessary steps to complete it, and specify the frame range for each step. Follow these instructions:

1. **Task Identification**: First, identify the main task the robotic arm is performing. This task could be a clear goal or a series of related activities (e.g., assembling furniture, repairing equipment, preparing food, etc.). Briefly describe the primary task in one sentence.

2. **Step Extraction**: Once the task is identified, extract the key steps required to complete it, ensuring that each step is clearly described and logically ordered. Each step may include:
    - Specific actions (e.g., tightening screws, stirring mixtures, pressing buttons, etc.)
    - Frame window: Specify the start and end frame for each step (from `0` to `29`, since the video has 30 frames).

3. **Output Format**: Provide the task description and steps in two parts, formatted as JSON:
    - **Task Summary**: A string summarizing the primary task in the video without mentioning the subjects - the robotic arm.
    - **Steps**: An array where each element represents a step, containing:
        - `step_description`: A concise description of the step which the action being performed in the format of verb phrases without mentioning the subjects - the robotic arm (e.g., "Add syrup in the glass").
        - `start_frame`: The start frame of the step (from `0` to `29`).
        - `end_frame`: The end frame of the step (from `0` to `29`).

**Task Description**: {desc}
---

**Example Output Format:**
```json
{{
  "task_summary": "Assembling an office desk.",
  "steps": [
    {{
      "step_description": "Remove all components and screws from the package.",
      "start_frame": 0,
      "end_frame": 4
    }},
    {{
      "step_description": "Use a screwdriver to attach the legs to the tabletop.",
      "start_frame": 5,
      "end_frame": 14
    }},
    {{
      "step_description": "Install the leg pads at the bottom.",
      "start_frame": 15,
      "end_frame": 19
    }},
    {{
      "step_description": "Fix the support beam between the legs with screws.",
      "start_frame": 20,
      "end_frame": 28
    }},
    {{
      "step_description": "Ensure all screws are tight and the desk is stable.",
      "start_frame": 29,
      "end_frame": 29
    }}
  ]
}}"""
