SEGMENTATION_PROMPT ='''
You will analyze a video (represented by image frames) of a robotic arm performing a specific task, where the task is described as: ``{desc}``. Note that the referenced task summary might not accurate or complete. Your task is to identify the primary task during the video with the help of the referenced descrition, summarize the task and rewrite the description, extract the necessary steps to complete it, and specify the frame range for each step. Follow these instructions:

1.  **Task Identification & Multi-Arm Handling**: First, identify the main task the robotic arm is performing. If multiple robotic arms (e.g., left arm, right arm) are present and operating **independently or collaboratively**, you MUST identify and segment the steps for **each arm separately**. The steps for different arms can have **overlapping frame ranges**. Briefly describe the primary task, noting if it involves multiple arms.

2. **Step Extraction**: Once the task is identified, extract the key steps required to complete it, ensuring that each step is clearly described and logically ordered. Each step may include:
    - Specific actions (e.g., tightening screws, stirring mixtures, pressing buttons, etc.)
    - Frame window: Specify the start and end frame for each step
    - **For multiple arms**: Segment the actions for each arm into their own logical steps. The timeline for Arm A and Arm B's steps can overlap. In the `step_description`, explicitly mention which arm (e.g., "Left arm picks...", "Right arm places...") is performing the action if multiple arms are present.

3.  **Output Format**: Provide the task description and steps in two parts, formatted as JSON:
    -   **Task Summary**: A string summarizing the primary task in the video, including mention of multiple arms if applicable.
    -   **Steps**: An array where each element represents a step, containing:
        -   `step_description`: A concise description of the step which the action being performed in the format of verb phrases. **If multiple arms are present, prefix the description with the arm identifier (e.g., 'Left arm: ', 'Right arm: ')**.
        -   `start_frame`: The start frame of the step.
        -   `end_frame`: The end frame of the step.
        -   `arm_id` (Optional but Recommended): An identifier for the arm performing the step (e.g., "left", "right", "arm_a", "arm_b"). This is crucial for disambiguating parallel actions.

4. **Skill Library (Action Verbs)**
  You MUST select the verb from the 'Skill Name (English)' column of the table below. Do not use any other verbs.

  | Skill Name (English) |
  | :------------------- |
  | Go                   |
  | Turn                 |
  | Lower                |
  | Lift                 |
  | Pick                 |
  | Place                |
  | Push                 |
  | Pull                 |
  | Insert               |
  | Divide               |
  | Open                 |
  | Close                |
  | Turn on              |
  | Turn off             |
  | Rotate               |
  | Flip                 |
  | Fold                 |
  | Unfold               |
  | Unwind               |
  | Drop                 |
  | Flatten              |
  | Shake                |
  | Stretch              |
  | Straighten           |
  | Press                |
  | Scan                 |
  | Swipe                |
  | Stick                |
  | Screw                |
  | Unscrew              |
  | Drill                |
  | Spread               |
  | Wipe                 |
  | Sweep                |
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
  
  # delete words not above
  
  Instruction Templates
  Template 1 (Robot Movement - Navigation): For moving the robot base. Don't use this template for manipulating objects or for robots without a mobile base. Don't use this template when you can only see robotic arms.
  Verbs: Go, Turn.

  Template 2 (Robot Movement - Vertical): For moving the robot base vertically without manipulating an object. Don't use this template for manipulating objects or for robots without a vertical base movement capability. Don't use this template when you can only see robotic arms.
  Verbs: Lower, Lift.

  Template 3 (Object Movement Actions): For actions involving clear displacement of an object from a start point to an end point.
  Verbs: Place, Push, Pull, Insert, Throw, Pour, Hand to.

  Template 4 (In-Place & State-Change Actions): For actions performed on an object at a single, specific location.
  Verbs: Pick, Drop, Open, Close, Turn on, Turn off, Rotate, Flip, Press, Screw, Unscrew, Squeeze, Twist, Hang, Fold, Unfold, Tie, Pluck, Knock, Beat, Flatten, Shake, Stretch, Straighten, Wave, Clap, Point, Divide.

  Template 5 (Surface/Tool Actions): For actions performed upon a surface or with a tool.
  Verbs: Scan, Swipe, Stick, Drill, Spread, Wipe, Sweep, Stir, Peel, Suction, Scoop, Scratch.


**Task Description**: {desc}
---

**Example Output Format:**
```json
{{
  "task_summary": "Assembling an office desk.",
  "steps": [
    {{
      "step_description": "Pick all components and screws from the package.",
      "start_frame": 0,
      "end_frame": 4
    }},
    {{
      "step_description": "Twist a screwdriver to attach the legs to the tabletop.",
      "start_frame": 5,
      "end_frame": 14
    }},
    {{
      "step_description": "Place the leg pads at the bottom.",
      "start_frame": 15,
      "end_frame": 19
    }},
    {{
      "step_description": "Insert the support beam between the legs.",
      "start_frame": 20,
      "end_frame": 28
    }},
    {{
      "step_description": "Tighten all screws.",
      "start_frame": 29,
      "end_frame": 42
    }}
  ]
}}
'''

DETAILED_ANALYSIS_PROMPT ='''
Role and Goal
You are an expert system for robotic motion analysis, specializing in understanding and annotating robot manipulation tasks from video frames. Your primary goal is to meticulously analyze a provided sequence of chronologically ordered frames and a pre-segmented list of sub-action frames to produce a structured JSON annotation. You must describe each pre-defined segment with high fidelity.

Core Task
 * Analyze Video Frames: Carefully examine the sequence of frames, focusing on the actions within each pre-defined segment. **Pay close attention to which robotic arm (e.g., left, right) is performing the action in each segment.**
 * Generate Structured Annotations: For each segment provided by the frame list, fill out a JSON object that details the action, manipulated object, and spatial information. The verb for the action must be selected from the Skill Library provided. **If the segment involves a specific arm, ensure the `arm_id` from the segmentation phase is considered and used to resolve ambiguities in object/location.**
 * Create Natural Language Instructions: Generate a concise instruction for each sub-action using the appropriate Instruction Template.
 * Strictly Adhere to Output Format: The final output must be a single, valid JSON object containing a list of all identified segments.
Skill Library (Action Verbs)
You MUST select the verb from the 'Skill Name (English)' column of the table below. Do not use any other verbs.

| Skill Name (English) |
| :------------------- |
| Go                   |
| Turn                 |
| Lower                |
| Lift                 |
| Pick                 |
| Place                |
| Push                 |
| Pull                 |
| Insert               |
| Divide               |
| Open                 |
| Close                |
| Turn on              |
| Turn off             |
| Rotate               |
| Flip                 |
| Fold                 |
| Unfold               |
| Unwind               |
| Drop                 |
| Flatten              |
| Shake                |
| Stretch              |
| Straighten           |
| Press                |
| Scan                 |
| Swipe                |
| Stick                |
| Screw                |
| Unscrew              |
| Drill                |
| Spread               |
| Wipe                 |
| Sweep                |
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
Template 1 (Robot Movement - Navigation): For moving the robot base. Don't use this template for manipulating objects or for robots without a mobile base. Don't use this template when you can only see robotic arms.
Verbs: Go, Turn.
Format for 'Go': "Go from [start_location] to [end_location]"
Format for 'Turn': "Turn [direction: left/right] at [location]"

Template 2 (Robot Movement - Vertical): For moving the robot base vertically without manipulating an object. Don't use this template for manipulating objects or for robots without a vertical base movement capability. Don't use this template when you can only see robotic arms.
Verbs: Lower, Lift.
Format: "[verb] the base at [location]"

Template 3 (Object Movement Actions): For actions involving clear displacement of an object from a start point to an end point.
Verbs: Place, Push, Pull, Drag, Insert, Throw, Pour, Hand to.
Format: "[verb] the [object] from [start_location] to [end_location]"

Template 4 (In-Place & State-Change Actions): For actions performed on an object at a single, specific location.
Verbs: Pick, Drop, Open, Close, Turn on, Turn off, Rotate, Turn, Flip, Press, Screw, Unscrew, Squeeze, Twist, Hang, Fold, Unfold, Tie, Pluck, Knock, Beat, Flatten, Shake, Stretch, Straighten, Wave, Clap, Point, Divide.
Format: "[verb] the [object] at [location]"
Note: Use start_location for actions like Pick and end_location for actions like Drop. For most others, start_location and end_location are the same; use this location in the template.

Template 5 (Surface/Tool Actions): For actions performed upon a surface or with a tool.
Verbs: Scan, Swipe, Stick, Drill, Spread, Wipe, Sweep, Stir, Peel, Suction, Scoop, Scratch.
Format: "[verb] the [object] on/at [location]"
Note: Use the location where the action is being performed (typically end_location).

Output Format (JSON Only)
{
  "segments_of_sub_actions": [
    {
      "start_frame": "Start frame index from original video (integer)",
      "end_frame": "End frame index from original video (integer)",
      "validity": "",
      "arm_id": "Identifier for the arm performing this action (e.g., 'left', 'right', 'single'). Crucial for disambiguating parallel actions.", // NEW FIELD ADDED
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
 * **Parallel Arm Handling:**
   * **Segments from different arms CAN have overlapping `start_frame` and `end_frame` values.** This represents parallel execution.
   * The `arm_id` field MUST be used to distinguish between segments executed concurrently by different arms.
 * Spatial Awareness (start_location / end_location):
   * BE A SPATIAL DETECTIVE. You must precisely describe locations. **When multiple arms are present, your description should also help distinguish the workspace of each arm (e.g., 'left side of the jig', 'right component tray').**
   * Use Landmarks: Describe locations relative to fixed, visible landmarks in the environment (e.g., "blue component tray", "left side of the main assembly jig", "slot 3B of the storage rack", "inside the CNC machine").
   * Avoid Vague Terms: Do not use ambiguous terms like "on the table", "near the robot", or "in the area" if a more specific description is possible. Quantify if you can (e.g., "top shelf" vs. "shelf").
   * Movement vs. In-Place: For movement actions, start_location and end_location must be different. For in-place actions, they should be the same and describe the location of the action.
Critical Constraints
 * Analyze ONLY the provided frames and the pre-segmented frame list. Do not hallucinate actions or objects not visible.
 * The verb MUST be a valid skill from the 'Skill Name (English)' column in the Skill Library.
 * Locations MUST be as specific and descriptive as possible, based on visual evidence.
 * sub_action_instruction MUST be generated using the specified templates.
 * The final output MUST be a single valid JSON object.
 * **You MUST include the `arm_id` field in each segment to identify the acting arm, especially when multiple arms are present.**
'''