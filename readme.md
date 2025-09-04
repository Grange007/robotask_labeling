# Prompt for task labeling

Organize the files in this way:

```
.
├── robotask_labeling
│   ├── prompts
│   ├── 1_pipeline_universal_two_stage.py
│   ├── 2_json_reformat_two_stage.py
│   └── ...
├── demo_frames_mine
│   ├── franka_fr3_dualArm-gripper-6cameras_2_find_out_packaging_tape_into_the_other_basket_20250507_observation.rgb_images.camera_top_episode_000000.mp4_sampled_rank0
│   ├── ur_03_flip_the_cup_upright_observation.rgb_images.camera_top_episode_000000.mp4_sampled_rank0
│   └── ...
└── demo_description
    └── my_desc.json
```

All the prompts are in the `code/prompts` folder. You can modify them as needed.

To choose a different prompt, in `prompt_manager.py`:

change the `from prompts.planning_prompt_1 import ...` line to the desired prompt version, e.g.,  `from prompts.planning_prompt_1_pick_and_place import ...`

```python
def _load_prompts(self):
    if self.prompt_version == "v1":
        from prompts.planning_prompt_1 import SEGMENTATION_PROMPT, DETAILED_ANALYSIS_PROMPT
        self.segmentation_prompt = SEGMENTATION_PROMPT
        self.detailed_prompt = DETAILED_ANALYSIS_PROMPT
        self.output_format = "task_summary_steps"
    elif self.prompt_version == "v2":
        from prompts.planning_prompt_2 import SEGMENTATION_PROMPT, DETAILED_ANALYSIS_PROMPT
        self.segmentation_prompt = SEGMENTATION_PROMPT
        self.detailed_prompt = DETAILED_ANALYSIS_PROMPT
        self.output_format = "segments_only"
```

To run the two stage labeling pipeline, use the following command:

For Linux:

```bash
./run_universal_two_stage.sh
```

For Windows (PowerShell):

```powershell
.\run_universal_two_stage.ps1
```

To prohibit sending the step_description from stage 1 to stage 2, set `exclude_step_descriptions` to `true` in scripts before running the pipeline.
