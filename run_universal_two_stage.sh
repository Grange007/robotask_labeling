#!/bin/bash

# Universal Two-Stage RTX Planning Pipeline - Supports multiple prompt versions

# --- Configuration ---
embodiment="table_action_single_arm"
timestamp=$(date +"%Y%m%d_%H%M%S")
output_name="gemini_2_5_output_universal_no_desc_$timestamp"
prompt_version="v2"  # Options: v1 or v2, v1 for example_segment + our_detect, v2 for our_segment+our_detect
exclude_step_descriptions=true  # Set to true to exclude step descriptions from stage 1 in stage 2

# --- ANSI Color Codes ---
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
RED='\033[0;31m'
WHITE='\033[0;37m'
NC='\033[0m' # No Color

# --- Script Start ---
echo -e "${GREEN}Starting Universal Two-Stage RTX Planning Pipeline...${NC}"
echo -e "${YELLOW}Embodiment: $embodiment${NC}"
echo -e "${YELLOW}Prompt Version: $prompt_version${NC}"
echo -e "${YELLOW}Output name: $output_name${NC}"
echo -e "${YELLOW}Exclude step descriptions: $exclude_step_descriptions${NC}"

# Display version information
echo -e "\n${CYAN}Prompt Version Info:${NC}"
if [ "$prompt_version" == "v1" ]; then
    echo -e "${WHITE}  - v1: Task summary + detailed steps format${NC}"
    echo -e "${WHITE}  - Output: task_summary and steps with descriptions${NC}"
elif [ "$prompt_version" == "v2" ]; then
    echo -e "${WHITE}  - v2: Simple segments only format${NC}"
    echo -e "${WHITE}  - Output: segments_of_sub_actions with frame ranges only${NC}"
fi

# --- Step 1: Run universal two-stage planning pipeline ---
echo -e "\n${BLUE}Step 1: Running universal two-stage RTX planning pipeline...${NC}"

# Build Python command arguments
python_args=(
    "1_pipeline_universal_two_stage.py"
    "--embodiment=$embodiment"
    "--root=../demo_frames"
    "--json=../demo_description/rtx_desc.json"
    "--output=../demo_output/${output_name}/${prompt_version}.json"
    "--prompt-version=$prompt_version"
    "--max=10"
)

# Add step descriptions parameter based on setting
if [ "$exclude_step_descriptions" = true ]; then
    python_args+=("--exclude-step-descriptions")
    echo -e "${YELLOW}  Note: Step descriptions from stage 1 will be excluded from stage 2${NC}"
else
    echo -e "${YELLOW}  Note: Step descriptions from stage 1 will be included in stage 2 (default)${NC}"
fi

# Execute Python command
python "${python_args[@]}"

# Check if Step 1 was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Step 1 completed successfully!${NC}"
else
    echo -e "${RED}Step 1 failed with exit code: $?${NC}"
    exit $?
fi

# --- Step 2: Reformat JSON output ---
echo -e "\n${BLUE}Step 2: Reformatting universal JSON output...${NC}"
python 2_json_reformat_two_stage.py \
  --input-json="../demo_output/${output_name}/${prompt_version}.json" \
  --output-json="../demo_output/${output_name}/${prompt_version}_reformat.json"

# Check if Step 2 was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Step 2 completed successfully!${NC}"
    echo -e "\n${GREEN}Universal two-stage pipeline completed successfully!${NC}"
    echo -e "${YELLOW}Output files:${NC}"
    echo -e "${CYAN}  - ../demo_output/${output_name}/${prompt_version}.json (Raw results)${NC}"
    echo -e "${CYAN}  - ../demo_output/${output_name}/${prompt_version}_reformat.json (Formatted results)${NC}"
    
    echo -e "\n${MAGENTA}To try a different prompt version:${NC}"
    echo -e "${WHITE}  1. Edit this script and change the 'prompt_version' variable${NC}"
    echo -e "${WHITE}  2. Available versions: v1, v2${NC}"
else
    echo -e "${RED}Step 2 failed with exit code: $?${NC}"
    exit $?
fi
