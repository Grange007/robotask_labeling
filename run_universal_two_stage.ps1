# PowerShell script for Universal Two-Stage RTX Planning Pipeline
# 通用两阶段RTX规划管道 - 支持多种prompt版本

# 设置变量
$embodiment = "dual_arm_mine"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$output_name = "gemini_internvl_output_universal_no_desc_$timestamp"  # 输出文件夹名称
$prompt_version = "v1"  # 可选: v1 或 v2, v1 for example_segment + our_detect, v2 for our_segment+our_detect
$exclude_step_descriptions = $true  # 设置为 $true 来排除第一阶段的step descriptions

# --- Model Configuration ---
$stage2_model = "internvl"  # Options: "gemini" (cloud API) or "internvl" (local InternVL2.5)
$internvl_model_path = "/media/users/wd/hf/hf_models/InternVL3-78B"  # Path to InternVL model (only needed if stage2_model=internvl)

Write-Host "Starting Universal Two-Stage RTX Planning Pipeline..." -ForegroundColor Green
Write-Host "Embodiment: $embodiment" -ForegroundColor Yellow
Write-Host "Prompt Version: $prompt_version" -ForegroundColor Yellow
Write-Host "Output name: $output_name" -ForegroundColor Yellow
Write-Host "Exclude step descriptions: $exclude_step_descriptions" -ForegroundColor Yellow
Write-Host "Stage 2 Model: $stage2_model" -ForegroundColor Yellow
if ($stage2_model -eq "internvl") {
    Write-Host "InternVL Model Path: $internvl_model_path" -ForegroundColor Yellow
}

# 显示配置信息
Write-Host "`nPipeline Configuration:" -ForegroundColor Cyan
Write-Host "  Stage 1: Gemini (Segmentation)" -ForegroundColor White
if ($stage2_model -eq "internvl") {
    Write-Host "  Stage 2: InternVL2.5 (Local Detailed Analysis)" -ForegroundColor White
} else {
    Write-Host "  Stage 2: Gemini (Cloud Detailed Analysis)" -ForegroundColor White
}

# 显示版本信息
Write-Host "`nPrompt Version Info:" -ForegroundColor Cyan
if ($prompt_version -eq "v1") {
    Write-Host "  - v1: Task summary + detailed steps format" -ForegroundColor White
    Write-Host "  - Output: task_summary and steps with descriptions" -ForegroundColor White
} elseif ($prompt_version -eq "v2") {
    Write-Host "  - v2: Simple segments only format" -ForegroundColor White
    Write-Host "  - Output: segments_of_sub_actions with frame ranges only" -ForegroundColor White
}

# 第一步：运行通用两阶段规划管道
Write-Host "`nStep 1: Running universal two-stage RTX planning pipeline..." -ForegroundColor Blue

# 构建Python命令参数
$python_args = @(
    "1_pipeline_universal_two_stage.py",
    "--embodiment=$embodiment",
    "--root=../demo_frames_mine",
    "--json=../demo_description/my_desc.json",
    "--output=../demo_output/${output_name}/${prompt_version}.json",
    "--prompt-version=$prompt_version",
    "--stage2-model=$stage2_model",
    "--max=10"
)

# Add InternVL model path if using internvl
if ($stage2_model -eq "internvl") {
    $python_args += "--internvl-model-path=$internvl_model_path"
}

# 根据设置添加step descriptions参数
if ($exclude_step_descriptions) {
    $python_args += "--exclude-step-descriptions"
    Write-Host "  Note: Step descriptions from stage 1 will be excluded from stage 2" -ForegroundColor Yellow
} else {
    Write-Host "  Note: Step descriptions from stage 1 will be included in stage 2 (default)" -ForegroundColor Yellow
}

# Display command to be executed
Write-Host "  Command: python $($python_args -join ' ')" -ForegroundColor Cyan

# 执行Python命令
python @python_args

# 检查第一步是否成功
if ($LASTEXITCODE -eq 0) {
    Write-Host "Step 1 completed successfully!" -ForegroundColor Green
} else {
    Write-Host "Step 1 failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# 第二步：格式化JSON输出
Write-Host "`nStep 2: Reformatting universal JSON output..." -ForegroundColor Blue
python 2_json_reformat_two_stage.py `
  --input-json=../demo_output/${output_name}/${prompt_version}.json `
  --output-json=../demo_output/${output_name}/${prompt_version}_reformat.json

# 检查第二步是否成功
if ($LASTEXITCODE -eq 0) {
    Write-Host "Step 2 completed successfully!" -ForegroundColor Green
    Write-Host "`nUniversal two-stage pipeline completed successfully!" -ForegroundColor Green
    Write-Host "Output files:" -ForegroundColor Yellow
    Write-Host "  - ../demo_output/${output_name}/${prompt_version}.json (Raw results)" -ForegroundColor Cyan
    Write-Host "  - ../demo_output/${output_name}/${prompt_version}_reformat.json (Formatted results)" -ForegroundColor Cyan    Write-Host "`nTo try different configurations:" -ForegroundColor Magenta
    Write-Host "  1. Edit this script and change `$prompt_version variable (v1, v2)" -ForegroundColor White
    Write-Host "  2. Edit this script and change `$stage2_model variable:" -ForegroundColor White
    Write-Host "     - 'gemini': Use cloud Gemini API for stage 2" -ForegroundColor White
    Write-Host "     - 'internvl': Use local InternVL2.5 model for stage 2" -ForegroundColor White
    Write-Host "  3. Update `$internvl_model_path if using InternVL with different model location" -ForegroundColor White
} else {
    Write-Host "Step 2 failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
