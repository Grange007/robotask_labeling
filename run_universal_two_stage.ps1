# PowerShell script for Universal Two-Stage RTX Planning Pipeline
# 通用两阶段RTX规划管道 - 支持多种prompt版本

# 设置变量
$embodiment = "dual_arm_mine"
$output_name = "planning_prompt_1_pick_and_place"  # 输出文件夹名称
$prompt_version = "v1"  # 可选: v1 或 v2, v1 for example_segment + our_detect, v2 for our_segment+our_detect

Write-Host "Starting Universal Two-Stage RTX Planning Pipeline..." -ForegroundColor Green
Write-Host "Embodiment: $embodiment" -ForegroundColor Yellow
Write-Host "Prompt Version: $prompt_version" -ForegroundColor Yellow
Write-Host "Output name: $output_name" -ForegroundColor Yellow

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
python 1_pipeline_universal_two_stage.py `
  --embodiment=$embodiment `
  --root=../demo_frames_mine `
  --json=../demo_description/my_desc.json `
  --output=../demo_output/${output_name}/${prompt_version}.json `
  --prompt-version=$prompt_version `
  --max=10

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
    Write-Host "  - ../demo_output/${output_name}/${prompt_version}_reformat.json (Formatted results)" -ForegroundColor Cyan

    Write-Host "`nTo try different prompt version:" -ForegroundColor Magenta
    Write-Host "  1. Edit this script and change `$prompt_version variable" -ForegroundColor White
    Write-Host "  2. Available versions: v1, v2" -ForegroundColor White
} else {
    Write-Host "Step 2 failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
