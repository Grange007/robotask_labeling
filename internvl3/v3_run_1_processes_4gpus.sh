#! /bin/bash
# 2进程配置脚本

conda activate /media/users/wd/conda_envs/internvl

# export http_proxy=http://192.168.32.28:18000
# export https_proxy=http://192.168.32.28:18000
export https_proxy=127.0.0.1:7890 && export http_proxy=127.0.0.1:7890

# 设置环境变量
export WORLD_SIZE=1
export RANK=0
# 2进程配置
export NPROC_PER_NODE=1

MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-"2"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
--nproc_per_node $NPROC_PER_NODE
--nnodes $NNODES
--node_rank $NODE_RANK
--master_addr $MASTER_ADDR
--master_port $MASTER_PORT
)

echo DISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS[@]}

export PYTHONPATH=/media/users/wd/projects/dense_recaption/internvl3:$PYTHONPATH

PYTHONPATH=$PYTHONPATH \
    /media/users/wd/conda_envs/internvl/bin/torchrun ${DISTRIBUTED_ARGS[@]} \
    /media/users/wd/projects/dense_recaption/internvl3/lmdeploy_ano_after_seg_new_4gpus.py \
    --dataset_path /media/users/wd/projects/dense_recaption/internvl3/data/test.jsonl \
    --output_path /media/users/wd/projects/dense_recaption/internvl3/result/0906_robomind_annotation_V3_test_4gpu.jsonl \
    --model_path /media/users/wd/hf/hf_models/InternVL3-78B \
    --max_frames 100 \
    --max_new_tokens 1024 \
    --batch_size 1 \
    --new_key annotation \
    --target_fps 3 \
    --show_video \
    --seg_path /media/users/wd/projects/dense_recaption/internvl3/data/seg/example_seg_our_detect--prompt828_v1_reformat.json