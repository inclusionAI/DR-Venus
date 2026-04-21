#!/bin/bash
# --- 配置区域 ---

MODEL_PATH="your-model-path"
QUESTION="In April of 1977, who was the Prime Minister of the first place mentioned by name in the Book of Esther (in the New International Version)?"
OUTPUT_FILE="./demo_trajactory.jsonl"

NUM_GPUS=2               # GPU 数量（deploy_mode:tp 所有卡共同启动一个 vLLM 服务）
BASE_PORT=6051           # 起始端口（依次递增：6001, 6002, ..., 6008）
GPU_UTIL=0.95            # 显存占用比例
MAX_LEN=261000           # 最大上下文长度
BATCH_SIZE=1            # 每个 vLLM 服务的 max-num-seqs
NUM_WORKERS=1           # 线程池并发数（建议 >= NUM_GPUS 以充分利用所有服务）
MAX_STEPS=200            # 每题最大步数
SOLVER_MAX_LEN=230000    # Solver token 上限，比MAX_LEN略少，留最后一轮的生成空间
TIME_LIMIT=18000          # 每题时间限制（秒，300 分钟）

# --- 执行 ---

echo "========================================="
echo "  Deploy Mode : tp"
echo "  Num GPUs    : $NUM_GPUS"
echo "  Ports       : $BASE_PORT ~ $((BASE_PORT + NUM_GPUS - 1))"
echo "  Model       : $MODEL_PATH"
echo "  Input       : $INPUT_FILE"
echo "  Output      : $OUTPUT_FILE"
echo "  Log File    : $LOG_FILE"
echo "========================================="

python run_demo.py \
    --deploy_mode tp \
    --num_gpus $NUM_GPUS \
    --base_port $BASE_PORT \
    --model_path "$MODEL_PATH" \
    --gpu_util $GPU_UTIL \
    --max_model_len $MAX_LEN \
    --question "$QUESTION" \
    --output_file "$OUTPUT_FILE" \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --max_steps $MAX_STEPS \
    --solver_max_len $SOLVER_MAX_LEN \
    --time_limit $TIME_LIMIT \
    2>&1 | tee "$LOG_FILE"

echo "Job Finished."
