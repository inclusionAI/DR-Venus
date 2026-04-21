#!/bin/bash
# DR-Venus Web Demo Startup Script

# 默认参数
PORT=7860
SHARE=false
MODEL_PATH="your-model-path"
NUM_GPUS=2
BASE_PORT=6000
GPU_UTIL=0.95
MAX_LEN=261000

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "  DR-Venus Web Demo Launcher"
echo "========================================="
echo "Port: $PORT"
echo "Model: $MODEL_PATH"
echo "GPUs: $NUM_GPUS"
echo ""

# 检查vLLM是否已经运行
echo "[1/3] Checking vLLM service..."
if curl -s http://127.0.0.1:${BASE_PORT}/v1/models > /dev/null 2>&1; then
    echo "  ✓ vLLM service already running on port $BASE_PORT"
    VLLM_RUNNING=true
else
    echo "  ✗ vLLM service not running, starting..."
    VLLM_RUNNING=false
fi

# 如果vLLM没有运行，启动它
if [ "$VLLM_RUNNING" = false ]; then
    echo "[2/3] Starting vLLM service..."


    # 启动vLLM
    nohup vllm serve "$MODEL_PATH" \
        --served-model-name model \
        --gpu-memory-utilization $GPU_UTIL \
        -tp $NUM_GPUS \
        --max-num-seqs 1 \
        --max-model-len $MAX_LEN \
        --enforce-eager \
        --port $BASE_PORT \
        > /tmp/vllm_server.log 2>&1 &

    VLLM_PID=$!
    echo "  vLLM started with PID: $VLLM_PID"

    # 等待vLLM服务就绪
    echo "  Waiting for vLLM to be ready..."
    for i in {1..60}; do
        if curl -s http://127.0.0.1:${BASE_PORT}/v1/models > /dev/null 2>&1; then
            echo "  ✓ vLLM service is ready!"
            break
        fi
        sleep 2
        if [ $i -eq 60 ]; then
            echo "  ✗ vLLM failed to start. Check /tmp/vllm_server.log"
            exit 1
        fi
    done
else
    echo "[2/3] Skipping vLLM start (already running)"
fi

# 启动Web Demo
echo "[3/3] Starting DR-Venus Web Demo..."


nohup python web_demo.py --port $PORT > /tmp/web_demo.log 2>&1 &
WEB_PID=$!
echo "  Web Demo started with PID: $WEB_PID"

sleep 3

echo ""
echo "========================================="
echo "  ✓ All services started successfully!"
echo "========================================="
echo ""
echo "Access the web interface at:"
echo "  http://localhost:$PORT"
echo ""
echo "To view logs:"
echo "  vLLM: tail -f /tmp/vllm_server.log"
echo "  Web:  tail -f /tmp/web_demo.log"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# 等待用户中断
trap "echo 'Stopping services...'; kill $VLLM_PID $WEB_PID 2>/dev/null; exit 0" INT TERM

wait