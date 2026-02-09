#!/bin/bash
set -e

# Shared Settings
PYTHON=python
export TOKENIZERS_PARALLELISM=true

# Model Settings
MODEL_NAME=tinystories
VOCAB_SIZE=100
MAX_SEQ_LEN=256

# Generation Settings
TEMPERATURE=0.8
INPUT_PROMPT="Hello world, "
GEN_MAX_SEQ_LEN=256

# Training Settings
D_MODEL=128
NUM_LAYERS=4
NUM_HEADS=2
D_FF=256
ROPE_THETA=10000
NUM_STEPS=10000
LR=6e-4
BETA1=0.9
BETA2=0.95
EPS=1e-8
WEIGHT_DECAY=0.001
LR_MAX=2e-4
LR_MIN=3e-5
WARMUP_ITERS=1000
COSINE_CYCLE_ITERS=8000
MAX_L2_NORM=5.0
USE_GRADIENT_CHECKPOINT=False
BATCH_SIZE=32
MODE=train
DEVICE=mps
WANDB_RUN_ID=""
NUM_WORKERS=$(python3 -c "import os; print(max(1, int(os.cpu_count() * 0.6)))")

ABLATION_CONFIG='{"mHC": true, "value_residual": false}'

# Functions
setup() {
    pip install uv
    uv venv --python 3.13
}

download_dataset() {
    uv run download_dataset_${MODEL_NAME}.py
}

generate() {
    local WANDB_RUN_ID="$1"
    uv run -m core.generate \
        --temperature "$TEMPERATURE" \
        --input_prompt "$INPUT_PROMPT" \
        --max_seq_len "$GEN_MAX_SEQ_LEN" \
        --model_name "$MODEL_NAME" \
        --wandb_run_id "$WANDB_RUN_ID" \
        --device "$DEVICE"
}

train_tokenizer() {
    uv run -m core.tokenization \
        --vocab_size "$VOCAB_SIZE" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --model_name "$MODEL_NAME" 
}

train_model() {
    local RESUME=""
    if [ "$1" = "resume" ]; then
        RESUME="--train_resume --wandb_run_id ${2:-$WANDB_RUN_ID}"
    fi
    
    uv run -m core.train_model \
        --vocab_size "$VOCAB_SIZE" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --d_model "$D_MODEL" \
        --num_layers "$NUM_LAYERS" \
        --num_heads "$NUM_HEADS" \
        --d_ff "$D_FF" \
        --rope_theta "$ROPE_THETA" \
        --num_steps "$NUM_STEPS" \
        --lr "$LR" \
        --beta1 "$BETA1" \
        --beta2 "$BETA2" \
        --eps "$EPS" \
        --weight_decay "$WEIGHT_DECAY" \
        --lr_max "$LR_MAX" \
        --lr_min "$LR_MIN" \
        --use_gradient_checkpoint "$USE_GRADIENT_CHECKPOINT" \
        --warmup_iters "$WARMUP_ITERS" \
        --cosine_cycle_iters "$COSINE_CYCLE_ITERS" \
        --max_l2_norm "$MAX_L2_NORM" \
        --mode "$MODE" \
        --model_name "$MODEL_NAME" \
        --device "$DEVICE" \
        --ablation_config "$ABLATION_CONFIG" \
        $RESUME
}

init() {
    setup
    echo "✓ Environment ready and dataset downloaded"
}

build_tokenizer() {
    download_dataset
    train_tokenizer
    echo "✓ Complete setup with tokenizer trained"
}

train_pipeline() {
    train_model
    echo "✓ Training pipeline complete"
}

case "${1:-help}" in
    setup) setup ;;
    download) download_dataset ;;
    generate)
        if [ -z "$2" ]; then
            echo "Error: WANDB_RUN_ID required"
            echo "Usage: $0 generate <wandb_run_id>"
            exit 1
        fi
        generate "$2"
        ;;
    train-tokenizer) train_tokenizer ;;
    train) train_model ;;
    resume) 
        if [ -z "$2" ]; then
            echo "Error: WANDB_RUN_ID required"
            echo "Usage: $0 resume <wandb_run_id>"
            exit 1
        fi
        train_model resume "$2"
        ;;
    init) init ;;
    build-tokenizer) build_tokenizer ;;
    train-pipeline) train_pipeline ;;
    *)
        echo "Usage: $0 {setup|download|generate|train-tokenizer|train|resume|init|build-tokenizer|train-pipeline}"
        exit 1
        ;;
esac