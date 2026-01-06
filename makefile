# === Shared Settings ===
PYTHON=python

# === General Model Settings ===
MODEL_NAME=finewebedu10b
VOCAB_SIZE=10000
MAX_SEQ_LEN=256

# === Make Targets ===
.PHONY: generate train_tokenizer run_tokenizer train_model resume_train_model

# === Generate ===
TEMPERATURE=0.8
INPUT_PROMPT="Hello world, "
GEN_MAX_SEQ_LEN=256

# === Training ===
D_MODEL=1024
NUM_LAYERS=16
NUM_HEADS=8
D_FF=1536
ROPE_THETA=10000
NUM_STEPS=5000
LR=6e-4
BETA1=0.9
BETA2=0.95
EPS=1e-8
WEIGHT_DECAY=0.1
LR_MAX=2e-4
LR_MIN=6e-5
WARMUP_ITERS=1000
COSINE_CYCLE_ITERS=8000
MAX_L2_NORM=1.0
USE_GRADIENT_CHECKPOINT=True
BATCH_SIZE=4
MODE=train
WANDB_RUN_ID=
### latest model to use to resume

generate:
	uv run -m core.generate \
		--temperature $(TEMPERATURE) \
		--input_prompt $(INPUT_PROMPT) \
		--max_seq_len $(GEN_MAX_SEQ_LEN) \
		--model_name $(MODEL_NAME) \
		--wandb_id $(WANDB_RUN_ID)

# === Tokenizer ===
train_tokenizer:
	uv run -m core.tokenization \
		--vocab_size $(VOCAB_SIZE) \
		--max_seq_len $(MAX_SEQ_LEN) \
		--model_name $(MODEL_NAME) \
		--train_tokenizer

run_tokenizer:
	uv run -m core.tokenization \
		--vocab_size $(VOCAB_SIZE) \
		--max_seq_len $(MAX_SEQ_LEN) \
		--model_name $(MODEL_NAME)

train_model:
	uv run -m core.train_model \
		--vocab_size $(VOCAB_SIZE) \
		--max_seq_len $(MAX_SEQ_LEN) \
		--d_model $(D_MODEL) \
		--num_layers $(NUM_LAYERS) \
		--num_heads $(NUM_HEADS) \
		--d_ff $(D_FF) \
		--rope_theta $(ROPE_THETA) \
		--num_steps $(NUM_STEPS) \
		--lr $(LR) \
		--beta1 $(BETA1) \
		--beta2 $(BETA2) \
		--eps $(EPS) \
		--weight_decay $(WEIGHT_DECAY) \
		--lr_max $(LR_MAX) \
		--lr_min $(LR_MIN) \
		--use_gradient_checkpoint $(USE_GRADIENT_CHECKPOINT) \
		--warmup_iters $(WARMUP_ITERS) \
		--cosine_cycle_iters $(COSINE_CYCLE_ITERS) \
		--max_l2_norm $(MAX_L2_NORM) \
		--mode $(MODE) \
		--model_name $(MODEL_NAME)

resume_train_model:
	uv run -m core.train_model \
		--vocab_size $(VOCAB_SIZE) \
		--max_seq_len $(MAX_SEQ_LEN) \
		--d_model $(D_MODEL) \
		--num_layers $(NUM_LAYERS) \
		--num_heads $(NUM_HEADS) \
		--d_ff $(D_FF) \
		--rope_theta $(ROPE_THETA) \
		--num_steps $(NUM_STEPS) \
		--lr $(LR) \
		--beta1 $(BETA1) \
		--beta2 $(BETA2) \
		--eps $(EPS) \
		--weight_decay $(WEIGHT_DECAY) \
		--lr_max $(LR_MAX) \
		--lr_min $(LR_MIN) \
		--use_gradient_checkpoint $(USE_GRADIENT_CHECKPOINT) \
		--warmup_iters $(WARMUP_ITERS) \
		--cosine_cycle_iters $(COSINE_CYCLE_ITERS) \
		--max_l2_norm $(MAX_L2_NORM) \
		--mode $(MODE) \
		--model_name $(MODEL_NAME) \
		--train_resume \
		--wandb_run_id $(WANDB_RUN_ID)