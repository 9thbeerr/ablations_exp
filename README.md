# Ablations

Based on CS336 scaffolding to test run different research ideas.

## Requirements

- Python 3.12+
- PyTorch 2.8+
- [uv](https://github.com/astral-sh/uv) package manager

## Quick Start

### 1. Setup Environment

```bash
./server_setup.sh setup
```

### 2. Download Dataset & Train Tokenizer

```bash
./server_setup.sh download
./server_setup.sh train-tokenizer
```

Or run both together:

```bash
./server_setup.sh build-tokenizer
```

### 3. Train Model

```bash
./server_setup.sh train
```

To resume training from a checkpoint:

```bash
./server_setup.sh resume <wandb_run_id>
```

### 4. Generate Text

```bash
./server_setup.sh generate <wandb_run_id>
```

## Configuration

Edit `server_setup.sh` to configure:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | tinystories | Dataset to use |
| `D_MODEL` | 128 | Model dimension |
| `NUM_LAYERS` | 4 | Number of transformer layers |
| `NUM_HEADS` | 2 | Number of attention heads |
| `D_FF` | 256 | Feed-forward dimension |
| `NUM_STEPS` | 10000 | Training steps |
| `BATCH_SIZE` | 32 | Batch size |
| `DEVICE` | mps | Device (mps/cuda/cpu) |

## Project Structure

```
├── core/
│   ├── layers/          # Transformer layer implementations
│   ├── model.py         # Main transformer model
│   ├── train_model.py   # Training loop
│   ├── tokenization.py  # BPE tokenizer training
│   ├── dataloader.py    # Data loading utilities
│   ├── generate.py      # Text generation
│   └── evals.py         # Evaluation metrics
├── checkpoints/         # Model checkpoints
├── server_setup.sh      # Main CLI script
└── download_dataset_*.py # Dataset downloaders
```

## Available Commands

```bash
./server_setup.sh setup           # Install uv and create venv
./server_setup.sh download        # Download dataset
./server_setup.sh train-tokenizer # Train BPE tokenizer
./server_setup.sh train           # Train model
./server_setup.sh resume <id>     # Resume training
./server_setup.sh generate <id>   # Generate text
./server_setup.sh init            # Setup environment
./server_setup.sh build-tokenizer # Download + train tokenizer
./server_setup.sh train-pipeline  # Full training pipeline
```


