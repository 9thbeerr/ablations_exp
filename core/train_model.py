import enum
import sys
import signal
import argparse
import os
from tokenizers import Tokenizer
import torch

from core.dataloader import StreamingMegatronDataset
from core.generate import generate_next_tokens_batch
from core.model import (
    AdamW,
    TransformerLM,
    calculate_cross_entropy,
    gradient_clipping,
    learning_rate_schedule,
    load_checkpoint,
    save_checkpoint,
)
from pathlib import Path

import wandb


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Model + Training Config
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--rope_theta", type=int, default=10000)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)

    # Optimizer Config
    parser.add_argument("--lr", type=float, default=0.99)
    parser.add_argument("--beta1", type=float, default=0.99)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.04)

    # LR Schedule
    parser.add_argument("--lr_max", type=float, default=0.01)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--cosine_cycle_iters", type=int, default=900)

    # Gradient Clipping
    parser.add_argument("--max_l2_norm", type=float, default=0.99)
    parser.add_argument("--use_gradient_checkpoint", type=bool, default=False)

    # Data / Paths
    parser.add_argument("--mode", type=str, default="valid")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--device", type=str, required=True, default="mps")

    resume_group = parser.add_argument_group("Resume Training Options")
    resume_group.add_argument(
        "--train_resume", action="store_true", help="Resume training from previous run"
    )
    resume_group.add_argument("--wandb_run_id", type=str, help="W&B run ID to resume")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Conditional logic
    if args.train_resume:
        if not args.wandb_run_id:
            args.error("--wandb_run_id is required when --train_resume is set")
        run_id = args.wandb_run_id
    else:
        run_id = wandb.util.generate_id()  # Generate consistent ID for new runs

    logger_run = wandb.init(
        entity="nikeshnaik-dev",
        project=args.model_name,
        resume="allow",
        config={
            "vocab_size": args.vocab_size,
            "max_seq_len": args.max_seq_len,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff,
            "rope_theta": args.rope_theta,
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            # Optimizer
            "lr": args.lr,
            "betas": (args.beta1, args.beta2),
            "eps": args.eps,
            "weight_decay": args.weight_decay,
            "use_gradient_checkpoint": args.use_gradient_checkpoint,
            # LR Schedule
            "lr_max": args.lr_max,
            "lr_min": args.lr_min,
            "warmup_iters": args.warmup_iters,
            "cosine_cycle_iters": args.cosine_cycle_iters,
            # Gradient Clipping
            "max_l2_norm": args.max_l2_norm,
            "device": args.device,
        },
    )

    root_path = Path(os.getcwd())
    model_name = args.model_name
    model_run = f"{logger_run.project}-{run_id}"

    mode = args.mode

    checkpoint_dir = root_path / "checkpoints"

    # to check if checkpoint exists and run from there or error for no dir found nor model found
    model_checkpoint_path = checkpoint_dir / model_name / f"{model_run}.pt"

    tokenizer_path = checkpoint_dir / model_name / f"tokenizer.json"

    raw_data_dir = root_path / model_name / "raw_data"

    model_config = {
        "vocab_size": args.vocab_size,
        "context_length": args.max_seq_len,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "use_gradient_checkpoint": args.use_gradient_checkpoint,
        "rope_theta": args.rope_theta,
        "device": args.device,
    }

    hyperparameters = {
        # Optimizer
        "num_steps": args.num_steps,
        "lr": args.lr,
        "betas": (args.beta1, args.beta2),
        "eps": args.eps,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        # LR Schedule
        "lr_max": args.lr_max,
        "lr_min": args.lr_min,
        "warmup_iters": args.warmup_iters,
        "cosine_cycle_iters": args.cosine_cycle_iters,
        # Gradient Clipping
        "max_l2_norm": args.max_l2_norm,
    }

    train_files = [
        f
        for f in (raw_data_dir / "train").rglob("*")
        if f.suffix.lower() in {".txt", ".parquet", ".csv"}
    ]

    valid_files = [
        f
        for f in (raw_data_dir / "valid").rglob("*")
        if f.suffix.lower() in {".txt", ".parquet", ".csv"}
    ]
    current_checkpoints = [
        f for f in model_checkpoint_path.rglob(f"*{run_id}.pt") if f.is_file()
    ]

    print(
        f"Model Name: {model_name},\nRoot: {root_path},\nMode: {mode},\nDatasets: {train_files + valid_files},\nCheckpoint_path: {current_checkpoints}, \nWandb ID: {run_id}"
    )

    print(tokenizer_path)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # tokenized_data = np.memmap(str(train_dataset_path), dtype=np.int64, mode="r")

    train_dataset = StreamingMegatronDataset(
        files=train_files,
        tokenizer_path=str(tokenizer_path),
        context_length=args.max_seq_len,
        batch_size=args.batch_size,
        device=args.device,
        shuffle_buffer=8192,
    )

    valid_dataset = StreamingMegatronDataset(
        files=valid_files,
        tokenizer_path=str(tokenizer_path),
        context_length=args.max_seq_len,
        batch_size=args.batch_size,
        device=args.device,
        shuffle_buffer=2048,  # smaller is fine for eval
    )

    model = TransformerLM(**model_config)
    model = torch.compile(model, backend="aot_eager")
    optimizer = AdamW(
        params=model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    step = 0

    if args.train_resume:
        step = load_checkpoint(
            src=str(model_checkpoint_path), model=model, optimizer=optimizer
        )
        print(
            f"Resuming training from step: {step} \nmodel config: {model_config} \nHyperParameters: {hyperparameters}"
        )

    else:
        print(
            f"Training start with \nmodel config: {model_config} \nHyperParameters: {hyperparameters}"
        )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {total_params:.2f}M")
    trainable_params = (
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    )
    print(f"Trainable parameters: {trainable_params:.2f}M")

    for step, (x, y) in enumerate(train_dataset):
        logits = model(x)
        loss = calculate_cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        total_norm = gradient_clipping(model.parameters(), args.max_l2_norm)

        wandb.log(data={"grad_norm": total_norm})
        optimizer.step()
        lr = learning_rate_schedule(
            step,
            lr_max=args.lr_max,
            lr_min=args.lr_min,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        print(f"Step:{step}, Loss: {loss}, LR:{lr}")

        wandb.log({"loss": loss.item(), "step": step, "learning_rate": lr})

        if step % 100 == 0 and step >= 10:
            print("Saving Checkpoint at step:", step)
            save_checkpoint(
                model,
                optimizer,
                step,
                out=str(model_checkpoint_path),
                config=model_config,
            )

            val_loss_per_step = []
            for val_steps, (x_val, y_val) in enumerate(valid_dataset):
                if val_steps < 50:
                    model.eval()
                    with torch.no_grad():
                        val_logits = model(x_val)
                        val_loss = calculate_cross_entropy(logits, y_val)
                        val_loss_per_step.append(val_loss.item())

            val_final_loss = sum(val_loss_per_step) / len(val_loss_per_step)
            wandb.log({"val_loss": val_final_loss, "step": step})
            print(f"Validation Loss at step: {step} | {val_final_loss}")

            ## Generation of Text
            eval_prompts = [
                # 1) Next-token sanity
                "Once upon a time",
                # 2) Grammar acquisition
                "The cat sat on the",
                # 3) Short-range coherence
                "Tom had a red ball. He threw the ball and it",
                # 4) Memorization vs generalization
                "Alice went to the forest to find a",
                # 5) Repetition / collapse detector
                "ha ha ha ha ha ha ha",
                # 6) Long-range dependency
                "John picked up the key. Mary locked the door. John tried to open it but",
                # 7) Counting / structure
                "Count from 1 to 5:",
                # 8) Dataset fingerprint
                "Explain why learning is important.",
            ]

            generate_next_tokens_batch(
                0.8, tokenizer, model, eval_prompts, 64, device=args.device
            )

            model.train()

        if step > args.num_steps:
            break

    def cleanup(*_):
        if wandb.run is not None:
            wandb.finish()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
