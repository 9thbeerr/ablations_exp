from tokenizers import Tokenizer
import argparse
import torch
import wandb
from core.model import TransformerLM, softmax
from pathlib import Path
import os


# single prompt
def generate_with_temperature(
    temperature: float,
    tokenizer: Tokenizer,
    model: torch.nn.Module,
    input_prompt: str,
    max_seq_len: int,
):
    input_ids = tokenizer.encode(input_prompt).ids
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(
        device
    )  # shape: [1, seq_len]

    for _ in range(max_seq_len - input_ids.shape[1]):
        logits = model(input_ids)
        probs = torch.exp(logits[0, -1] / temperature) / torch.sum(
            torch.exp(logits[0, -1] / temperature), dim=-1
        )
        next_token_id = torch.argmax(probs).item()
        next_token = tokenizer.decode([next_token_id])
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token_id]], device=device)], dim=1
        )

    print("-->", tokenizer.decode(input_ids[0].tolist()))


def generate_next_tokens_batch(
    temperature: float,
    tokenizer: Tokenizer,
    model: torch.nn.Module,
    prompts: list[str],
    max_seq_len: int,
):
    # Encode all prompts
    encoded = [tokenizer.encode(p).ids for p in prompts]
    max_len = max(len(ids) for ids in encoded)

    input_ids = torch.zeros((len(prompts), max_len), dtype=torch.long)
    for i, ids in enumerate(encoded):
        input_ids[i, : len(ids)] = torch.tensor(ids)

    input_ids = input_ids.to(device)

    # Generate tokens until max_seq_len
    model.eval()  # Add eval mode
    with torch.no_grad():  # Disable gradients
        for _ in range(max_seq_len - input_ids.shape[1]):
            logits = model(input_ids)
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_tokens = torch.argmax(probs, dim=-1).unsqueeze(1)

            input_ids = torch.cat([input_ids, next_tokens], dim=1)

    # Decode results
    for i, seq in enumerate(input_ids):
        print("_" * 100)
        print(f"[{i}] {tokenizer.decode(seq.tolist())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature setting for softmax"
    )
    parser.add_argument(
        "--input_prompt",
        type=str,
        default="Hello world, ",
        help="Input prompt to generate text max 512 context length",
    )

    parser.add_argument(
        "--max_seq_len", type=int, default=10, help="Max Sequence Length"
    )
    parser.add_argument("--model_name", type=str, help="Model Name")
    parser.add_argument("--device", type=str, help="cuda|mps|cpu", default="cpu")

    args = parser.parse_args()

    print(args)

    model_name = args.model_name
    device = args.device

    root_path = Path(os.getcwd())
    tokenizer_path = (
        root_path / "checkpoints" / model_name / f"tokenizer_{model_name}.json"
    )

    model_checkpoint = (
        root_path / "checkpoints" / model_name / f"{model_name}-{wandb_id}.pt"
    )

    print(str(model_checkpoint))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    checkpoint = torch.load(str(model_checkpoint))
    model_config = checkpoint["config"]

    print("Loading model config", model_config)

    model = TransformerLM(**model_config)
    model = torch.compile(model, backend="aot_eager")
    model.load_state_dict(checkpoint["weights"])

    generate_with_temperature(
        args.temperature, tokenizer, model, args.input_prompt, args.max_seq_len
    )
