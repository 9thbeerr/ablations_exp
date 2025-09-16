from tokenizers import Tokenizer
import argparse
import torch
import wandb
from core.model import TransformerLM, softmax
from pathlib import Path
import os

def generate_with_temperature(temperature:float, tokenizer:Tokenizer, model:torch.nn.Module, input_prompt:str, max_seq_len:int, ):

    input_ids = tokenizer.encode(input_prompt).ids
    input_ids = torch.tensor([input_ids], dtype=torch.long).to("mps")  # shape: [1, seq_len]

    for i in range(max_seq_len):
        logits = model(input_ids)
        probs = torch.exp(logits[0, -1]/temperature) / torch.sum(torch.exp(logits[0, -1]/temperature), dim=-1)
        next_token_id = torch.argmax(probs).item() 
        next_token = tokenizer.decode([next_token_id])
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device="mps")], dim=1)
        print("-->",tokenizer.decode(input_ids[0].tolist()))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--temperature', type=float, default=0.8, help="Temperature setting for softmax")
    parser.add_argument("--input_prompt", type=str, default="Hello world, ", help="Input prompt to generate text max 512 context length")

    parser.add_argument("--max_seq_len", type=int, default=10, help="Max Sequence Length")
    parser.add_argument("--model_name", type=str, help="Model Name")
    parser.add_argument("--wandb_id", type=str, help="Wandb_id of saved model checkpoint")

    args = parser.parse_args()

    print(args)

    model_name = args.model_name
    wandb_id = args.wandb_id

    root_path = Path(os.getcwd())
    tokenizer_path = root_path / "checkpoints" / model_name / f"tokenizer_{model_name}.json"

    model_checkpoint = root_path / "checkpoints" / model_name / f"{model_name}-{wandb_id}.pt"

    device = "mps"

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    checkpoint = torch.load(str(model_checkpoint))
    model_config = checkpoint["config"]

    print("Loading model config", model_config)

    model = TransformerLM(**model_config) 
    model = torch.compile(model, backend="aot_eager")
    model.load_state_dict(checkpoint["weights"])


    generate_with_temperature(args.temperature, tokenizer, model, args.input_prompt, args.max_seq_len)




