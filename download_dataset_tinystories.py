import os
from huggingface_hub import hf_hub_download


def split_file(input_path, train_path, valid_path, split_ratio=0.9):
    """Split file into train/valid sets"""
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    split_idx = int(len(lines) * split_ratio)

    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(lines[:split_idx])

    with open(valid_path, "w", encoding="utf-8") as f:
        f.writelines(lines[split_idx:])

    print(
        f"Split {len(lines)} lines -> {split_idx} train, {len(lines) - split_idx} valid"
    )


# Create directories
base_dir = os.path.join(os.path.dirname(__file__), "tinystories", "raw_data")
train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# Download TinyStories-valid.txt (smaller for dry run)
print("Downloading TinyStories-valid.txt...")
downloaded = hf_hub_download(
    repo_id="roneneldan/TinyStories",
    filename="TinyStories-valid.txt",
    repo_type="dataset",
)

# Split into train and valid
train_path = os.path.join(train_dir, "train.txt")
valid_path = os.path.join(valid_dir, "valid.txt")

print("Splitting file...")
split_file(downloaded, train_path, valid_path, split_ratio=0.9)

print("Complete!")
print(f"Train: {train_path}")
print(f"Valid: {valid_path}")
