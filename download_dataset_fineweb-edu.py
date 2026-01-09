import os
import sys
import shutil
from huggingface_hub import hf_hub_download


def get(fname, target_dir):
    subpath = os.path.join("sample", "10BT", fname)
    os.makedirs(target_dir, exist_ok=True)
    final_path = os.path.join(target_dir, fname)

    if not os.path.exists(final_path):
        downloaded_path = hf_hub_download(
            repo_id="HuggingFaceFW/fineweb-edu",
            filename=subpath,
            repo_type="dataset",
            cache_dir=None,
        )
        shutil.copy2(downloaded_path, final_path)
        print(f"Downloaded: {fname} -> {target_dir}")
    else:
        print(f"Already exists: {fname}")


# Get number of chunks
chunk_no = int(sys.argv[1]) if len(sys.argv) >= 2 else 14

# Create directories
base_dir = os.path.join(os.path.dirname(__file__), "fineweb-edu", "raw_data")
train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")

# Download files
for i in range(chunk_no):
    fname = f"{i:03d}_00000.parquet"
    # Last file goes to valid, rest to train
    target_dir = valid_dir if i == chunk_no - 1 else train_dir
    print(f"Downloading file {i + 1}/{chunk_no}: {fname}")
    get(fname, target_dir)

print("Download complete!")
print(f"Train files: {chunk_no - 1}")
print(f"Valid files: 1")
