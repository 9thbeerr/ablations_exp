import os
import sys
import shutil
from huggingface_hub import hf_hub_download


def get(fname):
    local_dir = os.path.join(os.path.dirname(__file__), "finewebedu10bt", "raw_data")
    subpath = os.path.join("sample", "10BT", fname)

    # Create raw_data directory
    os.makedirs(local_dir, exist_ok=True)

    final_path = os.path.join(local_dir, fname)

    if not os.path.exists(final_path):
        # Download to temp location (will create sample/10BT structure)
        downloaded_path = hf_hub_download(
            repo_id="HuggingFaceFW/fineweb-edu",
            filename=subpath,
            repo_type="dataset",
            cache_dir=None,
        )
        # Move to raw_data/ directly
        shutil.copy2(downloaded_path, final_path)
        print(f"Downloaded: {fname}")
    else:
        print(f"Already exists: {fname}")


# Default number of chunks
chunk_no = 14
if len(sys.argv) >= 2:
    chunk_no = int(sys.argv[1])

# Download files
for i in range(chunk_no):
    fname = f"{i:03d}_00000.parquet"
    print(f"Downloading file {i + 1}/{chunk_no}: {fname}")
    get(fname)

print("Download complete!")
