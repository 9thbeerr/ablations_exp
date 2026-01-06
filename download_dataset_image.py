import os
import sys
from huggingface_hub import hf_hub_download


def get(fname):
    local_dir = os.path.join(os.path.dirname(__file__), "imagedata")
    subpath = os.path.join(fname)
    local_path = os.path.join(local_dir, subpath)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if not os.path.exists(local_path):
        hf_hub_download(
            repo_id="pixparse/cc3m-wds",
            filename=subpath,
            repo_type="dataset",
            local_dir=local_dir,
        )


# Default number of chunks to download (can override via CLI argument)
chunk_no = 20  # adjust based on how many parquet files exist
if len(sys.argv) >= 2:
    chunk_no = int(sys.argv[1])

# Download files like 000_00000.parquet, 000_00001.parquet, ...
for i in range(chunk_no):
    fname = f"cc3m-train-{i:04d}.tar"
    print("Downloading fiile:", fname)
    get(fname)
