import os
import sys
from huggingface_hub import hf_hub_download

# Download PleIAs/SYNTH Parquet chunks from Hugging Face
def get(fname):
    local_dir = os.path.join(os.path.dirname(__file__), 'synth')
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, fname)
    if not os.path.exists(local_path):
        hf_hub_download(
            repo_id="PleIAs/SYNTH",
            filename=fname,
            repo_type="dataset",
            local_dir=local_dir
        )

# Default number of chunks to download (can override via CLI argument)
chunk_no = 50  # adjust based on how many parquet files exist
if len(sys.argv) >= 2:
    chunk_no = int(sys.argv[1])

# Download files like synth_001.parquet, synth_002.parquet, ...
for i in range(1, chunk_no + 1):
    fname = f"synth_{i:03d}.parquet"
    print("Downloading file:", fname)
    get(fname)