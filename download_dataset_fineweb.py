# import os
# import sys
# from huggingface_hub import hf_hub_download

# # Download FineWeb-Edu 10BT Parquet chunks from Hugging Face

# def get(fname):
#     local_dir = os.path.join(os.path.dirname(__file__), 'fineweb10B')
#     os.makedirs(local_dir, exist_ok=True)
#     local_path = os.path.join(local_dir, fname)
#     if not os.path.exists(local_path):
#         hf_hub_download(
#             repo_id="HuggingFaceFW/fineweb-edu",
#             filename=f"sample/10BT/{fname}",
#             repo_type="dataset",
#             local_dir=local_dir
#         )

# # Default number of chunks to download (can override via CLI argument)
# chunk_no = 13  # adjust based on how many parquet files exist
# if len(sys.argv) >= 2:
#     chunk_no = int(sys.argv[1])

# # Download files like 000_00000.parquet, 000_00001.parquet, ...
# for i in range(chunk_no):
#     fname = f"{chunk_no:03d}_00000.parquet"
#     print("Downloading fiile:",fname)
#     get(fname)
