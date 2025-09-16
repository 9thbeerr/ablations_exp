import argparse
from tokenizers import ByteLevelBPETokenizer
from tokenizers import Tokenizer
import numpy as np
import os
from pathlib import Path
import struct

# Data / Paths

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--vocab_size', type=int, default=1024)
    parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--train_tokenizer', action='store_true')


    args = parser.parse_args()
    print(args)

    model_name = args.model_name ## tineystoriesv2-gpt
    root_path = Path(os.getcwd())
    data_dir = root_path / model_name


    train_tokenizer = True if args.train_tokenizer else False


    ## this will get all text file from raw_data/train and valid to train a tokenizer.
    training_filename_list = [
        str(f.relative_to(root_path))
        for f in Path(data_dir, "raw_data").rglob("*.txt")
        if f.is_file()
    ]

    checkpoints = root_path / "checkpoints" / model_name 
    if not checkpoints.exists():
        checkpoints.mkdir(parents=True)
    
    tokenizer_path = checkpoints / f"tokenizer_{model_name}.json"


    if train_tokenizer:

        print("Training a BPETokenizer with files:", training_filename_list)

        bpe_tokenizer = ByteLevelBPETokenizer()
        bpe_tokenizer.train(
            files=training_filename_list,
            vocab_size=args.vocab_size,
            min_frequency=4,
            special_tokens=["<UNK>", "<|endoftext|>"]
        )    
        bpe_tokenizer.save(str(tokenizer_path))

    else:
        print("Starting Tokenization of Training data..")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

        for mode in ["train", "valid"]:
            processed_data_dir = data_dir / "processed_data" / mode
            if not processed_data_dir.exists():
                processed_data_dir.mkdir(parents=True)

            tokenized_data_bin_path = processed_data_dir / f"{mode}.bin"
            tokenized_data_idx_path = processed_data_dir / f"{mode}.idx"

            ## this will get all text file for tokenization.

            tokenize_filename_list = [
                str(f.relative_to(root_path))
                for f in Path(data_dir, "raw_data", mode).rglob("*")
                if f.is_file()
            ]

            print("List file to tokenize:", tokenize_filename_list)

            # total_size = sum(os.path.getsize(f) for f in tokenize_filename_list)

            # chunk_count = 0
            # chunk_size = args.max_seq_len * 1024 * 256
            # total_chunks = total_size // chunk_size
            # print("Total Chunks Estimated: ", total_chunks)

            # with open(tokenized_data_save_path, "wb") as out_f:

            #     for raw_data_file in tokenize_filename_list:
            #         with open(raw_data_file, "r") as f:
            #             while True:
            #                 text_chunk = f.read(chunk_size)
            #                 if not text_chunk:
            #                     break
            #                 chunk_count +=1
            #                 tokenized_data = tokenizer.encode(text_chunk).ids
            #                 np_array = np.array(tokenized_data, dtype=np.uint16)
            #                 out_f.write(np_array.tobytes())
            #                 print(f"Total Chunks written: {chunk_count}")

            chunk_size = args.max_seq_len * 1024 * 256
            doc_offsets = []
            chunk_count = 0
            total_tokens = 0

            with open(tokenized_data_bin_path, "wb") as out_file:
                for raw_data_file in tokenize_filename_list:
                    with open(raw_data_file, "r", encoding="utf-8") as f:
                        while True:
                            text_chunk = f.read(chunk_size)
                            if not text_chunk:
                                break

                            token_ids = tokenizer.encode(text_chunk).ids
                            token_arr = np.array(token_ids, dtype=np.uint16)
                            out_file.write(token_arr.tobytes())

                            doc_offsets.append((total_tokens, len(token_ids)))
                            total_tokens = total_tokens + len(token_ids)
                            chunk_count += 1
                            print(f"Chunks written: {chunk_count}, Tokens so far: {total_tokens}")

            with open(tokenized_data_idx_path, "wb") as f:
                f.write(struct.pack("<Q", len(doc_offsets)))  # number of documents
                for offset, length in doc_offsets:
                    f.write(struct.pack("<QQ", offset, length))




