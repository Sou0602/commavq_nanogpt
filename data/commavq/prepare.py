# Updated from the commavq github
# https://github.com/commaai/commavq/blob/master/nanogpt/prepare.py
# saves commaVQ in a format that can be used by nanogpt.train.py
# this writes 40 files of 774MB each, for a total of 30GB
# modified from https://github.com/karpathy/nanoGPT/tree/master/data
import os

import numpy as np
from datasets import load_dataset  # Huggingface datasets
from tqdm import tqdm

# Number of workers in .map() call
num_proc = 40
num_proc_load_dataset = num_proc

# Constants for special tokens
BOS_TOKEN = 1024
EOT_TOKEN = 1025


def main():
    # Load the dataset
    dataset = load_dataset("commaai/commavq", num_proc=num_proc_load_dataset)

    # Define a function to process individual examples
    def process(example):
        tokens = np.load(example["path"])
        tokens = tokens.reshape(tokens.shape[0], -1)
        tokens = np.c_[np.ones(len(tokens), dtype=np.int16) * BOS_TOKEN, tokens]
        tokens = tokens.reshape(-1)
        tokens = np.r_[tokens, EOT_TOKEN]
        return {"ids": tokens.astype(np.int16), "len": len(tokens.astype(np.int16))}

    # Tokenize, add special tokens, and flatten the dataset
    tokenized = dataset.map(
        process,
        desc="Tokenizing the splits",
        num_proc=num_proc,
    )
    dir_name = os.path.dirname(__file__)
    # Save tokenized data into binary files for each split
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(dir_name, f"{split}.bin")
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 100 if split == "40" else 1024
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # Create train.bin by concatenating files from 0.bin to 39.bin
    concatenation_command = f'for i in $(seq 0 39); do cat "{dir_name}/$i.bin" >> "{dir_name}/train.bin"; rm "{dir_name}/$i.bin"; done'
    os.system(concatenation_command)

    # Rename 40.bin to val.bin
    rename_command = f'cp "{dir_name}/40.bin" "{dir_name}/val.bin"'
    os.system(rename_command)

    # Remove 40.bin
    remove_command = f'rm "{dir_name}/40.bin"'
    os.system(remove_command)


if __name__ == "__main__":
    main()
