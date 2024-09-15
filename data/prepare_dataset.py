# prepare OpenWebText dataset and split into train and validation sets

import os
import random
import json
import pathlib 
import shutil

import tiktoken
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

enc = tiktoken.get_encoding('gpt2')

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

if __name__ == "__main__":

    # load dataset
    # note: load_dataset() loads the dataset to the cache, which is ~/.cache/huggingface/datasets
    # this cache is not cleaned up automatically, so it's better to delete it manually
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset,)
    # split dataset into train and validation sets
    split_dataset = dataset['train'].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

    # tokenize dataset
    def process(dataset):
        ids = enc.encode_ordinary(dataset['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    tokenized_dataset = split_dataset.map(
        process, 
        remove_columns=['text'],
        desc="Tokenizing the dataset",
        num_proc=num_proc
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized_dataset.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # delete the dataset cache manually, because huggingface is not doing it
    home_dir = pathlib.Path.home()
    dataset_dir = home_dir / ".cache" / "huggingface"
    
    shutil.rmtree(str(dataset_dir))