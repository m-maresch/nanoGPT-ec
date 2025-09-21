# saves a subset of the openwebtext dataset to a binary file for training.

import io
import lzma
import os
import pickle
import tarfile

import numpy as np


def remove_non_ascii(input):
    return "".join(character for character in input if ord(character) < 128)


def read_text_files_in_xz_tar(tar_file_path):
    """
    Reads .txt files within .xz files inside a .tar
    """
    with tarfile.open(tar_file_path, "r") as tar:
        for item in tar:
            if item.isfile() and item.name.endswith(".xz"):
                print(f"Found {item.name}")

                xz_data = tar.extractfile(item)
                decompressed_xz_data = lzma.decompress(xz_data.read())

                with io.BytesIO(decompressed_xz_data) as fileobj:
                    with tarfile.open(fileobj=fileobj) as files:
                        for file in files.getmembers():
                            if file.isfile() and file.name.endswith(".txt"):
                                print(f"Found {file.name}")
                                with files.extractfile(file) as txt_file:
                                    content = txt_file.read().decode("utf-8")
                                    content = remove_non_ascii(content)
                                    yield content


openwebtext = read_text_files_in_xz_tar("data/openwebtext/urlsf_subset01.tar")
data = "\n".join(list(openwebtext))  # join all text files into a single string

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", "".join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]  # encoder: take a string, output a list of integers


def decode(l):
    return "".join(
        [itos[i] for i in l]
    )  # decoder: take a list of integers, output a string


# create the train and test splits
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

# save the meta information as well, to help us encode/decode later
meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
