import nltk
from transformers import AutoTokenizer
import numpy as np

nltk.download('punkt')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Add padding token and resize token embeddings
# tokenizer.add_special_tokens({"pad_token": "<pad>"})
# pad_id = tokenizer.pad_token_id
pad_id = tokenizer.bos_token_id

def chunk_text(file_path, block_size=4096):
    # Read text file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split text into sentences
    sentences = nltk.tokenize.sent_tokenize(text)

    current_chunk = []
    current_length = 0
    for sentence in sentences:
        tokens = tokenizer.encode(sentence, add_special_tokens=False)
        token_length = len(tokens)

        if current_length + token_length > block_size:
            # Pad the current chunk if it's not the block size
            current_chunk.extend([pad_id] * (block_size - current_length))
            yield current_chunk
            current_chunk = tokens
            current_length = token_length
        else:
            current_chunk.extend(tokens)
            current_length += token_length

    # Pad and yield the last chunk if it's not empty
    if current_chunk:
        current_chunk.extend([pad_id] * (block_size - len(current_chunk)))
        yield current_chunk

# Example: process and save chunks
chunks = list(chunk_text("datasets/ms_harrypotter_data/raw_data.txt"))

# Save chunks to npy file
np.save("datasets/ms_harrypotter_data/rawdata_chunkedtokenized.npy", chunks)