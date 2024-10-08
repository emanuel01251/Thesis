import pickle
import os

current_dir = os.getcwd()
cache_file_path = os.path.join(current_dir, "Dataset", "Unlabeled Corpus", "cached_sop_BertTokenizerFast_512_New Hiligaynon Corpus.txt")

# Load the cached data
with open(cache_file_path, 'rb') as cache_file:
    cached_data = pickle.load(cache_file)

""" print(type(cached_data))
print(cached_data[0]) """

##### To decode the IDs to text

from transformers import BertTokenizerFast

vocab_file = "./BERT-SSP/tokenizer-corpus-hiligaynon/uncased-vocab.txt"
merges_file = "./BERT-SSP/tokenizer-corpus/cased.json"  # If merges_file is required (optional for BERT)
""" tokenizer = BertTokenizerFast(
    vocab_file=vocab_file,
    tokenizer_file=merges_file
) """
tokenizer = BertTokenizerFast(
    vocab_file=vocab_file
)

# Example: cached_data[0] is the first example, containing input_ids and same_sentence_label
example = cached_data[1]

# Extract input_ids and same_sentence_label
input_ids = example['input_ids'].tolist()
same_sentence_label = example['sentence_order_label'].item()  # Assuming it's a tensor, convert to int

# Decode the token IDs back to text (including special tokens for clarity)
tokens = tokenizer.convert_ids_to_tokens(input_ids)

# Print the same_sentence_label first
print(f"Same Sentence Label: {same_sentence_label}\n")

# Print the input_ids and corresponding tokens together
print("Input IDs and Corresponding Tokens:")
for token_id, token in zip(input_ids, tokens):
    print(f"ID: {token_id} -> Token: {token}")
