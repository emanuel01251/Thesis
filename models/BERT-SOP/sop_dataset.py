import json
import os
import pickle
import random
import time
import warnings
from typing import Dict, List, Optional
import time

import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

class TextDatasetForSentenceOrderPrediction(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            file_path: str,
            block_size: int,
            overwrite_cache=False,
            sop_probability=0.5,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        self.sop_probability = sop_probability

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            f"cached_sop_{tokenizer.__class__.__name__}_{block_size}_{filename}",
        )

        self.tokenizer = tokenizer

        # Lock file to ensure that only the first process processes the dataset and caches it.
        lock_path = cached_features_file + ".lock"

        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.documents = [[]]
                with open(file_path, encoding="utf-8") as f:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        line = line.strip()

                        # Empty lines are used as document delimiters (new document)
                        if not line and len(self.documents[-1]) != 0:
                            self.documents.append([])

                        tokens = tokenizer.tokenize(line)
                        tokens = tokenizer.convert_tokens_to_ids(tokens)
                        if tokens:
                            self.documents[-1].append(tokens)

                logger.info(f"Creating examples from {len(self.documents)} documents.")
                self.examples = []
                for doc_index, document in enumerate(self.documents):
                    self.create_examples_from_document(document, doc_index, block_size)

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    def create_examples_from_document(self, document: List[List[int]], doc_index: int, block_size: int):
        """Creates examples for a single document for the SOP task."""
        i = 0
        sentences_in_document = len(document)

        while i < sentences_in_document - 1:
            sentence_a = document[i]
            sentence_b = document[i + 1]

            if random.random() < self.sop_probability:
                # 50% chance to swap sentence order (for negative examples)
                is_in_correct_order = False
                sentence_a, sentence_b = sentence_b, sentence_a
            else:
                # Correct order
                is_in_correct_order = True

            # Truncate sentences to fit within block_size with special tokens
            max_num_tokens = block_size - self.tokenizer.num_special_tokens_to_add(pair=True)
            truncate_seq_pair(sentence_a, sentence_b, max_num_tokens)

            if len(sentence_a) > 0 and len(sentence_b) > 0:
                # Prepare input features
                self.examples.append(self.create_example(sentence_a, sentence_b, is_in_correct_order))

            i += 1

    def create_example(self, seq_a, seq_b, is_in_correct_order):
        # Add special tokens and create input ids
        input_ids = self.tokenizer.build_inputs_with_special_tokens(seq_a, seq_b)

        # Truncate input_ids if it exceeds block_size
        if len(input_ids) > 512:
            input_ids = input_ids[:512]

        # Token type ids: 0 for seq_a, 1 for seq_b
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(seq_a, seq_b)

        # Truncate token_type_ids if needed
        if len(token_type_ids) > 512:
            token_type_ids = token_type_ids[:512]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "sentence_order_label": torch.tensor(0 if is_in_correct_order else 1, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()