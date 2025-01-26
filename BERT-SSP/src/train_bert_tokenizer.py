"""
Program Title: Custom Tokenizer Training for BERT (train_tokenizer.py)
Programmers:
    - Emanuel Jamero
    - John Nicolas Oandasan
    - Vince Favorito
    - Kyla Marie Alcantara

Date Written: Tue Oct 1 2024
Last Revised: Fri Nov 8 2024

Purpose:
    To train and save a WordPiece tokenizer configured for Tagalog/Ilonggo, 
    using parameters from a JSON configuration file.

Algorithms:
  - Tokenizer Training: Uses `BertWordPieceTokenizer.train()` with parameters such 
    as vocabulary size and minimum frequency to generate a custom tokenizer for 
    the specified corpus.

Control Flow:
  - JSON Configuration: `read_json_file()` loads the tokenizer settings from a JSON 
    file.
  - Saving Model: After training, `tokenizer.save_model()` and `tokenizer.save()` 
    save the tokenizer's vocabulary and configuration to specified paths for 
    reuse with future pretraining.
"""

from utils.file_utils import read_json_file

tokenizer_config = read_json_file("./BERT-SSP/src/tokenizer.json")

from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=tokenizer_config["lowercase"],
)

trainer = tokenizer.train(
    tokenizer_config["input_file"],
    vocab_size=tokenizer_config["vocabulary_size"],
    min_frequency=tokenizer_config["minimum_frequency"],
    show_progress=True,
    special_tokens=tokenizer_config["special_tokens"],
    limit_alphabet=tokenizer_config["alphabet_limit"],
    wordpieces_prefix="##"
)

tokenizer.save_model(tokenizer_config["output_path"], tokenizer_config["output_name"])
tokenizer.save("{}/{}.json".format(tokenizer_config["output_path"], 
                                   tokenizer_config["output_name"]), pretty=True)