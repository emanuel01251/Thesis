"""
Program Title: BERT Pretraining with Same Sentence Prediction (main_ssp.py)
Programmers: 
    - Emanuel Jamero
    - John Nicolas Oandasan
    - Vince Favorito
    - Kyla Marie Alcantara

Date Written: Tue Oct 1 2024
Last Revised: Fri Nov 8 2024

Purpose:
    The purpose of this program is to pretrain a BERT model using a dataset of 
    Tagalog/Ilonggo texts with the addition of SSP for better contextual understanding.

Data Structures, Algorithms, and Control:
    - Data Structures:
        - `BertTokenizer`: Tokenizes Tagalog/Ilonggo corpus data using a vocabulary file located 
          at "./Tokenizer/corpus-merged/uncased-vocab.txt".
        - `TextDatasetForSameSentencePrediction`: A custom dataset class that formats input 
          text data with a `ssp_probability` parameter, determining the likelihood that 
          a sample will involve same sentence prediction.
        - `DataCollatorForLanguageModeling`: Prepares input batches with masked tokens for 
          MLM using a 15% mask probability.
    
    - Algorithms:
        - Masked Language Modeling (MLM) and Same Sentence Prediction (SSP): The model 
          simultaneously learns to predict masked words and determine if sentence pairs 
          are consecutive, improving language representation.
        - `Trainer` API: Handles the training loop, including batch processing, gradient 
          computation, and optimization steps.

    - Control Flow:
        - `TrainingArguments`: Specifies parameters for training, such as `num_train_epochs`, 
          `per_device_train_batch_size`, `learning_rate`, and checkpointing frequency.
        - Checkpointing: Saves model checkpoints at 500 steps and limits saved checkpoints to 
          2, enabling resumption from the specified checkpoint directory if interrupted.
        - Training Execution: `trainer.train()` initiates the training loop, using `dataset` 
          and `data_collator` with custom arguments to manage training duration, checkpointing, 
          and output.
"""

from transformers import BertConfig, BertTokenizer
from bert_with_ssp_head import BertForPreTrainingMLMAndSSP
from ssp_dataset import TextDatasetForSameSentencePrediction
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

vocab_file = "./Tokenizer/corpus-merged/uncased-vocab.txt"
tokenizer = BertTokenizer(
    vocab_file=vocab_file
)

import os

current_dir = os.getcwd()
texts = os.path.join(current_dir, "Dataset", "Unlabeled Corpus", "Merged Corpus.txt")

dataset = TextDatasetForSameSentencePrediction(
    tokenizer=tokenizer,
    file_path=texts,
    block_size=512, 
    overwrite_cache=False, 
    ssp_probability=0.5
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=0.15
)

config = BertConfig.from_pretrained('gklmip/bert-tagalog-base-uncased')
model = BertForPreTrainingMLMAndSSP.from_pretrained('gklmip/bert-tagalog-base-uncased')

# Define new training arguments
training_args = TrainingArguments(
    output_dir="./BERT-SSP/output_model_merged",  # Directory where the model checkpoints will be saved
    num_train_epochs=30,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    save_strategy="steps", 
    save_steps=500,
    save_total_limit=2,
    save_safetensors=False,
    no_cuda=False,  # Use GPU if available
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

checkpoint_dir = "./BERT-SSP/output_model_merged/checkpoint-221000"  # Replace with the correct checkpoint step
""" resume_from_checkpoint=checkpoint_dir """
trainer.train(resume_from_checkpoint=checkpoint_dir)