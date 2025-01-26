"""
Program Title: BERT Pretraining with Sentence Order Prediction (main_sop.py)
Programmers: 
    - Emanuel Jamero
    - John Nicolas Oandasan
    - Vince Favorito
    - Kyla Marie Alcantara

Date Written: Tue Oct 1 2024
Last Revised: Fri Nov 8 2024

Purpose:
    The purpose of this program is to pretrain a BERT model on a Tagalog/Ilonggo 
    corpus with sentence order prediction to better capture contextual dependencies.

Data Structures, Algorithms, and Control:
    - Data Structures:
        - `BertTokenizer`: Tokenizes the Tagalog/Ilonggo corpus using a custom vocabulary file.
        - `TextDatasetForSentenceOrderPrediction`: Custom dataset class handling SOP 
          tasks, formatting input text data with a specified SOP probability.
        - `DataCollatorForLanguageModeling`: Prepares MLM batches with a 15% probability 
          of token masking.

    - Algorithms:
        - Masked Language Modeling (MLM) and Sentence Order Prediction (SOP): The model 
          learns to predict masked words and distinguish correct sentence order, improving 
          contextualized language representations.
        - `Trainer` API: Manages the training process, handling batch processing, gradient 
          computation, and optimization steps.

    - Control Flow:
        - `TrainingArguments`: Sets training configurations, including batch size, learning rate, 
          checkpoint frequency, and serialization options.
        - Checkpointing: Configured to save model checkpoints every 500 steps with a limit of 2.
        - Training Execution: `trainer.train()` initiates the training loop, using `dataset` 
          and `data_collator` with specified arguments to manage model training, checkpointing, 
          and output location.
"""

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, BertTokenizer
from sop_dataset import TextDatasetForSentenceOrderPrediction
from bert_with_sop_head import BertForPreTrainingMLMandSOP
import os

vocab_file = "./Tokenizer/corpus-merged/uncased-vocab.txt"
tokenizer = BertTokenizer(
    vocab_file=vocab_file
)

# Define your dataset
current_dir = os.getcwd()
texts = os.path.join(current_dir, "Dataset", "Unlabeled Corpus", "Merged Corpus.txt")

dataset = TextDatasetForSentenceOrderPrediction(
    tokenizer=tokenizer,
    file_path=texts,
    block_size=512, 
    overwrite_cache=False, 
    sop_probability=0.5
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=0.15
)

pretrained_model_dir = "gklmip/bert-tagalog-base-uncased"
model = BertForPreTrainingMLMandSOP.from_pretrained(pretrained_model_dir)

training_args = TrainingArguments(
    output_dir=".\BERT-SOP\output_model_merged",
    num_train_epochs=30,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    save_strategy="steps", 
    save_steps=500,
    save_total_limit=2,
    save_safetensors=False,  # Disable safe serialization to avoid weight-sharing issues
    no_cuda=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

checkpoint_dir = "./BERT-SOP/output_model_merged/checkpoint-131500"
""" resume_from_checkpoint=checkpoint_dir """
trainer.train(resume_from_checkpoint=checkpoint_dir)
