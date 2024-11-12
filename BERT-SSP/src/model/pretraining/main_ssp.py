from transformers import BertConfig, BertTokenizer, BertTokenizerFast
from bert_with_ssp_head import BertForPreTrainingMLMAndSSP
from ssp_dataset import TextDatasetForSameSentencePrediction
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoTokenizer


""" vocab_file = "./BERT-SSP/tokenizer-corpus-tagalog/uncased-vocab.txt" """
vocab_file = "./BERT-SSP/tokenizer-corpus-hiligaynon/uncased-vocab.txt"
""" tokenizer = BertTokenizerFast(
    vocab_file=vocab_file,
    tokenizer_file=merges_file
) """
tokenizer = BertTokenizerFast(
    vocab_file=vocab_file
)

import os


current_dir = os.getcwd()
current_dir += "\Dataset"
texts = os.path.join(current_dir, "Unlabeled Corpus", "New Hiligaynon Corpus.txt")

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

config = BertConfig.from_pretrained('./BERT-SSP/output_model_tagalog/checkpoint-478000')
""" config.vocab_size = len(tokenizer) """

""" # Optional: Save the modified config if needed
config.save_pretrained('path_to_save_modified_config') """
model = BertForPreTrainingMLMAndSSP(config)

# Define new training arguments
training_args = TrainingArguments(
    output_dir="./BERT-SSP/output_model_tagalog_hiligaynon3",  # Directory where the model checkpoints will be saved
    num_train_epochs=80,
    per_device_train_batch_size=8,
    learning_rate=1e-5, #1e-5 is next
    save_strategy="steps", 
    save_steps=1000,
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

checkpoint_dir = "./BERT-SSP/output_model_tagalog/checkpoint-478000"  # Replace with the correct checkpoint step
""" resume_from_checkpoint=checkpoint_dir """
trainer.train()