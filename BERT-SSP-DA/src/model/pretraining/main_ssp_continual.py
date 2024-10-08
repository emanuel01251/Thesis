from transformers import BertConfig, BertTokenizer, BertTokenizerFast
from bert_with_ssp_head import BertForPreTrainingMLMAndSSP
from ssp_dataset import TextDatasetForSameSentencePrediction
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoTokenizer
import os

vocab_file = "./BERT-SSP-DA/tokenizer-corpus-hiligaynon/uncased-vocab.txt"
merges_file = "./BERT-SSP/tokenizer-corpus/cased.json"  # If merges_file is required (optional for BERT)
""" tokenizer = BertTokenizerFast(
    vocab_file=vocab_file,
    tokenizer_file=merges_file
) """
tokenizer = BertTokenizerFast(
    vocab_file=vocab_file
)

# Define your dataset
current_dir = os.getcwd()
texts = os.path.join(current_dir, "Data Augmentation", "EDA", "Unlabeled Corpus", "New Hiligaynon Corpus augmented_output.txt")

dataset = TextDatasetForSameSentencePrediction(
    tokenizer=tokenizer,
    file_path=texts,
    block_size=512, 
    overwrite_cache=False, 
    ssp_probability=0.5
)

# Data collator for masked language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=0.15
)

# Load the pretrained model from the directory where it was saved
pretrained_model_dir = "gklmip/bert-tagalog-base-uncased"

model = BertForPreTrainingMLMAndSSP.from_pretrained(pretrained_model_dir)

# Define new training arguments
training_args = TrainingArguments(
    output_dir="./BERT-SSP-DA/output_model_continual",  # Directory where the model checkpoints will be saved
    num_train_epochs=10,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    save_strategy="steps", 
    save_steps=100,
    save_total_limit=2,
    save_safetensors=False,
    no_cuda=False,  # Use GPU if available
)

# Create a Trainer instance with the loaded model and dataset
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

""" checkpoint_dir = "./BERT-SSP/output_model_tagalog/checkpoint-65000"  # Replace with the correct checkpoint step
resume_from_checkpoint=checkpoint_dir """
trainer.train()
