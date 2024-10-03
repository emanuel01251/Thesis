from transformers import BertConfig, BertTokenizer
from bert_with_ssp_head import BertForPreTrainingMLMAndSSP
from ssp_dataset import TextDatasetForSameSentencePrediction
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, BertTokenizerFast

# Load your custom tokenizer
""" tokenizer_path = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(tokenizer_path) """

# Load your custom tokenizer from your own files
vocab_file = "./BERT-SSP/tokenizer-corpus/cased-vocab.txt"
merges_file = "./BERT-SSP/tokenizer-corpus/cased.json"  # If merges_file is required (optional for BERT)
tokenizer = BertTokenizerFast(
    vocab_file=vocab_file, 
    tokenizer_file=merges_file, 
    model_max_length=512,
    truncation=True
)


import os


current_dir = os.getcwd()
current_dir += "\Dataset"
texts = os.path.join(current_dir, "Unlabeled Corpus", "Merged Corpus.txt")

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
    
config = BertConfig.from_pretrained('bert-base-cased')
model = BertForPreTrainingMLMAndSSP(config)

training_args = TrainingArguments(
    output_dir=".\BERT-SSP\output_model",
    num_train_epochs=25,
    per_device_train_batch_size=8,
    learning_rate=1e-4,
    save_strategy="steps", 
    save_steps=30000,
    save_total_limit=2,
    save_safetensors=False,  # Disable safe serialization to avoid weight-sharing issues
    no_cuda=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()


