from transformers import BertConfig, BertTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import BertForPreTraining, LineByLineWithSOPTextDataset, AlbertConfig, AlbertForPreTraining, BertTokenizerFast, AutoTokenizer
from sop_dataset import TextDatasetForSentenceOrderPrediction
from bert_with_sop_head import BertForPreTrainingMLMandSOP
import os

""" vocab_file = "./BERT-SSP/tokenizer-corpus-tagalog/uncased-vocab.txt" """
vocab_file = "./BERT-SSP/tokenizer-corpus-hiligaynon/uncased-vocab.txt"
tokenizer = BertTokenizerFast(
    vocab_file=vocab_file
)

# Define your dataset
current_dir = os.getcwd()
texts = os.path.join(current_dir, "Dataset", "Unlabeled Corpus", "New Hiligaynon Corpus.txt")

""" current_dir = os.getcwd()
current_dir += "\BERT-SOP"
texts = os.path.join(current_dir, "src") """

dataset = TextDatasetForSentenceOrderPrediction(
    tokenizer=tokenizer,
    file_path=texts,
    block_size=512, 
    overwrite_cache=False, 
    sop_probability=0.5
)
""" dataset = LineByLineWithSOPTextDataset(
    tokenizer=tokenizer,
    file_dir=texts,
    block_size=512
) """

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=0.15
)

""" pretrained_model_dir = "gklmip/bert-tagalog-base-uncased" """
pretrained_model_dir = "./BERT-SOP/output_model_tagalog/checkpoint-97000"
""" pretrained_model_dir.vocab_size = len(tokenizer) """
model = BertForPreTrainingMLMandSOP.from_pretrained(pretrained_model_dir)

training_args = TrainingArguments(
    output_dir=".\BERT-SOP\output_model_tagalog_hiligaynon",
    num_train_epochs=25,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    save_strategy="steps", 
    save_steps=1000,
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

checkpoint_dir = "./BERT-SOP/output_model_tagalog/checkpoint-97000"
""" resume_from_checkpoint=checkpoint_dir """
trainer.train()
