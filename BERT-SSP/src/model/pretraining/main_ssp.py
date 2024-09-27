from transformers import BertConfig, BertTokenizer
from bert_with_ssp_head import BertForPreTrainingMLMAndSSP
from ssp_dataset import TextDatasetForSameSentencePrediction
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import BertForPreTraining

# Load your custom tokenizer
tokenizer_path = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

import os


current_dir = os.getcwd()
current_dir += "\BERT-SSP"
texts = os.path.join(current_dir, "src", "sample_text.txt")

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

config = BertConfig.from_pretrained('bert-base-uncased')
model = BertForPreTrainingMLMAndSSP(config)

training_args = TrainingArguments(
    output_dir=".\BERT-SSP\output_model",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    save_strategy="steps", 
    save_steps=30000,
    save_total_limit=2,
    save_safetensors=False  # Disable safe serialization to avoid weight-sharing issues
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()


