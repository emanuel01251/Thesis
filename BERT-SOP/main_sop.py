from transformers import BertConfig, BertTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import BertForPreTraining, LineByLineWithSOPTextDataset, AlbertConfig, AlbertForPreTraining

# Load your custom tokenizer
tokenizer_path = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

import os

current_dir = os.getcwd()
current_dir += "\BERT-SOP"
texts = os.path.join(current_dir, "src")

dataset = LineByLineWithSOPTextDataset(
    tokenizer=tokenizer,
    file_dir=texts,
    block_size=512
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=0.15
)

config = AlbertConfig.from_pretrained('bert-base-uncased')
model = AlbertForPreTraining(config)

training_args = TrainingArguments(
    output_dir=".\BERT-SOP\output_model",
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
    train_dataset=dataset,
)

trainer.train()


