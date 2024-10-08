from transformers import AutoTokenizer

# Load tokenizer from the pretrained model directly
tokenizer = AutoTokenizer.from_pretrained("jcblaise/bert-tagalog-base-uncased")
tokenizer.save_pretrained('./BERT-SOP/output_model/checkpoint-24025/')