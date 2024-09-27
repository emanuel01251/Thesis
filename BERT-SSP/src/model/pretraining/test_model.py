from transformers import BertConfig, BertTokenizer
from bert_with_ssp_head import BertForPreTrainingMLMAndSSP  # Your custom model class
import torch

# Step 1: Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')  # Use your tokenizer path

# Step 2: Load the model configuration
config = BertConfig.from_pretrained('output_dir_folder/checkpoint-8/config.json')

# Step 3: Load the trained model weights
model = BertForPreTrainingMLMAndSSP.from_pretrained(
    'output_dir_folder/checkpoint-8',  # Path to checkpoint folder containing pytorch_model.bin
    config=config
)

# Step 4: Move the model to the correct device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()  # Set the model to evaluation mode

# Sample input for testing
text = "This is a sample sentence for testing."

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)

# Move inputs to the same device as the model
inputs = {key: value.to(device) for key, value in inputs.items()}

# Step 5: Perform inference
with torch.no_grad():  # No need to calculate gradients for inference
    outputs = model(**inputs)

print(inputs)
exit()

predicted_ids = torch.argmax(outputs, dim=-1)

# Depending on the task, extract the relevant predictions
# For example, if you're working with a pretraining head with MLM + SSP, you'll need to interpret the outputs accordingly
mlm_logits = outputs.mlm_logits  # For masked language model predictions

ssp_logits = outputs.ssp_logits  # For Same-Sentence Prediction logits (if applicable)

# If you are predicting masked tokens (MLM), find the most likely tokens
predicted_ids = torch.argmax(mlm_logits, dim=-1)

# Decode the predicted token IDs back into words
predicted_tokens = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

print(f"Predicted tokens: {predicted_tokens}")
