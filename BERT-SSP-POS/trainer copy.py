"""
Program Title: BERT POS Tagging Trainer for SSP and SOP, 
               with and without Data Augmentation (trainer.py)

Programmers:
    Emanuel Jamero
    John Nicolas Oandasan
    Vince Favorito
    Kyla Marie Alcantara

Date Written: Tue Oct 1 2024
Date Last Revised: Fri Nov 8 2024

Purpose:
    This program serves as a unified trainer for part-of-speech (POS) tagging models,
    specifically designed for Tagalog-Ilonggo text using BERT-based architectures.
    It is a central component in our tool for comparing different model 
    to determine the most accurate POS tagging model. The system evaluates four configurations: 
      - SSP (Same Sentence Prediction) with and without Data Augmentation
      - SOP (Sentence Order Prediction) with and without Data Augmentation

Data Structures:
    - Tokenizer: `BertTokenizer` initialized with "bert-base-cased"
    - Model Configurations: Dictionary and model class to configure and store SSP, SOP,
      and data augmentation settings.
    - Training Data: Augmented and non-augmented datasets for comparative model training.
    - Metrics: Evaluation metrics include precision, recall, F1 score, and accuracy,
      structured to assess model performance across different configurations.

Control Flow:
    1. Load and preprocess data with augmentation as per specified configuration.
    2. Initialize BERT model with selected SSP or SOP with or without Data augmentation objective.
    3. Train the model on the chosen configuration with specified hyperparameters.
    4. Evaluate model using precision, recall, F1 score, and accuracy.
    5. Log results for each configuration.
"""

import re
import torch
from transformers import BertTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print("Device: ", device)

# Input Files:
train_corpus = "./Dataset/Labeled Corpus/hil-train-set-less-tagset.txt"
val_corpus = "./Dataset/Labeled Corpus/hil-eval-set-less-tagset.txt"

vocab_file = "./Tokenizer/corpus-merged/uncased-vocab.txt"
tokenizer = BertTokenizer(
    vocab_file=vocab_file
)

bert_model = "./BERT-SSP/output_model_merged/checkpoint-331500/"
output_file = "./BERT-SSP-POS/predictions_output_merged_less_test.txt"
args_directory = "./BERT-SSP-POS/checkpoint_merged_less_test"
save_path = "./BERT-SSP-POS/confusion_matrix_merged_less_test.png"
epoch_number = 2
save_model = "./BERT-SSP-POS/BERTPOS_merged_less_test"

num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': ['[PUNCT]']})

pos_tag_mapping = {
    '[PAD]': 0,
    'NN': 1,
    'PRNN': 2,
    'DET' : 3,
    'VB' : 4,
    'ADJ' : 5,
    'ADV' : 6,
    'NUM' : 7,
    'CONJ' : 8,
    'PUNCT' : 9,
    'FW' : 10,
}

# Mapping of specific POS tags to their general categories
general_pos_mapping = {
    'NN': 'Noun',
    'PRNN': 'Pronoun',
    'DET': 'Determiner',
    'CONJ': 'Conjunction',
    'VB': 'Verb',
    'ADJ': 'Adjective',
    'ADV': 'Adverb',
    'NUM': 'Digit, Rank, Count', 
    'FW': 'Foreign Words', 
    'PUNCT': 'Punctuation'
}

num_labels = len(pos_tag_mapping)
id2label = {idx: tag for tag, idx in pos_tag_mapping.items()}
label2id = {tag: idx for tag, idx in pos_tag_mapping.items()}

def symbol2token(symbol):
    special_symbols = ['-', '&', "\"", "[", "]", "/", "$", "(", ")", "%", ":", "'", '.']
    # Check if the symbol is a comma
    if symbol == ',':
        return '[PUNCT] '

    elif symbol == '.':
        return '[PUNCT] '

    # Check if the symbol is in the list of special symbols
    elif symbol in special_symbols:
        return '[PUNCT] '

    # If the symbol is not a comma or in the special symbols list, keep it as it is
    return symbol

def preprocess_sentence(tagged_sentence):
    # Remove the line identifier (e.g., SNT.80188.3)
    sentence = re.sub(r'SNT\.\d+\.\d+\s+', '', tagged_sentence)
    special_symbols = ['-', '&', ",", "\"", "[", "]", "/", "$", "(", ")", "%", ":", "'", '.']
    # Construct the regex pattern for extracting words inside <TAGS> including special symbols
    special_symbols_regex = '|'.join([re.escape(sym) for sym in special_symbols])
    regex_pattern = r'<(?:[^<>]+? )?([a-zA-Z0-9.,&"!?{}]+)>'.format(special_symbols_regex)
    words = re.findall(regex_pattern, tagged_sentence)

    # Join the words to form a sentence
    sentence = ' '.join(words)
    sentence = sentence.lower()

    # Loop through the sentence and convert hyphen to '[PUNCT]' if the next character is a space
    new_sentence = ""
    i = 0
    while i < len(sentence):
        if any(sentence[i:].startswith(symbol) for symbol in special_symbols):
            if i + 2 < len(sentence) and sentence[i:i + 3] == '...':
                # Ellipsis found, replace with '[PUNCT]'
                new_sentence += symbol2token(sentence[i])
                i += 3
            elif i + 1 == len(sentence):
                new_sentence += symbol2token(sentence[i])
                break
            elif sentence[i + 1] == ' ' and i == 0:
                new_sentence += symbol2token(sentence[i])
                i += 1
            elif sentence[i - 1] == ' ' and sentence[i + 1] == ' ':
                new_sentence += symbol2token(sentence[i])
                i += 1
            elif sentence[i - 1] != ' ':
                new_sentence += ''
            else:
                word_after_symbol = ""
                while i + 1 < len(sentence) and sentence[i + 1] != ' ' and not any(
                        sentence[i + 1:].startswith(symbol) for symbol in special_symbols):
                    word_after_symbol += sentence[i + 1]
                    i += 1
                new_sentence += word_after_symbol
        elif any(sentence[i:].startswith(symbol) for symbol in special_symbols):
            if i + 1 < len(sentence) and (sentence[i + 1] == ' ' and sentence[i - 1] != ' '):
                new_sentence += '[PUNCT] '
                i += 1
            elif i + 1 == len(sentence):
                new_sentence += '[PUNCT] '
                break
            else:
                word_after_symbol = ""
                while i + 1 < len(sentence) and sentence[i + 1] != ' ' and not any(
                        sentence[i + 1:].startswith(symbol) for symbol in special_symbols):
                    word_after_symbol += sentence[i + 1]
                    i += 1
                new_sentence += word_after_symbol
        else:
            new_sentence += sentence[i]
        i += 1

    return new_sentence

def extract_tags(input_sentence):
    tags = re.findall(r'<([A-Z_]+)\s.*?>', input_sentence)
    return tags

def align_tokenization(sentence, tags):

    sentence = sentence.split()

    tokenized_sentence = tokenizer.tokenize(' '.join(sentence))
    if len(tags) > 12:
        """  """

    aligned_tagging = []
    current_word = ''
    index = 0

    for token in tokenized_sentence:
        if len(tags) > index:
            current_word += re.sub(r'^##', '', token)

            if sentence[index] == current_word:  # if we completed a word
                current_word = ''
                aligned_tagging.append(tags[index])
                index += 1
            else:  # otherwise insert padding
                aligned_tagging.append(0)

    decoded_tags = [list(pos_tag_mapping.keys())
                    [list(pos_tag_mapping.values()).index(tag_id)] 
                    for tag_id in aligned_tagging]

    aligned_tagging = [0] + aligned_tagging
    return tokenized_sentence, aligned_tagging

def process_tagged_sentence(tagged_sentence):

    sentence = preprocess_sentence(tagged_sentence)
    tags = extract_tags(tagged_sentence) # returns the tags

    encoded_tags = [pos_tag_mapping[tag] for tag in tags]

    # Align tokens
    tokenized_sentence, encoded_tags = align_tokenization(sentence, encoded_tags)
    encoded_sentence = tokenizer(sentence, padding='max_length', 
                                 truncation=True, max_length=128)

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = [1] * len(encoded_sentence['input_ids'])
    while len(encoded_sentence['input_ids']) < 128:
        encoded_sentence['input_ids'].append(0)  # Pad with zeros
        attention_mask.append(0)  # Pad attention mask


    while len(encoded_tags) < 128:
        encoded_tags.append(0)  # Pad with the ID of '[PAD]'

    encoded_sentence['encoded_tags'] = encoded_tags

    decoded_sentence = tokenizer.convert_ids_to_tokens(
        encoded_sentence['input_ids'], skip_special_tokens=False)

    decoded_tags = [list(pos_tag_mapping.keys())
                    [list(pos_tag_mapping.values()).index(tag_id)] 
                    for tag_id in encoded_tags]

    #
    word_tag_pairs = list(zip(decoded_sentence, decoded_tags))

    return encoded_sentence

def encode_corpus(input_file):

    encoded_sentences = []

    with open(input_file, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Processing corpus"):
        input_sentence = line.strip()

        encoded_sentence = process_tagged_sentence(input_sentence)
        encoded_sentences.append(encoded_sentence)

    return encoded_sentences

def createDataset(train_set, val_set, test_set=None):
    train_dataset_dict = {
        'input_ids': [],
        'attention_mask': [],
        'labels': [],
    }

    for entry in tqdm(train_set, desc="Converting training set"):
        train_dataset_dict['input_ids'].append(entry['input_ids'])
        train_dataset_dict['attention_mask'].append(entry['attention_mask'])
        train_dataset_dict['labels'].append(entry['encoded_tags'])

    train_dataset = Dataset.from_dict(train_dataset_dict)

    val_dataset_dict = {
        'input_ids': [],
        'attention_mask': [],
        'labels': [],
    }

    for entry in tqdm(val_set, desc="Converting validation set"):
        val_dataset_dict['input_ids'].append(entry['input_ids'])
        val_dataset_dict['attention_mask'].append(entry['attention_mask'])
        val_dataset_dict['labels'].append(entry['encoded_tags'])

    val_dataset = Dataset.from_dict(val_dataset_dict)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
    })

    if test_set is not None:
        test_dataset_dict = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
        }

        for entry in tqdm(test_set, desc="Converting test set"):
            test_dataset_dict['input_ids'].append(entry['input_ids'])
            test_dataset_dict['attention_mask'].append(entry['attention_mask'])
            test_dataset_dict['labels'].append(entry['encoded_tags'])

        test_dataset = Dataset.from_dict(test_dataset_dict)

        dataset_dict['test'] = test_dataset

    print("Dataset created.")
    return dataset_dict

train_corpus = encode_corpus(train_corpus)
val_corpus = encode_corpus(val_corpus)

encoded_dataset = createDataset(train_corpus, val_corpus)
max_token_length = 128
vocab_size = tokenizer.vocab_size

encoded_dataset.set_format("torch")

model = AutoModelForTokenClassification.from_pretrained(bert_model,
                                                           num_labels=num_labels,
                                                           id2label=id2label,
                                                           label2id=label2id)

model.resize_token_embeddings(len(tokenizer))

batch_size = 16
metric_name = "f1"

args = TrainingArguments(
    args_directory,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=3e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epoch_number,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    label_names=["labels"],
)

def compute_metrics(p):
    y_true = p.label_ids  # True labels
    y_pred = p.predictions.argmax(-1)  # Predicted labels

    y_true_flat = [tag_id for tags in y_true for tag_id in tags]
    y_pred_flat = [tag_id for tags in y_pred for tag_id in tags]

    # Calculate the confusion matrix (for logging, not returning)
    conf_matrix = confusion_matrix(
        y_true_flat, y_pred_flat, labels=list(label2id.values()))

    # Print the confusion matrix for manual inspection
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate precision, recall, and f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_flat, y_pred_flat, average='macro')

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

training = trainer.train()
results = trainer.evaluate()

# Get the predictions and true labels manually
predictions = trainer.predict(encoded_dataset["validation"])
y_true = predictions.label_ids
y_pred = predictions.predictions.argmax(-1)

# Flatten the labels
y_true_flat = []
y_pred_flat = []

for true_labels, pred_labels in zip(y_true, y_pred):
    for true_label, pred_label in zip(true_labels, pred_labels):
        if true_label != label2id['[PAD]']:  # Ignore padding tokens
            y_true_flat.append(true_label)
            y_pred_flat.append(pred_label)

# Convert IDs to labels using id2label dictionary
y_true_labels = [id2label[tag_id] for tag_id in y_true_flat]
y_pred_labels = [id2label[tag_id] for tag_id in y_pred_flat]

# Map the specific POS tags to general categories
def map_to_general_pos(labels, mapping):
    return [mapping.get(label, label) for label in labels]

# Flatten the labels and apply the general POS mapping
y_true_flat_general = map_to_general_pos(
    [id2label[tag_id] for tag_id in y_true_flat], 
    general_pos_mapping)

y_pred_flat_general = map_to_general_pos(
    [id2label[tag_id] for tag_id in y_pred_flat], 
    general_pos_mapping)

# Print each sentence with its corresponding predictions 
# for each word and write to a text file
def print_sentences_with_predictions_to_file(
        validation_dataset, 
        y_true_filtered, 
        y_pred_filtered, 
        output_file=output_file):
    with open(output_file, "w") as f:
        for i in range(len(validation_dataset)):
            # Decode the original sentence
            input_ids = validation_dataset[i]["input_ids"]
            original_sentence_tokens = tokenizer.convert_ids_to_tokens(input_ids)

            # Skip special tokens like [CLS], [SEP], and [PAD]
            original_sentence_tokens = [
                token for token in original_sentence_tokens 
                if token not in ["[CLS]", "[SEP]", "[PAD]"]
            ]

            # Get the true and predicted labels for the sentence
            true_labels = y_true_filtered[
                i * len(original_sentence_tokens): 
                (i + 1) * len(original_sentence_tokens)
            ]
            pred_labels = y_pred_filtered[
                i * len(original_sentence_tokens): 
                (i + 1) * len(original_sentence_tokens)
            ]

            # Remove padding tokens from the sentence
            filtered_sentence_tokens, filtered_true_labels, filtered_pred_labels = [], [], []
            for token, true_label, pred_label in zip(
                original_sentence_tokens, 
                true_labels, 
                pred_labels):
                if true_label != '[PAD]' and pred_label != '[PAD]':
                    filtered_sentence_tokens.append(token)
                    filtered_true_labels.append(true_label)
                    filtered_pred_labels.append(pred_label)

            # Write each word with its corresponding true and predicted labels to the file
            f.write(f"Sentence {i + 1}:\n")
            for token, true_label, pred_label in zip(
                filtered_sentence_tokens, 
                filtered_true_labels, 
                filtered_pred_labels):
                f.write(f"Word: {token}\tTrue Label: {true_label}\tPredicted Label: {pred_label}\n")
            f.write("\n")

# Get unique general POS categories for the confusion matrix
unique_labels = sorted(set(general_pos_mapping.values()))

# Create confusion matrix
conf_matrix_general = confusion_matrix(y_true_flat_general, y_pred_flat_general, labels=unique_labels)

# Plot the confusion matrix as a heatmap
def plot_confusion_matrix(conf_matrix, labels, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (General POS Categories)")
    # Save the figure to a file
    plt.savefig(save_path)
    plt.show()

def calculate_pos_metrics(conf_matrix, labels, output_file="pos_metrics.txt"):
    """Calculate precision, recall, and F1 score from confusion matrix"""
    metrics = []
    
    # Calculate metrics for each class from confusion matrix
    for i, label in enumerate(labels):
        true_positives = conf_matrix[i, i]
        false_positives = conf_matrix[:, i].sum() - true_positives
        false_negatives = conf_matrix[i, :].sum() - true_positives
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.append({
            'Category': label,
            'Precision': round(precision, 3),
            'Recall': round(recall, 3),
            'F-Measure': round(f1, 3)
        })
    
    # Convert to DataFrame for easier formatting
    metrics_df = pd.DataFrame(metrics)
    
    # Write results to file
    with open(output_file, 'w') as f:
        f.write("POS Tagging Metrics Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        # Write header
        f.write(f"{'Category':<15} {'Precision':<12} {'Recall':<12} {'F-Measure':<12}\n")
        f.write("-" * 51 + "\n")
        
        # Write each row
        for _, row in metrics_df.iterrows():
            f.write(f"{row['Category']:<15} {row['Precision']:<12} {row['Recall']:<12} {row['F-Measure']:<12}\n")
        
        # Calculate and write overall metrics
        f.write("\nOverall Metrics:\n")
        f.write("-" * 51 + "\n")
        overall_precision = metrics_df['Precision'].mean()
        overall_recall = metrics_df['Recall'].mean()
        overall_f1 = metrics_df['F-Measure'].mean()
        
        f.write(f"{'Average':<15} {overall_precision:.3f}{'':8} {overall_recall:.3f}{'':8} {overall_f1:.3f}\n")
    
    return metrics_df

# Calculate metrics directly from your confusion matrix
metrics_df = calculate_pos_metrics(conf_matrix_general, unique_labels, "pos_metrics.txt")
print("POS metrics have been saved to pos_metrics.txt")

# Plot the confusion matrix for general POS categories
plot_confusion_matrix(conf_matrix_general, unique_labels, save_path)

print("Evaluation: ", results)
trainer.save_model(save_model)
print(results)