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
#print("Device: ", device)

# Input Files:
train_corpus = "./Dataset/Labeled Corpus/tl_augmented_dataset.txt"
val_corpus = "./Dataset/Labeled Corpus/tl-hil-eval-set-less-tagset.txt"

vocab_file = "./Tokenizer/corpus-merged/uncased-vocab.txt"
tokenizer = BertTokenizer(
    vocab_file=vocab_file
)

bert_model = "./BERT-SOP/output_model_merged/checkpoint-197500/"
output_file = "./BERT-SOP-DA-POS/predictions_output_tl.txt"
output_file_sentence = "./BERT-SOP-DA-POS/sentence_predictions_output_tl.txt"
args_directory = "./BERT-SOP-DA-POS/checkpoint_tl"
save_path = "./BERT-SOP-DA-POS/confusion_matrix_tl.png"
epoch_number = 5
save_model = "./BERT-SOP-DA-POS/BERTPOS_tl"

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
    'LM' : 11,
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
    'LM': 'Lexical Marker',
    'PUNCT' : 'Punctuation'
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

    #print("Sentence \n: ", sentence)
    sentence = sentence.split()
    #print("Sentence Split\n: ", sentence)

    tokenized_sentence = tokenizer.tokenize(' '.join(sentence))
    tokenized_sentence_string = " ".join(tokenized_sentence)
    #print("ID2Token_string\n: ", tokenized_sentence_string)
    #print("Tags\n: ", [id2label[tag_id] for tag_id in tags])
    if len(tags) > 12:
        print(id2label[tags[11]])

    aligned_tagging = []
    current_word = ''
    index = 0

    for token in tokenized_sentence:
        if len(tags) > index:
            current_word += re.sub(r'^##', '', token)
            # print("Current word after replacing ##: ", current_word)
            # print("sentence[index]: ", sentence[index])

            if sentence[index] == current_word:  # if we completed a word
                #print("completed a word: ", current_word)
                current_word = ''
                aligned_tagging.append(tags[index])
                # print(f"Tag of index {index}: ", id2label[tags[index]])
                # print(f"Aligned tag of index {index}: ", (id2label[aligned_tagging[-1]]))
                # print("Tags1\n: ", [id2label[tag_id] for tag_id in tags])
                # print("Tags2\n: ", [id2label[tag_id] for tag_id in aligned_tagging])
                # print(f"{index+1}/{len(tags)} tags consumed")
                index += 1
            else:  # otherwise insert padding
                #print("incomplete word: ", current_word)
                aligned_tagging.append(0)

            #print("---")

    decoded_tags = [list(pos_tag_mapping.keys())[list(pos_tag_mapping.values()).index(tag_id)] for tag_id in
                    aligned_tagging]

    # print("Tokenized Sentence\n: ", tokenized_sentence)
    # print("Tokenized Len\n: ", len(tokenized_sentence))
    # print("Tags\n: ", decoded_tags)
    # print("Tags Count\n: ", len(decoded_tags))

    assert len(tokenized_sentence) == len(aligned_tagging)

    aligned_tagging = [0] + aligned_tagging
    return tokenized_sentence, aligned_tagging


def process_tagged_sentence(tagged_sentence):
    # print(tagged_sentence)

    sentence = preprocess_sentence(tagged_sentence)
    tags = extract_tags(tagged_sentence) # returns the tags (eto ilagay mo sa tags.txt)


    encoded_tags = [pos_tag_mapping[tag] for tag in tags]

    # Align tokens
    tokenized_sentence, encoded_tags = align_tokenization(sentence, encoded_tags)
    encoded_sentence = tokenizer(sentence, padding="max_length" ,truncation=True, max_length=128)

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = [1] * len(encoded_sentence['input_ids'])
    #print("len(encoded_sentence['input_ids']):", len(encoded_sentence['input_ids']))
    while len(encoded_sentence['input_ids']) < 128:
        encoded_sentence['input_ids'].append(0)  # Pad with zeros
        attention_mask.append(0)  # Pad attention mask


    while len(encoded_tags) < 128:
        encoded_tags.append(0)  # Pad with the ID of '[PAD]'

    encoded_sentence['encoded_tags'] = encoded_tags

    decoded_sentence = tokenizer.convert_ids_to_tokens(encoded_sentence['input_ids'], skip_special_tokens=False)

    decoded_tags = [list(pos_tag_mapping.keys())[list(pos_tag_mapping.values()).index(tag_id)] for tag_id in
                    encoded_tags]

    #
    word_tag_pairs = list(zip(decoded_sentence, decoded_tags))
    #print(encoded_sentence)
    #print("Sentence:", decoded_sentence)
    #print("Tags:", decoded_tags)
    #print("Decoded Sentence and Tags:", word_tag_pairs)
    #print("---")

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

    #print("Dataset created.")
    return dataset_dict

def analyze_mismatches(validation_output):
    """
    Analyze and count mismatches in validation predictions
    Returns dictionary with mismatch statistics
    """
    # Unpack the validation output
    predictions = validation_output.predictions.argmax(-1)  # Get predicted labels
    true_labels = validation_output.label_ids  # Get true labels
    
    mismatch_stats = {
        'total_tokens': 0,
        'total_mismatches': 0,
        'mismatch_types': {},
        'tag_accuracy': {}
    }
    
    # Initialize accuracy counters for each tag
    for tag in pos_tag_mapping.keys():
        mismatch_stats['tag_accuracy'][tag] = {
            'correct': 0,
            'total': 0
        }
    
    # Analyze each prediction
    for true_sent, pred_sent in zip(true_labels, predictions):
        for true_label, pred_label in zip(true_sent, pred_sent):
            # Skip padding tokens
            if true_label == pos_tag_mapping['[PAD]']:
                continue
                
            mismatch_stats['total_tokens'] += 1
            
            # Get tag names
            true_tag = id2label[true_label]
            pred_tag = id2label[pred_label]
            
            # Update tag accuracy
            mismatch_stats['tag_accuracy'][true_tag]['total'] += 1
            if true_label == pred_label:
                mismatch_stats['tag_accuracy'][true_tag]['correct'] += 1
            else:
                mismatch_stats['total_mismatches'] += 1
                
                # Track mismatch type
                mismatch_type = f'{true_tag}->{pred_tag}'
                mismatch_stats['mismatch_types'][mismatch_type] = \
                    mismatch_stats['mismatch_types'].get(mismatch_type, 0) + 1
    
    # Calculate percentage of mismatches
    mismatch_stats['mismatch_percentage'] = (
        mismatch_stats['total_mismatches'] / mismatch_stats['total_tokens'] * 100
        if mismatch_stats['total_tokens'] > 0 else 0
    )
    
    # Calculate per-tag accuracy
    for tag in mismatch_stats['tag_accuracy']:
        total = mismatch_stats['tag_accuracy'][tag]['total']
        correct = mismatch_stats['tag_accuracy'][tag]['correct']
        accuracy = (correct / total * 100) if total > 0 else 0
        mismatch_stats['tag_accuracy'][tag]['accuracy'] = accuracy
    
    return mismatch_stats

def print_mismatch_analysis(stats):
    """
    Print formatted analysis of mismatches
    """
    print("\n=== Mismatch Analysis ===")
    print(f"Total tokens analyzed: {stats['total_tokens']}")
    print(f"Total mismatches: {stats['total_mismatches']}")
    print(f"Mismatch percentage: {stats['mismatch_percentage']:.2f}%")
    
    print("\n=== Most Common Mismatch Types ===")
    sorted_mismatches = sorted(
        stats['mismatch_types'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    for mismatch_type, count in sorted_mismatches[:10]:  # Top 10 mismatches
        percentage = (count / stats['total_mismatches'] * 100)
        print(f"{mismatch_type}: {count} ({percentage:.2f}%)")
    
    print("\n=== Per-Tag Accuracy ===")
    # Sort tags by accuracy
    sorted_tags = sorted(
        stats['tag_accuracy'].items(),
        key=lambda x: x[1]['accuracy'] if x[1]['total'] > 0 else -1,
        reverse=True
    )
    for tag, data in sorted_tags:
        if data['total'] > 0:  # Only show tags that appear in the dataset
            print(f"{tag}: {data['accuracy']:.2f}% ({data['correct']}/{data['total']})")

def analyze_validation_results(trainer, validation_dataset):
    """
    Run analysis on validation results
    """
    print("\nAnalyzing validation results...")
    predictions = trainer.predict(validation_dataset)
    stats = analyze_mismatches(predictions)
    print_mismatch_analysis(stats)
    return stats

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

    # #print the confusion matrix for manual inspection
    # #print("Confusion Matrix:")
    # #print(conf_matrix)

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

# Get predictions during evaluation
val_predictions = trainer.predict(encoded_dataset["validation"])
y_true = val_predictions.label_ids
y_pred = val_predictions.predictions.argmax(-1)

# Flatten labels and filter out padding
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

# Create confusion matrix using these labels
conf_matrix_general = confusion_matrix(
    y_true_labels, 
    y_pred_labels,
    labels=list(pos_tag_mapping.keys())[1:]  # Skip [PAD] token
)

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

def calculate_pos_metrics(conf_matrix, labels, output_file):
    metrics = []
    
    # Calculate metrics for each class from confusion matrix
    for i, label in enumerate(labels):
        true_positives = conf_matrix[i, i]
        false_positives = conf_matrix[:, i].sum() - true_positives
        false_negatives = conf_matrix[i, :].sum() - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Map the label to its general category
        category = general_pos_mapping[label]
        
        metrics.append({
            'Category': category,
            'Precision': round(precision, 3),
            'Recall': round(recall, 3),
            'F-Measure': round(f1, 3)
        })
    
    # Write results to file
    with open(output_file, 'w') as f:
        f.write("POS Tagging Metrics Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        # Write header
        f.write(f"{'Category':<15} {'Precision':<12} {'Recall':<12} {'F-Measure':<12}\n")
        f.write("-" * 51 + "\n")
        
        # Write each row
        for metric in metrics:
            f.write(f"{metric['Category']:<15} {metric['Precision']:<12} {metric['Recall']:<12} {metric['F-Measure']:<12}\n")
        
        # Calculate and write overall metrics
        f.write("\nOverall Metrics:\n")
        f.write("-" * 51 + "\n")
        overall_precision = sum(m['Precision'] for m in metrics) / len(metrics)
        overall_recall = sum(m['Recall'] for m in metrics) / len(metrics)
        overall_f1 = sum(m['F-Measure'] for m in metrics) / len(metrics)
        
        f.write(f"{'Average':<15} {overall_precision:.3f}{'':8} {overall_recall:.3f}{'':8} {overall_f1:.3f}\n")

# Calculate metrics
pos_labels = list(pos_tag_mapping.keys())[1:]  # Skip [PAD] token
metrics_df = calculate_pos_metrics(conf_matrix_general, pos_labels, output_file)
#print("POS metrics have been saved.")

# Plot the confusion matrix for general POS categories
#plot_confusion_matrix(conf_matrix_general, pos_labels, save_path)

def write_validation_predictions(trainer, dataset, tokenizer, id2label, output_file_sentence):
    # Get predictions
    predictions = trainer.predict(dataset["validation"])
    y_true = predictions.label_ids
    y_pred = predictions.predictions.argmax(-1)
    
    with open(output_file_sentence, "w", encoding='utf-8') as f:
        for i in range(len(dataset["validation"])):
            # Get the input ids for this sentence
            input_ids = dataset["validation"][i]["input_ids"]
            
            # Convert ids to tokens
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            # Get true and predicted labels for this sentence
            true_labels = y_true[i]
            pred_labels = y_pred[i]
            
            # Write sentence header
            f.write(f"\nSentence {i+1}:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Token':<20} {'True Label':<15} {'Predicted Label':<15} {'Match?':<10}\n")
            f.write("-" * 60 + "\n")
            
            # Process each token in the sentence
            for token, true_label, pred_label in zip(tokens, true_labels, pred_labels):
                # Skip special tokens and padding
                if token in ["[PAD]", "[CLS]", "[SEP]"]:
                    continue
                
                # Convert label ids to actual labels
                true_label_str = id2label[true_label]
                pred_label_str = id2label[pred_label]
                
                # Check if prediction matches truth
                match = "✓" if true_label_str == pred_label_str else "✗"
                
                # Write the token and its labels
                f.write(f"{token:<20} {true_label_str:<15} {pred_label_str:<15} {match:<10}\n")
            
            # Add a separator between sentences
            f.write("\n" + "=" * 60 + "\n")
            
            # Reconstruct and write the full sentence
            sentence_tokens = [t for t in tokens if t not in ["[PAD]", "[CLS]", "[SEP]"]]
            full_sentence = tokenizer.convert_tokens_to_string(sentence_tokens)
            f.write(f"Full sentence: {full_sentence}\n")

# Call the function after your evaluation
#write_validation_predictions(trainer, encoded_dataset, tokenizer, id2label, output_file_sentence)
print("Validation predictions have been saved")

# After training and evaluation
stats = analyze_validation_results(trainer, encoded_dataset["validation"])
print("Evaluation: ", results)
trainer.save_model(save_model)
print(results)