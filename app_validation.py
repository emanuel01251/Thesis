import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import logging
from tqdm import tqdm
from itertools import zip_longest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the model paths
model_paths = {
    "SSP with Augmentation TL only": "./BERT-SSP-DA-POS/BERTPOS_tl_only",
    "SOP with Augmentation": "./BERT-SOP-DA-POS/BERTPOS_tl",
    "SSP without Augmentation": "./BERT-SSP-POS/BERTPOS_tl",
    "SOP without Augmentation": "./BERT-SOP-POS/BERTPOS_tl",
    "Baseline": "./BERT-NSP-POS/BERTPOS_tl",
    "Baseline with Augmentation": "./BERT-NSP-DA-POS/BERTPOS_tl_only"
}

# POS tag mapping
pos_tag_mapping = {
    '[PAD]': 0,
    'NN': 1,
    'PRNN': 2,
    'DET': 3,
    'VB': 4,
    'ADJ': 5,
    'ADV': 6,
    'NUM': 7,
    'CONJ': 8,
    'PUNCT': 9,
    'FW': 10,
    'LM': 11,
}

pos_tag_mapping_1 = {
    'NN': 1,
    'PRNN': 2,
    'DET': 3,
    'VB': 4,
    'ADJ': 5,
    'ADV': 6,
    'NUM': 7,
    'CONJ': 8,
    'PUNCT': 9,
    'FW': 10,
    'LM': 11,
}

num_labels = len(pos_tag_mapping)
id2label = {idx: tag for tag, idx in pos_tag_mapping.items()}
label2id = {tag: idx for tag, idx in pos_tag_mapping.items()}

special_symbols = ['-', '&', "\"", "[", "]", "/", "$", "(", ")", "%", ":", "'", '.', '?', ',']

def symbol2token(symbol):
    if symbol == ',' or symbol == '.' or symbol in special_symbols:
        return '[PUNCT]'
    return symbol

def preprocess_sentence(tagged_sentence):
    """Extract sentence from tagged format and preprocess it"""
    pattern = r'<(\w+)\s+([^>]+)>'
    matches = re.findall(pattern, tagged_sentence)
    if not matches:
        return "", [], []
    
    words = [match[1] for match in matches]
    tags = [match[0] for match in matches]
    
    # Join words and preprocess
    sentence = ' '.join(words)
    sentence = sentence.lower()
    
    # Handle special symbols
    processed_tokens = []
    i = 0
    while i < len(sentence.split()):
        token = sentence.split()[i]
        if any(symbol in token for symbol in special_symbols):
            if '...' in token:
                processed_tokens.append('[PUNCT]')
            else:
                processed_tokens.append(symbol2token(token))
        else:
            processed_tokens.append(token)
        i += 1
    
    return ' '.join(processed_tokens), tags, words

def insert_and_shift(array, insert_position, shift_amount):
    # Convert tuple list to list for easier manipulation
    working_array = list(array)
    
    # Insert blank items
    for _ in range(shift_amount):
        working_array.insert(insert_position, (' ', ' '))
    
    """ # Shift the second elements starting from insert position
    for i in range(insert_position, len(working_array) - shift_amount):
        # Take second element from positions ahead
        working_array[i] = (working_array[i][0], working_array[i + shift_amount][1])
    
    # Clear the second elements of the last positions
    for i in range(len(working_array) - shift_amount, len(working_array)):
        working_array[i] = (working_array[i][0], '') """
        
    return working_array

def plot_confusion_matrix(conf_matrix, labels, save_path, model_name):
    """Plot and save confusion matrix visualization"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for {model_name}")
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def process_validation_file(file_path, model_name):
    """Process validation file and generate metrics using confusion matrix"""
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_paths[model_name])
        model = AutoModelForTokenClassification.from_pretrained(model_paths[model_name])
        
        # Add special tokens if needed
        num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': ['[PUNCT]']})
        model.resize_token_embeddings(len(tokenizer))
        
        # Initialize lists for true and predicted labels
        all_true_labels = []
        all_pred_labels = []
        predictions_output = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Process the sentence
                processed_sentence, true_tags, original_tokens = preprocess_sentence(line)
                
                if not processed_sentence:
                    continue
                
                # Tokenize
                encoded = tokenizer(processed_sentence, padding="max_length", truncation=True, 
                                 max_length=128, return_tensors="pt")
                
                # Get predictions
                with torch.no_grad():
                    outputs = model(**encoded)
                    logits = outputs.logits
                    probabilities = F.softmax(logits, dim=-1)
                    predictions = torch.argmax(probabilities, dim=-1)
                
                pred_labels = [id2label[tag.item()] for tag in predictions[0] if id2label[tag.item()] != '[PAD]']
                #pred_labels = pred_labels[:len(true_tags)]
                
                # Create true_label and predict_label lists
                true_label = list(zip_longest(original_tokens, true_tags, fillvalue=''))
                predict_label = list(zip_longest(original_tokens, pred_labels, fillvalue=''))
                #print(true_label)
                new_true_label = true_label

                # Implement the alignment correction logic
                for index in range(len(true_label)):
                    token = true_label[index][0]
                    has_special_symbols = any(symbol in token for symbol in special_symbols)
                    has_content = len(''.join(char for char in token if char not in special_symbols)) > 1
                    
                    if has_special_symbols and has_content:
                        special_symbol_count = sum(token.count(symbol) for symbol in special_symbols)
                        new_true_label = insert_and_shift(new_true_label, index+1, special_symbol_count+1)

                # Update predict_label with the corrected alignment
                true_label = new_true_label
                array1, array2 = zip_longest(*true_label, fillvalue='')
                true_tags = array2
                # Store labels for confusion matrix
                min_len = min(len(true_tags), len(pred_labels))
                all_true_labels.extend(true_tags[:min_len])
                all_pred_labels.extend(pred_labels[:min_len])
                
                # Store prediction details
                predictions_output.append(f"\nSentence {i}:")
                predictions_output.append("-" * 60)
                predictions_output.append(f"{'Token':<15} {'True Label':<15} {'Predicted Label':<15} {'Match?'}")
                predictions_output.append("-" * 60)
                
                for j in range(len(true_label)):
                    if j < len(predict_label):
                        token = true_label[j][0]
                        true_tag = true_label[j][1]
                        pred_tag = predict_label[j][1] if predict_label[j][1] else ''
                        match = "✓" if true_tag == pred_tag else "✗"
                        
                        if true_tag == " ":
                            match = "-"
                        
                        predictions_output.append(
                            f"{token:<15} {true_tag:<15} {pred_tag:<15} {match}"
                        )
                
                predictions_output.append(f"\nFull sentence: {' '.join(original_tokens)}\n")
        
        # Get unique labels for confusion matrix
        unique_labels = sorted(set(pos_tag_mapping_1.keys()))
        
        # Verify lengths before creating confusion matrix
        if len(all_true_labels) != len(all_pred_labels):
            logger.warning(f"Final length mismatch: true={len(all_true_labels)}, pred={len(all_pred_labels)}")
            # Truncate to shorter length
            min_len = min(len(all_true_labels), len(all_pred_labels))
            all_true_labels = all_true_labels[:min_len]
            all_pred_labels = all_pred_labels[:min_len]
        
        # Create confusion matrix
        conf_matrix = confusion_matrix(
            all_true_labels, 
            all_pred_labels,
            labels=unique_labels
        )
        
        # Plot confusion matrix
        save_path = f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
        plot_confusion_matrix(conf_matrix, unique_labels, save_path, model_name)
        
        # Calculate metrics from confusion matrix
        metrics_output = ["POS Tagging Metrics Analysis", "=" * 60]
        metrics_output.append(f"{'Category':<20} {'Precision':<12} {'Recall':<12} {'F-Measure':<12}")
        metrics_output.append("-" * 60)
        
        precisions = []
        recalls = []
        f1s = []
        
        for i, category in enumerate(unique_labels):
            true_positives = conf_matrix[i, i]
            false_positives = conf_matrix[:, i].sum() - true_positives
            false_negatives = conf_matrix[i, :].sum() - true_positives
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            
            metrics_output.append(
                f"{category:<20} {precision:.3f} {recall:.3f} {f1:.3f}"
            )
        
        # Calculate macro averages
        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        macro_f1 = np.mean(f1s)
        
        metrics_output.extend([
            "\nMacro Averages:",
            "-" * 60,
            f"Average {' '*13} {macro_precision:.3f} {macro_recall:.3f} {macro_f1:.3f}"
        ])
        
        return metrics_output, predictions_output, conf_matrix
        
    except Exception as e:
        logger.error(f"Error processing validation file: {str(e)}")
        raise

def main():
    validation_file = "./Dataset/Labeled Corpus/tl-hil-eval-set-less-tagset.txt"
    model_name = "SSP without Augmentation"
    #pos_metric_name = "pos_metrics_ssp_da_tlonly.txt"
    pos_metric_name = "sample.txt"
    #predictions_output_name = "predictions_output_ssp_da_tlonly.txt"
    predictions_output_name = "sample.txt"
    
    logger.info(f"Processing validation file: {validation_file}")
    logger.info(f"Using model: {model_name}")
    
    metrics_output, predictions_output, conf_matrix = process_validation_file(validation_file, model_name)
    
    # Write metrics to file
    with open(pos_metric_name, "w", encoding='utf-8') as f:
        f.write("\n".join(metrics_output))
    
    # Write predictions to file
    with open(predictions_output_name, "w", encoding='utf-8') as f:
        f.write("\n".join(predictions_output))
    
    logger.info("Processing complete. Results saved to files.")

if __name__ == "__main__":
    main()