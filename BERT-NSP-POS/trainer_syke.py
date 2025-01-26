"""
Program Title: BERT POS Tagging Trainer for NSP and SOP, 
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
      - NSP (Same Sentence Prediction) with and without Data Augmentation
      - SOP (Sentence Order Prediction) with and without Data Augmentation

Data Structures:
    - Tokenizer: `BertTokenizer` initialized with "bert-base-cased"
    - Model Configurations: Dictionary and model class to configure and store NSP, SOP,
      and data augmentation settings.
    - Training Data: Augmented and non-augmented datasets for comparative model training.
    - Metrics: Evaluation metrics include precision, recall, F1 score, and accuracy,
      structured to assess model performance across different configurations.

Control Flow:
    1. Load and preprocess data with augmentation as per specified configuration.
    2. Initialize BERT model with selected NSP or SOP with or without Data augmentation objective.
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
#print("Device: ", device)

# Input Files:
#train_corpus = "./Dataset/Labeled Corpus/hil-train-set-less-tagset.txt"
#val_corpus = "./Dataset/Labeled Corpus/hil-eval-set-less-tagset.txt"
train_corpus = "./BERT-NSP-POS/corpus/train-set.txt"
val_corpus = "./BERT-NSP-POS/corpus/eval-set.txt"

""" vocab_file = "./Tokenizer/corpus-merged/uncased-vocab.txt"
tokenizer = BertTokenizer(
    vocab_file=vocab_file
) """

bert_model = "gklmip/bert-tagalog-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model)
output_file = "./BERT-NSP-POS/predictions_output_syke.txt"
output_file_sentence = "./BERT-NSP-POS/sentence_predictions_output_syke.txt"
args_directory = "./BERT-NSP-POS/checkpoint_syke"
save_path = "./BERT-NSP-POS/confusion_matrix_syke.png"
epoch_number = 5
save_model = "./BERT-NSP-POS/BERTPOS_syke"

num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': ['[PMP]']})

pos_tag_mapping = {
    '[PAD]': 0,
    'NNC': 1,
    'NNP': 2,
    'NNPA': 3,
    'NNCA': 4,
    'PR': 5,
    'PRS': 6,
    'PRP': 7,
    'PRSP': 8,
    'PRO': 9,
    'PRQ': 10,
    'PRQP': 11,
    'PRL': 12,
    'PRC': 13,
    'PRF': 14,
    'PRI': 15,
    'DT': 16,
    'DTC': 17,
    'DTP': 18,
    'DTPP': 19,
    'LM': 20,
    'CC': 21,
    'CCT': 22,
    'CCR': 23,
    'CCB': 24,
    'CCA': 25,
    'PM': 26,
    'PMP': 27,
    'PME': 28,
    'PMQ': 29,
    'PMC': 30,
    'PMSC': 31,
    'PMS': 32,
    'VB': 33,
    'VBW': 34,
    'VBS': 35,
    'VBN': 36,
    'VBTS': 37,
    'VBTR': 38,
    'VBTF': 39,
    'VBTP': 40,
    'VBAF': 41,
    'VBOF': 42,
    'VBOB': 43,
    'VBOL': 44,
    'VBOI': 45,
    'VBRF': 46,
    'JJ': 47,
    'JJD': 48,
    'JJC': 49,
    'JJCC': 50,
    'JJCS': 51,
    'JJCN': 52,
    'JJCF': 53,
    'JJCB': 54,
    'JJT': 55,
    'RB': 56,
    'RBD': 57,
    'RBN': 58,
    'RBK': 59,
    'RBP': 60,
    'RBB': 61,
    'RBR': 62,
    'RBQ': 63,
    'RBT': 64,
    'RBF': 65,
    'RBW': 66,
    'RBM': 67,
    'RBL': 68,
    'RBI': 69,
    'RBS': 70,
    'RBJ': 71,
    'RBY': 72,
    'RBLI': 73,
    'TS': 74,
    'FW': 75,
    'CD': 76,
    'CCB_CCP': 77,
    'CCR_CCA': 78,
    'CCR_CCB': 79,
    'CCR_CCP': 80,
    'CCR_LM': 81,
    'CCT_CCA': 82,
    'CCT_CCP': 83,
    'CCT_LM': 84,
    'CCU_DTP': 85,
    'CDB_CCA': 86,
    'CDB_CCP': 87,
    'CDB_LM': 88,
    'CDB_NNC': 89,
    'CDB_NNC_CCP': 90,
    'JJCC_CCP': 91,
    'JJCC_JJD': 92,
    'JJCN_CCP': 93,
    'JJCN_LM': 94,
    'JJCS_CCB': 95,
    'JJCS_CCP': 96,
    'JJCS_JJC': 97,
    'JJCS_JJC_CCP': 98,
    'JJCS_JJD': 99,
    '[UNK]': 100,
    '[CLS]': 101,
    '[SEP]': 102,
    'JJCS_JJN': 103,
    'JJCS_JJN_CCP': 104,
    'JJCS_RBF': 105,
    'JJCS_VBAF': 106,
    'JJCS_VBAF_CCP': 107,
    'JJCS_VBN_CCP': 108,
    'JJCS_VBOF': 109,
    'JJCS_VBOF_CCP': 110,
    'JJCS_VBN': 111,
    'RBQ_CCP': 112,
    'JJC_CCB': 113,
    'JJC_CCP': 114,
    'JJC_PRL': 115,
    'JJD_CCA': 116,
    'JJD_CCB': 117,
    'JJD_CCP': 118,
    'JJD_CCT': 119,
    'JJD_NNC': 120,
    'JJD_NNP': 121,
    'JJN_CCA': 122,
    'JJN_CCB': 123,
    'JJN_CCP': 124,
    'JJN_NNC': 125,
    'JJN_NNC_CCP': 126,
    'JJD_NNC_CCP': 127,
    'NNC_CCA': 128,
    'NNC_CCB': 129,
    'NNC_CCP': 130,
    'NNC_NNC_CCP': 131,
    'NN': 132,
    'JJN': 133,
    'NNP_CCA': 134,
    'NNP_CCP': 135,
    'NNP_NNP': 136,
    'PRC_CCB': 137,
    'PRC_CCP': 138,
    'PRF_CCP': 139,
    'PRQ_CCP': 140,
    'PRQ_LM': 141,
    'PRS_CCB': 142,
    'PRS_CCP': 143,
    'PRSP_CCP': 144,
    'PRSP_CCP_NNP': 145,
    'PRL_CCP': 146,
    'PRL_LM': 147,
    'PRO_CCB': 148,
    'PRO_CCP': 149,
    'VBS_CCP': 150,
    'VBTR_CCP': 151,
    'VBTS_CCA': 152,
    'VBTS_CCP': 153,
    'VBTS_JJD': 154,
    'VBTS_LM': 155,
    'VBAF_CCP': 156,
    'VBOB_CCP': 157,
    'VBOF_CCP': 158,
    'VBOF_CCP_NNP': 159,
    'VBRF_CCP': 160,
    'CCP': 161,
    'CDB': 162,
    'RBW_CCP': 163,
    'RBD_CCP': 164,
    'DTCP': 165,
    'VBH': 166,
    'VBTS_VBOF': 167,
    'PRI_CCP': 168,
    'VBTR_VBAF_CCP': 169,
    'DQL': 170,
    'DQR': 171,
    'RBT_CCP': 172,
    'VBW_CCP': 173,
    'RBI_CCP': 174,
    'VBN_CCP': 175,
    'VBTR_VBAF': 176,
    'VBTF_CCP': 177,
    'JJCS_JJD_NNC': 178,
    'CCU': 179,
    'RBL_CCP': 180,
    'VBTR_VBRF_CCP': 181,
    'PRP_CCP': 182,
    'VBTR_VBRF': 183,
    'VBH_CCP': 184,
    'VBTS_VBAF': 185,
    'VBTF_VBOF': 186,
    'VBTR_VBOF': 187,
    'VBTF_VBAF': 188,
    'JJCS_JJD_CCB': 189,
    'JJCS_JJD_CCP': 190,
    'RBM_CCP': 191,
    'NNCS': 192,
    'PRI_CCB': 193,
    'NNA': 194,
    'VBTR_VBOB': 195,
    'DC': 196,
    'JJD_CP': 197,
    'NC': 198,
    'NC_CCP': 199,
    'VBO': 200,
    'JJD_CC': 201,
    'VBF': 202,
    'CP': 203,
    'NP': 204,
    'N': 205,
    'F': 206,
    'CT': 207,
    'MS': 208,
    'BTF': 209,
    'CA': 210,
    'VBOF_RBR': 211,
    'DP': 212,
    'PUNCT' : 213,
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
        return '[PMP] '

    elif symbol == '.':
        return '[PMP] '

    # Check if the symbol is in the list of special symbols
    elif symbol in special_symbols:
        return '[PMP] '

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

    # Loop through the sentence and convert hyphen to '[PMP]' if the next character is a space
    new_sentence = ""
    i = 0
    while i < len(sentence):
        if any(sentence[i:].startswith(symbol) for symbol in special_symbols):
            if i + 2 < len(sentence) and sentence[i:i + 3] == '...':
                # Ellipsis found, replace with '[PMP]'
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
                new_sentence += '[PMP] '
                i += 1
            elif i + 1 == len(sentence):
                new_sentence += '[PMP] '
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
    """
    Improved alignment function that properly handles tokenization and tag alignment.
    """
    # Split and #print initial state
    ##print("\nInitial sentence:", sentence)
    words = sentence.split()
    ##print("Words after split:", words)
    ##print("Tags:", tags)
    
    # Sanity check for lengths
    if len(words) != len(tags):
        print(f"Warning: Mismatch between words ({len(words)}) and tags ({len(tags)})")
        print(f"Words: {words}")
        print(f"Tags: {tags}")
        return [], []
    
    tokenized_sentence = []
    aligned_tagging = []
    
    # Add [CLS] token tag
    tokenized_sentence.append("[CLS]")
    aligned_tagging.append(pos_tag_mapping['[PAD]'])
    
    # Process each word and its corresponding tag
    for word, tag in zip(words, tags):
        # Handle punctuation tokens specially
        if word == '[PMP]':
            tokenized_sentence.append(word)
            aligned_tagging.append(pos_tag_mapping['PMP'])
            continue
            
        # Get subword tokens
        word_tokens = tokenizer.tokenize(word)
        ##print(f"\nWord: {word}")
        ##print(f"Tokenized into: {word_tokens}")
        ##print(f"Tag: {tag}")
        
        # Add all subword tokens with the same tag
        for subword in word_tokens:
            tokenized_sentence.append(subword)
            aligned_tagging.append(tag)
    
    # Add [SEP] token tag
    tokenized_sentence.append("[SEP]")
    aligned_tagging.append(pos_tag_mapping['[PAD]'])
    
    # #print final alignment for debugging
    ##print("\nFinal Alignment:")
    for t, tag in zip(tokenized_sentence, aligned_tagging):
        tag_name = [k for k, v in pos_tag_mapping.items() if v == tag][0] if isinstance(tag, int) else tag
        #print(f"Token: {t:15} Tag: {tag_name}")
    
    ##print(f"\nFinal lengths - Tokens: {len(tokenized_sentence)}, Tags: {len(aligned_tagging)}")
    
    return tokenized_sentence, aligned_tagging

def process_tagged_sentence(tagged_sentence):
    """
    Process a tagged sentence with improved error handling and alignment
    """
    # Preprocess the sentence
    sentence = preprocess_sentence(tagged_sentence)
    
    # Extract and encode tags
    tags = extract_tags(tagged_sentence)
    encoded_tags = []
    for tag in tags:
        if tag in pos_tag_mapping:
            encoded_tags.append(pos_tag_mapping[tag])
        else:
            #print(f"Warning: Unknown tag found: {tag}")
            encoded_tags.append(pos_tag_mapping['[PAD]'])
    
    # Get aligned tokens and tags
    tokenized_sentence, aligned_tags = align_tokenization(sentence, encoded_tags)
    
    # Create the final encoded sentence
    encoded_sentence = {
        'input_ids': tokenizer.convert_tokens_to_ids(tokenized_sentence),
        'attention_mask': [1] * len(tokenized_sentence),
        'encoded_tags': aligned_tags
    }
    
    # Pad sequences to max length
    max_length = 128
    padding_length = max_length - len(encoded_sentence['input_ids'])
    
    if padding_length > 0:
        encoded_sentence['input_ids'].extend([0] * padding_length)
        encoded_sentence['attention_mask'].extend([0] * padding_length)
        encoded_sentence['encoded_tags'].extend([pos_tag_mapping['[PAD]']] * padding_length)
    elif padding_length < 0:
        # Truncate if longer than max_length
        encoded_sentence['input_ids'] = encoded_sentence['input_ids'][:max_length]
        encoded_sentence['attention_mask'] = encoded_sentence['attention_mask'][:max_length]
        encoded_sentence['encoded_tags'] = encoded_sentence['encoded_tags'][:max_length]
    
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
trainer.save_model(save_model)
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
plot_confusion_matrix(conf_matrix_general, pos_labels, save_path)

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
write_validation_predictions(trainer, encoded_dataset, tokenizer, id2label, output_file_sentence)
print("Validation predictions have been saved")

# After training and evaluation
stats = analyze_validation_results(trainer, encoded_dataset["validation"])
print("Evaluation: ", results)
print(results)