import re
import torch
from transformers import BertTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, EvalPrediction
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print("Device: ", device)

# Input Files:
""" train_corpus = "./BERT-SOP-POS/corpus/train-set.txt"
val_corpus = "./BERT-SOP-POS/corpus/eval-set.txt" """
""" train_corpus = "./Dataset/Labeled Corpus/train-set.txt"
val_corpus = "./Dataset/Labeled Corpus/eval-set.txt" """
train_corpus = "./Dataset/Labeled Corpus/augmented hil-train-set-new.txt"
val_corpus = "./Dataset/Labeled Corpus/augmented hil-eval-set-new.txt"

""" vocab_file = "./BERT-SSP/tokenizer-corpus-tagalog/uncased-vocab.txt" """
""" vocab_file = "./BERT-SSP/tokenizer-corpus-hiligaynon/uncased-vocab.txt" """
vocab_file = "./BERT-SSP/tokenizer-corpus-hiligaynon/uncased-vocab.txt"
tokenizer = BertTokenizer(
    vocab_file=vocab_file
)

bert_model = "./BERT-SOP/output_model_tagalog_hiligaynon/checkpoint-67000/"
output_file = "./BERT-SOP-DA-POS/predictions_output.txt"
args_directory = "./BERT-SOP-DA-POS/checkpoint"
save_path = "./BERT-SOP-DA-POS/confusion_matrix.png"
epoch_number = 25
save_model = "./BERT-SOP-DA-POS/BERTPOS"

num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': ['[PMP]', '[PMS]', '[PMC]']})

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
    'CJN': 21,
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
    'VB': 41,
    'VB': 42,
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
    'CDB': 76,
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
    'JJCS_VB': 106,
    'JJCS_VB_CCP': 107,
    'JJCS_VBN_CCP': 108,
    'JJCS_VB': 109,
    'JJCS_VB_CCP': 110,
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
    'VB_CCP': 156,
    'VBOB_CCP': 157,
    'VB_CCP': 158,
    'VB_CCP_NNP': 159,
    'VBRF_CCP': 160,
    'CCP': 161,
    'CDB': 162,
    'RBW_CCP': 163,
    'RBD_CCP': 164,
    'DTCP': 165,
    'VBH': 166,
    'VBTS_VB': 167,
    'PRI_CCP': 168,
    'VBTR_VB_CCP': 169,
    'DQL': 170,
    'DQR': 171,
    'RBT_CCP': 172,
    'VBW_CCP': 173,
    'RBI_CCP': 174,
    'VBN_CCP': 175,
    'VBTR_VB': 176,
    'VBTF_CCP': 177,
    'JJCS_JJD_NNC': 178,
    'CCU': 179,
    'RBL_CCP': 180,
    'VBTR_VBRF_CCP': 181,
    'PRP_CCP': 182,
    'VBTR_VBRF': 183,
    'VBH_CCP': 184,
    'VBTS_VB': 185,
    'VBTF_VB': 186,
    'VBTR_VB': 187,
    'VBTF_VB': 188,
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
    'VB_RBR': 211,
    'DP': 212,
}

# Mapping of specific POS tags to their general categories
general_pos_mapping = {
    'NNC': 'Noun', 'NNP': 'Noun', 'NNPA': 'Noun', 'NNCA': 'Noun',
    'PR': 'Pronoun', 'PRS': 'Pronoun', 'PRP': 'Pronoun', 'PRSP': 'Pronoun', 'PRO': 'Pronoun', 'PRL': 'Pronoun', 'PRC': 'Pronoun',
    'DT': 'Determiner', 'DTC': 'Determiner', 'DTP': 'Determiner',
    'LM': 'Lexical Marker',
    'CJN': 'Conjunction', 'CCP': 'Conjunction', 'CCU': 'Conjunction',
    'VB': 'Verb', 'VBW': 'Verb', 'VBS': 'Verb', 'VBH': 'Verb', 'VBN': 'Verb', 'VBTS': 'Verb', 'VBTR': 'Verb', 'VBTF': 'Verb',
    'JJ': 'Adjective', 'JJD': 'Adjective', 'JJC': 'Adjective', 'JJCC': 'Adjective', 'JJCS': 'Adjective', 'JJN': 'Adjective',
    'RB': 'Adverb', 'RBD': 'Adverb', 'RBN': 'Adverb', 'RBK': 'Adverb', 'RBR': 'Adverb', 'RBQ': 'Adverb', 'RBT': 'Adverb', 'RBF': 'Adverb', 'RBW': 'Adverb', 'RBM': 'Adverb', 'RBL': 'Adverb', 'RBI': 'Adverb',
    'CDB': 'Digit, Rank, Count', 
    'FW': 'Foreign Words', 
    'PM': 'Punctuation', 'PMP': 'Punctuation', 'PME': 'Punctuation','PMQ': 'Punctuation', 'PMC': 'Punctuation', 'PMSC': 'Punctuation', 'PMS': 'Punctuation'
}

num_labels = len(pos_tag_mapping)
id2label = {idx: tag for tag, idx in pos_tag_mapping.items()}
label2id = {tag: idx for tag, idx in pos_tag_mapping.items()}

def symbol2token(symbol):
    special_symbols = ['-', '&', "\"", "[", "]", "/", "$", "(", ")", "%", ":", "'", '.']
    # Check if the symbol is a comma
    if symbol == ',':
        return '[PMC] '

    elif symbol == '.':
        return '[PMP] '

    # Check if the symbol is in the list of special symbols
    elif symbol in special_symbols:
        return '[PMS] '

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


    # print("---")
    # print("Sentence before:", sentence)

    # Loop through the sentence and convert hyphen to '[PMP]' if the next character is a space
    new_sentence = ""
    i = 0
    # print("Length: ", len(sentence))
    while i < len(sentence):
        # print(f"{i+1} == {len(sentence)}: {sentence[i]}")

        if any(sentence[i:].startswith(symbol) for symbol in special_symbols):
            if i + 2 < len(sentence) and sentence[i:i + 3] == '...':
                # Ellipsis found, replace with '[PMS]'
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
                new_sentence += '[PMS] '
                i += 1
            elif i + 1 == len(sentence):
                new_sentence += '[PMS] '
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

    # print("Sentence after:", new_sentence)
    # print("---")

    return new_sentence
def extract_tags(input_sentence):
    tags = re.findall(r'<([A-Z_]+)\s.*?>', input_sentence)
    return tags

""" # New function for prediction_output
def align_tokenization(sentence, tags):
    sentence = sentence.split()
    tokenized_sentence = tokenizer.tokenize(' '.join(sentence))

    aligned_tagging = []
    current_word = ''
    index = 0

    for token in tokenized_sentence:
        if len(tags) > index:
            current_word += re.sub(r'^##', '', token)
            
            if sentence[index] == current_word:  # Word completed
                current_word = ''
                aligned_tagging.append(tags[index])
                index += 1
            else:  # Subword token, repeat the current label
                aligned_tagging.append(tags[index])

    # Pad remaining labels if necessary
    while len(aligned_tagging) < len(tokenized_sentence):
        aligned_tagging.append(0)  # Use 0 (pad) or other appropriate padding value
    
    return tokenized_sentence, aligned_tagging """

def align_tokenization(sentence, tags):

    #print("Sentence \n: ", sentence)
    sentence = sentence.split()
    #print("Sentence Split\n: ", sentence)

    tokenized_sentence = tokenizer.tokenize(' '.join(sentence))
    tokenized_sentence_string = " ".join(tokenized_sentence)
    #print("ID2Token_string\n: ", tokenized_sentence_string)
    #print("Tags\n: ", [id2label[tag_id] for tag_id in tags])
    if len(tags) > 12:
        #print(id2label[tags[11]])
        """  """

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
    print(tokenized_sentence)
    print(aligned_tagging)
    print("\n")
    #assert len(tokenized_sentence) == len(aligned_tagging)

    aligned_tagging = [0] + aligned_tagging
    return tokenized_sentence, aligned_tagging

def process_tagged_sentence(tagged_sentence):
    # print(tagged_sentence)

    sentence = preprocess_sentence(tagged_sentence)
    tags = extract_tags(tagged_sentence) # returns the tags (eto ilagay mo sa tags.txt)


    encoded_tags = [pos_tag_mapping[tag] for tag in tags]

    # Align tokens
    tokenized_sentence, encoded_tags = align_tokenization(sentence, encoded_tags)
    encoded_sentence = tokenizer(sentence, padding='max_length' ,truncation=True, max_length=128)

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = [1] * len(encoded_sentence['input_ids'])
    """ print("len(encoded_sentence['input_ids']):", len(encoded_sentence['input_ids'])) """
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
    """ print(encoded_sentence)
    print("Sentence:", decoded_sentence)
    print("Tags:", decoded_tags)
    print("Decoded Sentence and Tags:", word_tag_pairs)
    print("---") """

    return encoded_sentence

def encode_corpus(input_file):

    encoded_sentences = []

    with open(input_file, 'r') as f:
        lines = f.readlines()

    # int = 1

    for line in tqdm(lines, desc="Processing corpus"):
        # print(int)
        # int += 1
        input_sentence = line.strip()
        # print(input_sentence)

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
""" print(encoded_dataset) """

max_token_length = 128
vocab_size = tokenizer.vocab_size

encoded_dataset.set_format("torch")

model = AutoModelForTokenClassification.from_pretrained(bert_model,
                                                           num_labels=num_labels,
                                                           id2label=id2label,
                                                           label2id=label2id)

model.resize_token_embeddings(len(tokenizer))

batch_size = 16
metric_name = "precision"

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
    conf_matrix = confusion_matrix(y_true_flat, y_pred_flat, labels=list(label2id.values()))

    # Print the confusion matrix for manual inspection
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate precision, recall, and f1
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_flat, y_pred_flat, average="macro")

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
y_true_flat_general = map_to_general_pos([id2label[tag_id] for tag_id in y_true_flat], general_pos_mapping)
y_pred_flat_general = map_to_general_pos([id2label[tag_id] for tag_id in y_pred_flat], general_pos_mapping)

# Print each sentence with its corresponding predictions for each word and write to a text file
def print_sentences_with_predictions_to_file(validation_dataset, y_true_filtered, y_pred_filtered, output_file=output_file):
    with open(output_file, "w") as f:
        for i in range(len(validation_dataset)):
            # Decode the original sentence
            input_ids = validation_dataset[i]["input_ids"]
            original_sentence_tokens = tokenizer.convert_ids_to_tokens(input_ids)

            # Skip special tokens like [CLS], [SEP], and [PAD]
            original_sentence_tokens = [
                token for token in original_sentence_tokens if token not in ["[CLS]", "[SEP]", "[PAD]"]
            ]

            # Get the true and predicted labels for the sentence
            true_labels = y_true_filtered[i * len(original_sentence_tokens): (i + 1) * len(original_sentence_tokens)]
            pred_labels = y_pred_filtered[i * len(original_sentence_tokens): (i + 1) * len(original_sentence_tokens)]

            # Remove padding tokens from the sentence
            filtered_sentence_tokens, filtered_true_labels, filtered_pred_labels = [], [], []
            for token, true_label, pred_label in zip(original_sentence_tokens, true_labels, pred_labels):
                if true_label != '[PAD]' and pred_label != '[PAD]':
                    filtered_sentence_tokens.append(token)
                    filtered_true_labels.append(true_label)
                    filtered_pred_labels.append(pred_label)

            # Write each word with its corresponding true and predicted labels to the file
            f.write(f"Sentence {i + 1}:\n")
            for token, true_label, pred_label in zip(filtered_sentence_tokens, filtered_true_labels, filtered_pred_labels):
                f.write(f"Word: {token}\tTrue Label: {true_label}\tPredicted Label: {pred_label}\n")
            f.write("\n")

# Call the function to print sentences and word-level predictions to a text file
#print_sentences_with_predictions_to_file(encoded_dataset['validation'], y_true_labels, y_pred_labels)

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

# Plot the confusion matrix for general POS categories
plot_confusion_matrix(conf_matrix_general, unique_labels, save_path)

print("Evaluation: ", results)
trainer.save_model(save_model)
print(results)