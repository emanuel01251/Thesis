import re
import torch
from transformers import BertTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, EvalPrediction
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from datasets import Dataset, DatasetDict

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print("Device: ", device)

# Input Files:
train_corpus = "corpus/train-set.txt"
val_corpus = "corpus/eval-set.txt"


bert_model = "gklmip/bert-tagalog-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model)

# print(tokenizer.additional_special_tokens_ids)
num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': ['[PMP]', '[PMS]', '[PMC]']})


# special_tokens = ['[PMP]', '[PMS]', '[PMC]']
# tokenizer.add_tokens(special_tokens, special_tokens=True)
# print("[PMP] token ID:", tokenizer.convert_tokens_to_ids('[PMP]'))
# print("[PMS] token ID:", tokenizer.convert_tokens_to_ids('[PMS]'))
# print("[PMC] token ID:", tokenizer.convert_tokens_to_ids('[PMC]'))

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

def align_tokenization(sentence, tags):

    print("Sentence \n: ", sentence)
    sentence = sentence.split()
    print("Sentence Split\n: ", sentence)

    tokenized_sentence = tokenizer.tokenize(' '.join(sentence))
    tokenized_sentence_string = " ".join(tokenized_sentence)
    print("ID2Token_string\n: ", tokenized_sentence_string)
    print("Tags\n: ", [id2label[tag_id] for tag_id in tags])
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
                print("completed a word: ", current_word)
                current_word = ''
                aligned_tagging.append(tags[index])
                # print(f"Tag of index {index}: ", id2label[tags[index]])
                # print(f"Aligned tag of index {index}: ", (id2label[aligned_tagging[-1]]))
                # print("Tags1\n: ", [id2label[tag_id] for tag_id in tags])
                # print("Tags2\n: ", [id2label[tag_id] for tag_id in aligned_tagging])
                # print(f"{index+1}/{len(tags)} tags consumed")
                index += 1
            else:  # otherwise insert padding
                print("incomplete word: ", current_word)
                aligned_tagging.append(0)

            print("---")

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
    print("len(encoded_sentence['input_ids']):", len(encoded_sentence['input_ids']))
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
    print(encoded_sentence)
    print("Sentence:", decoded_sentence)
    print("Tags:", decoded_tags)
    print("Decoded Sentence and Tags:", word_tag_pairs)
    print("---")

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


test_sentence = [
'SNT.108970.2066 <DTC Ang.> <PRI isa> <CCT sa> <DTCP mga> <NNC susog> <CCP na> <PRO ito> <PMC ,> <DTC ang> <NNP Post-9> <PMS /> <CDB 11> <NNP Batas> <NNP Pangtulong> <CCT sa> <NNP Edukasyon> <CCB ng> <DTCP mga> <NNP Beterano> <CDB 2008> <PMC ,> <LM ay> <RBT_CCP pwedeng> <VBAF magpakita> <CCT bilang> <JJD_CCP modernong> <NNC salin> <CCB ng> <NNC panahon> <CCB ng> <NNP_CCP Ikalawang> <NNP_CCP Digmaang> <JJD pangdaigdig> <PMP .>',
'SNT.206230.256	<VBTS -Sinabi-> <CCB n-g> <NNC tag-apag-salita> <CCT para> <CCT sa> <NNP Winner> <NNP International> <CCT sa> <PRI_CCP isang> <NNC pahayag> <PMC ,> <PMS "> <PMS [> <PRO ito> <LM ay> <PMS ]> <JJCS_JJD napakahirap> <CCP na> <NNC panahon> <CCT para> <CCT sa> <PRSP_CCP aming> <PRI lahat> <CCA at> <VBTR hinihiling> <CCB ng> <NNC pamilya> <CCP na> <VBTF igalang> <PRP ninyo> <DTC ang> <PRSP_CCP kanilang> <NNC pribasya> <PMP .> <PMS ">',
'SNT.187937.383	<VBTS Sinabi> <CCB ng> <JJN pangalawang> <PMS -> <NNC tagapangulo> <CCP na> <DTP si> <NNP Lee> <NNP Cheuk-yan> <CCP na> <DTC ang> <VBOF ginawa> <CCB ng> <NNPA C&ED> <LM ay> <RBF hindi> <JJD pangkaraniwan> <PMC ,> <CCA at> <VBOF tinawagan> <RBI na> <PRS niya> <DTC ang> <NNC departamento> <CCB upang> <VBW makita> <CCR kung> <DTC ang> <NNC departamento> <LM ay> <VBTR pinipilit> <CCP na> <VBW makita> <DTC ang> <DTCP mga> <NNC_CCP kagamitang> <VBH may> <NNC kaugnayan> <CCT sa> <NNC_CCP insidenteng> <VBTS naganap> <CCT sa> <NNP Tiananmen> <NNP Square> <PMP .>'
]


train_corpus = encode_corpus(train_corpus)
val_corpus = encode_corpus(val_corpus)


encoded_dataset = createDataset(train_corpus, val_corpus)
print(encoded_dataset)

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
    "checkpoint",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    label_names=["labels"],
)


def compute_metrics(p):
    y_true = p.label_ids #(sentence[num_of_sentences], words[number_of_words]) (800, 128)
    y_pred = p.predictions.argmax(-1)

    y_true_flat = [tag_id for tags in y_true for tag_id in tags]
    y_pred_flat = [tag_id for tags in y_pred for tag_id in tags]

    precision, recall, f1, _ = precision_recall_fscore_support(y_true_flat, y_pred_flat, average="micro")

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
print(training)
results = trainer.evaluate()
print("Evaluation: ", results)
trainer.save_model("BERTPOS")
print(results)
