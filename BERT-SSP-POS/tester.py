import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
import gradio as gr

# Load the trained model and tokenizer
model_checkpoint = "BERTPOS"
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

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

special_symbols = ['-', '&', "\"", "[", "]", "/", "$", "(", ")", "%", ":", "'", '.', '?', ',']

def symbol2token(symbol):

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

def preprocess_untagged_sentence(sentence):
    # Define regex pattern to capture all special symbols
    special_symbols_regex = '|'.join([re.escape(sym) for sym in ['-', '&', "\"", "[", "]", "/", "$", "(", ")", "%", ":", "'", '.']])

    # Replace all special symbols with spaces around them
    sentence = re.sub(rf'({special_symbols_regex})', r' \1 ', sentence)

    # Remove extra whitespaces
    sentence = re.sub(r'\s+', ' ', sentence).strip()

    upper = sentence

    # Convert the sentence to lowercase
    sentence = sentence.lower()

    # Loop through the sentence and convert special symbols to tokens [PMS], [PMC], or [PMP]
    new_sentence = ""
    i = 0
    while i < len(sentence):
        if any(sentence[i:].startswith(symbol) for symbol in special_symbols):
            # Check for ellipsis and replace with '[PMS]'
            if i + 2 < len(sentence) and sentence[i:i + 3] == '...':
                new_sentence += '[PMS]'
                i += 3
            # Check for single special symbols
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
        # Check for special symbols at the start of the sentence
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

    print("Sentence after:", new_sentence.split())
    print("---")

    return new_sentence, upper


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

    print("Sentence after:", new_sentence.split())
    print("---")

    return new_sentence
def extract_tags(input_sentence):
    tags = re.findall(r'<([A-Z_]+)\s.*?>', input_sentence)
    return tags

def align_tokenization(sentence, tags):

    print("Sentence \n: ", sentence)
    sentence = sentence.split()
    print("Sentence Split\n: ", sentence)

    tokenized_sentence = tokenizer.tokenize(' '.join(sentence))
    # tokenized_sentence_string = " ".join(tokenized_sentence)
    # print("ID2Token_string\n: ", tokenized_sentence_string)

    aligned_tagging = []
    current_word = ''
    index = 0  # index of the current word in the sentence and tagging

    for token in tokenized_sentence:
        current_word += re.sub(r'^##', '', token)
        print("Current word after replacing ##: ", current_word)
        print("sentence[index]: ", sentence[index])

        if sentence[index] == current_word:  # if we completed a word
            print("completed a word: ", current_word)
            current_word = ''
            aligned_tagging.append(tags[index])
            index += 1
        else:  # otherwise insert padding
            print("incomplete word: ", current_word)
            aligned_tagging.append(0)

        print("---")

    decoded_tags = [list(pos_tag_mapping.keys())[list(pos_tag_mapping.values()).index(tag_id)] for tag_id in
                    aligned_tagging]
    print("Tokenized Sentence\n: ", tokenized_sentence)
    print("Tags\n: ", decoded_tags)

    assert len(tokenized_sentence) == len(aligned_tagging)

    aligned_tagging = [0] + aligned_tagging
    return tokenized_sentence, aligned_tagging


def process_tagged_sentence(tagged_sentence):
    # print(tagged_sentence)

    # Preprocess the input tagged sentence and extract the words and tags
    sentence = preprocess_sentence(tagged_sentence)
    tags = extract_tags(tagged_sentence) # returns the tags (eto ilagay mo sa tags.txt)


    encoded_tags = [pos_tag_mapping[tag] for tag in tags]

    # Align tokens by adding padding if needed
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

import torch
import torch.nn.functional as F

def tag_sentence(input_sentence):
    # Preprocess the input tagged sentence and extract the words and tags
    sentence, upper = preprocess_untagged_sentence(input_sentence)

    # Tokenize the sentence and decode it
    encoded_sentence = tokenizer(sentence, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    # Pass the encoded sentence to the model to get the predicted logits
    with torch.no_grad():
        model_output = model(**encoded_sentence)

    # Get the logits and apply softmax to convert them into probabilities
    logits = model_output.logits
    probabilities = F.softmax(logits, dim=-1)

    # Get the predicted tag for each token in the sentence
    predicted_tags = torch.argmax(probabilities, dim=-1)

    # Convert the predicted tags to their corresponding labels using id2label
    labels = [id2label[tag.item()] for tag in predicted_tags[0] if id2label[tag.item()] != '[PAD]']

    return labels

# Example usage:
test_sentence = 'Ang bahay ay maganda na para bang may kumikislap sa bintana .'

def predict_tags(test_sentence):

    sentence, upper = preprocess_untagged_sentence(test_sentence)
    words_list = upper.split()
    print("Words: ", words_list)
    predicted_tags = tag_sentence(test_sentence)
    print(predicted_tags)

    pairs = list(zip(words_list, predicted_tags))
    return pairs

predict_tags(test_sentence)
tagger = gr.Interface(
    predict_tags,
    gr.Textbox(placeholder="Enter sentence here..."),
    ["highlight"],
    title="BERT Filipino Part of Speech Tagger",
    description="Enter a text in Tagalog to classify the tags for each word. Each word to tag needs to be space separated.",
    examples=[
        ["Ang bahay ay lumiliwanag na para bang may kumikislap sa bintana"],
        ["Naisip ko na kumain na lang tayo sa pinakasikat na restaurant sa Manila ."],
    ],
)

tagger.launch()
