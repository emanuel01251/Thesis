import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re

# Define the paths to your models
model_paths = {
    "SSP with Augmentation": "./BERT-SSP-DA-POS/BERTPOS",
    "SOP with Augmentation": "./BERT-SOP-DA-POS/BERTPOS",
    "SSP without Augmentation": "./BERT-SSP-POS/BERTPOS",
    "SOP without Augmentation": "./BERT-SOP-POS/BERTPOS"
}

models = {name: AutoModelForTokenClassification.from_pretrained(path) for name, path in model_paths.items()}
tokenizers = {name: AutoTokenizer.from_pretrained(path) for name, path in model_paths.items()}

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
    tags = re.findall(r'<([A-Z_]+)\s.*?>', input_sentence)
    return tags

def clear_function(): 
    return "", "SSP with Augmentation"

import torch
import torch.nn.functional as F

def tag_sentence(input_sentence, selected_model):
    model = models[selected_model]  # Get the selected model
    tokenizer = tokenizers[selected_model]  # Get the corresponding tokenizer

    # Preprocess the input sentence
    sentence, upper = preprocess_untagged_sentence(input_sentence)
    
    # Tokenize the sentence
    encoded_sentence = tokenizer(sentence, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    
    # Pass the encoded sentence to the model to get logits
    with torch.no_grad():
        model_output = model(**encoded_sentence)
    
    # Get the predicted tags
    logits = model_output.logits
    probabilities = F.softmax(logits, dim=-1)
    predicted_tags = torch.argmax(probabilities, dim=-1)
    
    # Convert predicted tags to their corresponding labels and scores
    scores = probabilities[0].max(dim=-1)[0].tolist()  # Get max scores for each token
    labels = [id2label[tag.item()] for tag in predicted_tags[0]]
    
    # Filter out the padding tokens and their scores
    filtered_pairs = [(labels[i], scores[i]) for i in range(len(labels)) if labels[i] != '[PAD]']
    
    return filtered_pairs  # Return the pairs of (label, score)

def predict_tags(test_sentence, selected_model):
    sentence, upper = preprocess_untagged_sentence(test_sentence)
    words_list = upper.split()
    predicted_pairs = tag_sentence(test_sentence, selected_model)

    # Align words with their corresponding predicted tags and scores
    formatted_output = [{"word": words_list[i], "entity": predicted_pairs[i][0], "score": predicted_pairs[i][1]}
                        for i in range(len(predicted_pairs))]
    
    return formatted_output

""" def pos_tagger(text, selected_model):
    model = models[selected_model]
    
    inputs = tokenizers[selected_model](text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_tags = outputs.logits.argmax(-1).tolist()
    
    tagged_text = " ".join([f"{word}/{tag}" for word, tag in zip(tokenizers.convert_ids_to_tokens(inputs["input_ids"][0]), predicted_tags[0])])
    
    return tagged_text """

html_table = """
<table style="width: 100%; border: 1px solid gray; border-collapse: collapse;">
    <thead>
        <tr>
            <th style="border: 1px solid black; padding: 8px;">Part of Speech</th>
            <th style="border: 1px solid black; padding: 8px;">Tags</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Common Noun</td>
            <td style="border: 1px solid black; padding: 8px;">NNC</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Proper Noun</td>
            <td style="border: 1px solid black; padding: 8px;">NNP</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Proper Noun Abbreviation</td>
            <td style="border: 1px solid black; padding: 8px;">NNPA</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Common Noun Abbreviation</td>
            <td style="border: 1px solid black; padding: 8px;">NNCA</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">as Subject (Palagyo)/Personal Pronouns Singular</td>
            <td style="border: 1px solid black; padding: 8px;">PRS</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Personal Pronouns</td>
            <td style="border: 1px solid black; padding: 8px;">PRP</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Possessive Subject (Paari)</td>
            <td style="border: 1px solid black; padding: 8px;">PRSP</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Pointing to an Object Demonstrative/(Paturo)/Pamatlig</td>
            <td style="border: 1px solid black; padding: 8px;">PRO</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Question/Interrogative (Pananong)/Singular</td>
            <td style="border: 1px solid black; padding: 8px;">PRQ</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Question/Interrogative Plural</td>
            <td style="border: 1px solid black; padding: 8px;">PRQP</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Location (Panlunan)</td>
            <td style="border: 1px solid black; padding: 8px;">PRL</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Comparison (Panulad)</td>
            <td style="border: 1px solid black; padding: 8px;">PRC</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Found (Pahimaton)</td>
            <td style="border: 1px solid black; padding: 8px;">PRF</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Indefinite</td>
            <td style="border: 1px solid black; padding: 8px;">PRI</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Determiner</td>
            <td style="border: 1px solid black; padding: 8px;">DT</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Determiner (Pantukoy) for Common Noun Plural</td>
            <td style="border: 1px solid black; padding: 8px;">DTC</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Determiner (Pantukoy) for Proper Noun</td>
            <td style="border: 1px solid black; padding: 8px;">DTP</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Determiner (Pantukoy) for Proper Noun Plural</td>
            <td style="border: 1px solid black; padding: 8px;">DTPP</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Lexical Marker</td>
            <td style="border: 1px solid black; padding: 8px;">LM</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Conjunctions (Pang-ugnay)</td>
            <td style="border: 1px solid black; padding: 8px;">CC, CCT, CCR, CCB, CCA</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Ligatures (Pang-angkop)</td>
            <td style="border: 1px solid black; padding: 8px;">CCP</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Preposition (Pang-ukol)</td>
            <td style="border: 1px solid black; padding: 8px;">CCU</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Verb (Pandiwa)</td>
            <td style="border: 1px solid black; padding: 8px;">VB</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Neutral/Infinitive</td>
            <td style="border: 1px solid black; padding: 8px;">VBW</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Auxiliary, Modal/Pseudo-verbs</td>
            <td style="border: 1px solid black; padding: 8px;">VBS</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Existential</td>
            <td style="border: 1px solid black; padding: 8px;">VBH</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Non-existential</td>
            <td style="border: 1px solid black; padding: 8px;">VBN</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Time Past (Perfective)</td>
            <td style="border: 1px solid black; padding: 8px;">VBTS</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Time Present (Imperfective)</td>
            <td style="border: 1px solid black; padding: 8px;">VBTR</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Time Future (Contemplative)</td>
            <td style="border: 1px solid black; padding: 8px;">VBTF</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Recent Past</td>
            <td style="border: 1px solid black; padding: 8px;">VBTP</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Actor Focus</td>
            <td style="border: 1px solid black; padding: 8px;">VBAF</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Object/Goal Focus</td>
            <td style="border: 1px solid black; padding: 8px;">VBOF</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Benefactive Focus</td>
            <td style="border: 1px solid black; padding: 8px;">VBOB</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Locative Focus</td>
            <td style="border: 1px solid black; padding: 8px;">VBOL</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Instrumental Focus</td>
            <td style="border: 1px solid black; padding: 8px;">VBOI</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Referential/Measurement Focus</td>
            <td style="border: 1px solid black; padding: 8px;">VBRF</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Adjective (Pang-uri)</td>
            <td style="border: 1px solid black; padding: 8px;">JJ</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Describing (Panlarawan)</td>
            <td style="border: 1px solid black; padding: 8px;">JJD</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Used for Comparison (same level) (Pahambing Magkatulad)</td>
            <td style="border: 1px solid black; padding: 8px;">JJC</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Comparison Comparative (more) (Palamang)</td>
            <td style="border: 1px solid black; padding: 8px;">JJCC</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Comparison Superlative (most) (Pasukdol)</td>
            <td style="border: 1px solid black; padding: 8px;">JJCS</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Comparison Negation (not quite) (Di-Magkatulad)</td>
            <td style="border: 1px solid black; padding: 8px;">JJCN</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Describing Number (Pamilang)</td>
            <td style="border: 1px solid black; padding: 8px;">JJN</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Adverb (Pang-Abay)</td>
            <td style="border: 1px solid black; padding: 8px;">RB</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Describing “How” (Pamaraang)</td>
            <td style="border: 1px solid black; padding: 8px;">RBD</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Number (Panggaano/Panukat)</td>
            <td style="border: 1px solid black; padding: 8px;">RBN</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Conditional (Kondisyunal)</td>
            <td style="border: 1px solid black; padding: 8px;">RBK</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Causative (Pananhi)</td>
            <td style="border: 1px solid black; padding: 8px;">RBP</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Benefactive (Benepaktibo)</td>
            <td style="border: 1px solid black; padding: 8px;">RBB</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Referential (Pangkaukulan)</td>
            <td style="border: 1px solid black; padding: 8px;">RBR</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Question (Pananong)</td>
            <td style="border: 1px solid black; padding: 8px;">RBQ</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Agree (Panang-ayon)</td>
            <td style="border: 1px solid black; padding: 8px;">RBT</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Disagree (Pananggi)</td>
            <td style="border: 1px solid black; padding: 8px;">RBF</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Frequency (Pamanahon)</td>
            <td style="border: 1px solid black; padding: 8px;">RBW</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Possibility (Pang-agam)</td>
            <td style="border: 1px solid black; padding: 8px;">RBM</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Place (Panlunan)</td>
            <td style="border: 1px solid black; padding: 8px;">RBL</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Enclitics (Paningit)</td>
            <td style="border: 1px solid black; padding: 8px;">RBI</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Interjections (Sambitla)</td>
            <td style="border: 1px solid black; padding: 8px;">RBJ</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Social Formula (Pormularyong Panlipunan)</td>
            <td style="border: 1px solid black; padding: 8px;">RBS</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Cardinal Number (Bilang)</td>
            <td style="border: 1px solid black; padding: 8px;">CD</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Digit, Rank, Count</td>
            <td style="border: 1px solid black; padding: 8px;">CDB</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Topicless (Walang Paksa)</td>
            <td style="border: 1px solid black; padding: 8px;">TS</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Foreign Words</td>
            <td style="border: 1px solid black; padding: 8px;">FW</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Punctuation (Pananda)</td>
            <td style="border: 1px solid black; padding: 8px;">PM</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Period</td>
            <td style="border: 1px solid black; padding: 8px;">PMP</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Exclamation Point</td>
            <td style="border: 1px solid black; padding: 8px;">PME</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Question Mark</td>
            <td style="border: 1px solid black; padding: 8px;">PMQ</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Comma</td>
            <td style="border: 1px solid black; padding: 8px;">PMC</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Semi-colon</td>
            <td style="border: 1px solid black; padding: 8px;">PMSC</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 8px;">Symbols</td>
            <td style="border: 1px solid black; padding: 8px;">PMS</td>
        </tr>
    </tbody>
</table>

"""

with gr.Blocks(theme='ParityError/Interstellar') as tagger:
    gr.Markdown("<div style='margin-top: 20px; text-align: center; font-size: 2em; font-weight: bold;'>TAGALONGGO: A Part-of-Speech (POS) Tagger For Tagalog-Ilonggo Texts Using Bilingual BERT</div>")

    with gr.Row(): 
        with gr.Column(): 
            sentence_input = gr.Textbox(placeholder="Enter sentence here...", label="Input")
            model_input_dropdown = gr.Dropdown(choices=list(model_paths.keys()), label="Select Model", value="SSP with Augmentation")
            
            with gr.Row():
                clear_button = gr.Button("Clear")
                clear_button.click(fn=clear_function, outputs=[sentence_input, model_input_dropdown])  # Set outputs for clearing
                submit_button = gr.Button("Submit")

            example_text = gr.Examples(
                examples=[
                    ["Luyag ko mag-bakasyon sa iloilo kay damo sang magagandang lugar."],
                    ["Nagbakal ako ng bakal."]
                ],
                inputs=[sentence_input, model_input_dropdown]
            )

        with gr.Column():  
            """ tagged_output = gr.HighlightedText(label="Tagged Sentence") """
            tagged_output = gr.JSON(label="Tagged Texts:")  
            

    submit_button.click(fn=predict_tags, inputs=[sentence_input, model_input_dropdown], outputs=tagged_output) 
    
    gr.HTML(html_table)

tagger.launch(favicon_path="favicon.png", share=True)