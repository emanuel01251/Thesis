from transformers import BertTokenizer, BertForMaskedLM, BertForTokenClassification, AutoTokenizer, AutoModelForTokenClassification
import torch
import random
import re
from tqdm import tqdm

# Load the trained model and tokenizer
model_checkpoint = "./BERT-SSP-POS/BERTPOS_merged_less_2"

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

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
num_labels = len(pos_tag_mapping)
id2label = {idx: tag for tag, idx in pos_tag_mapping.items()}
label2id = {tag: idx for tag, idx in pos_tag_mapping.items()}

special_symbols = ['-', '&', "\"", "[", "]", "/", "$", "(", ")", "%", ":", "'", '.', '?', ',']

def symbol2token(symbol):

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
                new_sentence += '[PUNCT]'
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

    # print("Sentence after:", new_sentence.split())
    # print("---")

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


    # # print("---")
    # # print("Sentence before:", sentence)

    # Loop through the sentence and convert hyphen to '[PMP]' if the next character is a space
    new_sentence = ""
    i = 0
    # # print("Length: ", len(sentence))
    while i < len(sentence):
        # # print(f"{i+1} == {len(sentence)}: {sentence[i]}")

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

    # print("Sentence after:", new_sentence.split())
    # print("---")

    return new_sentence
def extract_tags(input_sentence):
    tags = re.findall(r'<([A-Z_]+)\s.*?>', input_sentence)
    return tags

def align_tokenization(sentence, tags):

    # print("Sentence \n: ", sentence)
    sentence = sentence.split()
    # print("Sentence Split\n: ", sentence)

    tokenized_sentence = tokenizer.tokenize(' '.join(sentence))
    # tokenized_sentence_string = " ".join(tokenized_sentence)
    # # print("ID2Token_string\n: ", tokenized_sentence_string)

    aligned_tagging = []
    current_word = ''
    index = 0  # index of the current word in the sentence and tagging

    for token in tokenized_sentence:
        current_word += re.sub(r'^##', '', token)
        # print("Current word after replacing ##: ", current_word)
        # print("sentence[index]: ", sentence[index])

        if sentence[index] == current_word:  # if we completed a word
            # print("completed a word: ", current_word)
            current_word = ''
            aligned_tagging.append(tags[index])
            index += 1
        else:  # otherwise insert padding
            # print("incomplete word: ", current_word)
            aligned_tagging.append(0)

        # print("---")

    decoded_tags = [list(pos_tag_mapping.keys())[list(pos_tag_mapping.values()).index(tag_id)] for tag_id in
                    aligned_tagging]
    # print("Tokenized Sentence\n: ", tokenized_sentence)
    # print("Tags\n: ", decoded_tags)

    assert len(tokenized_sentence) == len(aligned_tagging)

    aligned_tagging = [0] + aligned_tagging
    return tokenized_sentence, aligned_tagging


def process_tagged_sentence(tagged_sentence):
    # # print(tagged_sentence)

    # Preprocess the input tagged sentence and extract the words and tags
    sentence = preprocess_sentence(tagged_sentence)
    tags = extract_tags(tagged_sentence) # returns the tags (eto ilagay mo sa tags.txt)


    encoded_tags = [pos_tag_mapping[tag] for tag in tags]

    # Align tokens by adding padding if needed
    tokenized_sentence, encoded_tags = align_tokenization(sentence, encoded_tags)
    encoded_sentence = tokenizer(sentence, padding="max_length" ,truncation=True, max_length=128)

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = [1] * len(encoded_sentence['input_ids'])
    # print("len(encoded_sentence['input_ids']):", len(encoded_sentence['input_ids']))
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
    # print(encoded_sentence)
    # print("Sentence:", decoded_sentence)
    # print("Tags:", decoded_tags)
    # print("Decoded Sentence and Tags:", word_tag_pairs)
    # print("---")

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

def predict_tags(test_sentence):

    sentence, upper = preprocess_untagged_sentence(test_sentence)
    words_list = upper.split()
    # print("Words: ", words_list)
    predicted_tags = tag_sentence(test_sentence)
    # print(predicted_tags)

    pairs = list(zip(words_list, predicted_tags))
    return pairs   

class TagalonggoDataAugmentor:
    def __init__(self, 
                 ssp_model_path,          # Your BERT SSP model path
                 pos_tagger_path,         # Your fine-tuned POS tagger path
                 mask_probability=0.8,
                 temperature=2.0):
        
        # Load SSP model for MLM
        vocab_file = "./Tokenizer/corpus-merged/uncased-vocab.txt"
        self.tokenizer = BertTokenizer(vocab_file=vocab_file)
        self.mlm_model = BertForMaskedLM.from_pretrained(ssp_model_path)
        self.temperature = temperature
        # Load POS tagger
        self.pos_tagger = AutoModelForTokenClassification.from_pretrained(pos_tagger_path)
        
        # POS tag masking probabilities
        self.pos_mask_probs = {
            'NN': 1,    # Nouns - highest probability
            'VB': 1,    # Verbs
            'ADJ': 1,   # Adjectives
            'ADV': 0,   # Adverbs
            'PRNN': 0,  # Pronouns
            'DET': 0,   # Determiners
            'NUM': 0,   # Numbers
            'CONJ': 0, # Conjunctions/Prepositions
            'PUNCT': 0,   # Don't mask punctuation
            'FW': 0     # Foreign Words
        }
        
        self.base_mask_prob = mask_probability

    def parse_tagged_sentence(self, tagged_text):
        """Parse tagged text into words and their POS tags"""
        pattern = r'<(\w+)\s+([^>]+)>'
        return re.findall(pattern, tagged_text)  # Returns list of (tag, word) tuples

    def create_masked_sentence(self, tagged_sentence):
        """Create masked version of sentence based on POS tags"""
        tokens = self.parse_tagged_sentence(tagged_sentence)
        masked_words = []
        original_tokens = []
        mask_positions = []

        for i, (pos, word) in enumerate(tokens):
            mask_prob = self.pos_mask_probs.get(pos, 0) * self.base_mask_prob
            
            if random.random() < mask_prob:
                masked_words.append(self.tokenizer.mask_token)
                original_tokens.append((pos, word))
                mask_positions.append(i)
            else:
                masked_words.append(word)

        ## print(' '.join(masked_words), original_tokens, mask_positions)
        return ' '.join(masked_words), original_tokens, mask_positions
    
    def generate_new_sentences(self, input_sentence):
        """Generate new sentence using sequential masking and prediction"""
        original_word_tags = self.parse_tagged_sentence(input_sentence)
        total_words = len(original_word_tags)
        target_masks = int(total_words * 0.4)
        
        # Keep original words and their replacements
        original_words = [word for _, word in original_word_tags]
        final_words = original_words.copy()
        
        # Get maskable positions
        maskable_positions = [
            i for i, (tag, _) in enumerate(original_word_tags) 
            if self.pos_mask_probs.get(tag, 0) > 0
        ]
        
        random.shuffle(maskable_positions)
        
        # Process each position
        for i in maskable_positions[:target_masks]:
            tag, word = original_word_tags[i]
            
            # Start with original sentence each time
            temp_words = original_words.copy()
            temp_words[i] = self.tokenizer.mask_token
            masked_sent = ' '.join(temp_words)
            
            # Tokenize input properly
            inputs = self.tokenizer.encode_plus(
                masked_sent,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Get mask token index
            mask_index = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]
            
            with torch.no_grad():
                outputs = self.mlm_model(**inputs)
                predictions = outputs.logits
                
                # Get probabilities for masked position
                mask_predictions = predictions[0, mask_index]
                probs = torch.nn.functional.softmax(mask_predictions, dim=-1)
                
                # Get top prediction
                top_prob, top_idx = torch.topk(probs, 1, dim=-1)
                predicted_token_id = top_idx[0].item()
                predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_token_id])[0]
                
                # Convert token to word
                predicted_word = self.tokenizer.convert_tokens_to_string([predicted_token])
                predicted_word = predicted_word.strip()
                
                # Apply original case format
                if word[0].isupper():
                    predicted_word = predicted_word.capitalize()
                
                # Store prediction in final words
                final_words[i] = predicted_word
        
        # Create untagged augmented sentence
        augmented_sentence = ' '.join(final_words)
        
        # Tokenize the augmented sentence
        sentence, upper = preprocess_untagged_sentence(augmented_sentence)
        words_list = upper.split()
        # print("Words: ", words_list)
        predicted_tags = tag_sentence(augmented_sentence)
        
        # Create final tagged sentence
        words = augmented_sentence.split()
        tagged_tokens = []

        for word, (tag) in zip(words, predicted_tags):
            tagged_tokens.append(f'<{tag} {word}>')
        
        return [' '.join(tagged_tokens)]

def augment_dataset(input_file, output_file, ssp_model_path, pos_tagger_path, num_augmentations=1):
    """Main function to augment the dataset"""
    augmentor = TagalonggoDataAugmentor(ssp_model_path, pos_tagger_path)
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        original_sentences = [line.strip() for line in f if line.strip()]
    
    augmented_data = []
    
    # Process each sentence
    for sentence in tqdm(original_sentences, desc="Augmenting dataset"):
        # Keep original sentence
        augmented_data.append(sentence)
        
        # Generate one new sentence
        new_sentence = augmentor.generate_new_sentences(sentence)[0]
        augmented_data.append(new_sentence)
    
    # Save augmented dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in augmented_data:
            f.write(sentence + '\n')

# Usage example:
ssp_model_path = "./BERT-SSP/output_model_merged/checkpoint-331500/"
pos_tagger_path = "./BERT-SSP-POS/BERTPOS_merged_less_2"

augment_dataset(
    input_file="./Dataset/Labeled Corpus/hil-train-set-less-tagset.txt",
    output_file="./Dataset/Labeled Corpus/augmented-hil-train-set-less-tagset.txt",
    ssp_model_path=ssp_model_path,
    pos_tagger_path=pos_tagger_path,
    num_augmentations=1
)