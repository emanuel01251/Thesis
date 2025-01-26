from transformers import BertTokenizer, BertForMaskedLM, BertForTokenClassification, AutoTokenizer, AutoModelForTokenClassification
import torch
import random
import re
from tqdm import tqdm

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

        #print(' '.join(masked_words), original_tokens, mask_positions)
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
        encoded_sentence = self.tokenizer(
            augmented_sentence,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Get POS tags
        with torch.no_grad():
            outputs = self.pos_tagger(**encoded_sentence)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_tags = torch.argmax(probabilities, dim=-1)
        
        # Get scores and labels
        scores = probabilities[0].max(dim=-1)[0].tolist()
        labels = [self.pos_tagger.config.id2label[tag.item()] for tag in predicted_tags[0]]
        
        # Filter out padding tokens
        filtered_pairs = [(labels[i], scores[i]) 
                        for i in range(len(labels)) 
                        if labels[i] != '[PAD]']
        
        # Create final tagged sentence
        words = augmented_sentence.split()
        tagged_tokens = []
        
        for word, (tag, score) in zip(words, filtered_pairs):
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
    output_file="./Dataset/Labeled Corpus/augmented-hil-train-set-less-tagset-validated.txt",
    ssp_model_path=ssp_model_path,
    pos_tagger_path=pos_tagger_path,
    num_augmentations=1
)