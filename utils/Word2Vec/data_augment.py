import re
from gensim.models import Word2Vec
import random

class TaggedPOSAugmenter:
    def __init__(self, word2vec_path):
        """Initialize augmenter with word2vec model"""
        self.model = Word2Vec.load(word2vec_path)
        # Define which POS tags can be augmented
        self.augmentable_tags = {'NN', 'PRNN', 'VB', 'ADV'}
        
    def parse_tagged_sentence(self, tagged_line):
        """Parse a line of tagged text into token-tag pairs"""
        # Extract tokens with their tags using regex
        pattern = r'<(\w+)\s+([^>]+)>'
        matches = re.findall(pattern, tagged_line)
        return [(word, tag) for tag, word in matches]
    
    def format_tagged_sentence(self, tokens_tags):
        """Convert token-tag pairs back to the original format"""
        return ' '.join(f'<{tag} {word}>' for word, tag in tokens_tags)
    
    def get_similar_word(self, word, tag, max_attempts=5):
        """
        Get similar word that:
        - maintains the same POS tag
        - has similarity >= 0.80
        - is different from the original word
        """
        if tag not in self.augmentable_tags:
            return word, False
            
        try:
            # Try multiple times to find a suitable word
            for _ in range(max_attempts):
                # Get similar words with their similarity scores
                similar_words = self.model.wv.most_similar(word.lower(), topn=20)  # Increased topn for more candidates
                
                # Filter words with similarity >= 0.80 and different from original word
                high_similarity_words = [
                    (w, score) for w, score in similar_words 
                    if score >= 0.80 and w.lower() != word.lower()
                ]
                
                if high_similarity_words:
                    # Return the most similar word that meets criteria
                    return high_similarity_words[0][0], True
                    
            # If no suitable word found after max attempts
            return word, False
                
        except KeyError:
            return word, False

    def synonym_replacement(self, tokens_tags, n_replacements=1, max_attempts=3):
        """Replace n words with similar words, ensuring minimum similarity threshold"""
        result = tokens_tags.copy()
        successful_replacements = 0
        attempts = 0
        
        while successful_replacements < n_replacements and attempts < max_attempts:
            # Get positions of augmentable words
            augmentable_positions = [
                i for i, (_, tag) in enumerate(result)
                if tag in self.augmentable_tags
            ]
            
            if not augmentable_positions:
                break
                
            # Randomly select a position to replace
            pos = random.choice(augmentable_positions)
            word, tag = result[pos]
            new_word, success = self.get_similar_word(word, tag)
            
            if success:
                result[pos] = (new_word, tag)
                successful_replacements += 1
            
            attempts += 1
                
        return result, successful_replacements > 0
    
    def random_swap(self, tokens_tags, n_swaps=1):
        """Randomly swap words with same POS tag"""
        result = tokens_tags.copy()
        # Group words by POS tag
        tag_positions = {}
        for i, (_, tag) in enumerate(result):
            if tag in self.augmentable_tags:
                tag_positions.setdefault(tag, []).append(i)
        
        # Perform swaps
        for _ in range(n_swaps):
            # Randomly select a tag that has multiple words
            valid_tags = [tag for tag, positions in tag_positions.items() 
                         if len(positions) > 1]
            if not valid_tags:
                continue
                
            tag = random.choice(valid_tags)
            pos1, pos2 = random.sample(tag_positions[tag], 2)
            result[pos1], result[pos2] = result[pos2], result[pos1]
        
        return result
    
    def augment_sentence(self, tagged_line, num_augmentations=4, max_attempts=5):
        """Generate multiple augmentations for a tagged sentence"""
        tokens_tags = self.parse_tagged_sentence(tagged_line)
        augmentations = []
        attempts = 0
        
        while len(augmentations) < num_augmentations and attempts < max_attempts:
            # Randomly choose augmentation technique
            if random.random() < 0.7:  # 70% chance for synonym replacement
                augmented, success = self.synonym_replacement(tokens_tags, n_replacements=5)
                if success:
                    augmented_line = self.format_tagged_sentence(augmented)
                    if augmented_line not in augmentations:  # Avoid duplicates
                        augmentations.append(augmented_line)
            else:  # 30% chance for word swap
                augmented = self.random_swap(tokens_tags, n_swaps=5)
                augmented_line = self.format_tagged_sentence(augmented)
                if augmented_line not in augmentations:  # Avoid duplicates
                    augmentations.append(augmented_line)
            
            attempts += 1
                
        return augmentations

def augment_dataset(input_file, output_file, word2vec_path, augmentations_per_sentence=1):
    """Augment entire dataset and save results"""
    augmenter = TaggedPOSAugmenter(word2vec_path)
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        # Write original data first
        original_lines = f_in.readlines()
        for line in original_lines:
            line = line.strip()
            if line:  # Skip empty lines
                f_out.write(line + '\n')
                
                # Keep trying until we get the desired number of unique augmentations
                augmentations = augmenter.augment_sentence(
                    line, 
                    num_augmentations=augmentations_per_sentence
                )
                for aug in augmentations:
                    f_out.write(aug + '\n')
        
if __name__ == "__main__":
    input_file = "./Dataset/Labeled Corpus/tl-hil-train-set-less-tagset.txt"
    output_file = "./Dataset/Labeled Corpus/tl_augmented_dataset_90.txt"
    word2vec_path = "./Word2Vec/Bilingual/bilingual_tagalog_hiligaynon.model"
    
    augment_dataset(input_file, output_file, word2vec_path)