import random
from gensim.models import Word2Vec
import re

def preprocess_text(file_path):
    sentences = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Skip headers and empty lines
            if line.startswith('[TAGALOG]') or line.startswith('[HILIGAYNON]') or \
               line.startswith('Title:') or not line.strip():
                continue
            
            # Basic preprocessing
            line = line.strip().lower()
            # Remove punctuation if needed
            line = re.sub(r'[^\w\s]', '', line)
            # Split into words
            words = line.split()
            if words:  # Only add non-empty sentences
                sentences.append(words)
    
    return sentences

# Train word2vec
def train_word2vec(sentences):
    model = Word2Vec(
        sentences=sentences,
        vector_size=300,  # embedding dimension
        window=5,         # context window size
        min_count=1,      # minimum word frequency
        workers=4         # number of threads
    )
    return model

# Train the model
corpus_path = './Dataset/Unlabeled Corpus/Merged Corpus.txt'
sentences = preprocess_text(corpus_path)
model = train_word2vec(sentences)

# Save the model
model.save("tagalog_hiligaynon_word2vec.model")

# Function for POS augmentation
def augment_pos_sentence(sentence, pos_tags):
    augmented_sentence = []
    augmented_pos = []
    
    for word, pos in zip(sentence, pos_tags):
        if random.random() < 0.2:  # 20% chance to replace
            try:
                similar_words = model.wv.most_similar(word.lower())
                # Take first similar word
                new_word = similar_words[0][0]
                augmented_sentence.append(new_word)
                augmented_pos.append(pos)
            except:
                augmented_sentence.append(word)
                augmented_pos.append(pos)
        else:
            augmented_sentence.append(word)
            augmented_pos.append(pos)
    
    return augmented_sentence, augmented_pos

# Example usage
test_words = ["tao", "balay", "tawo", "bahay"]  # Mix of Tagalog and Hiligaynon words
for word in test_words:
    try:
        similar = model.wv.most_similar(word)
        print(f"\nSimilar to '{word}':")
        for sim_word, score in similar[:5]:
            print(f"{sim_word}: {score:.4f}")
    except KeyError:
        print(f"\nWord '{word}' not found in vocabulary")