import gensim
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

class Word2VecSimilarityTester:
    def __init__(self, model_path):
        """Initialize with a trained Word2Vec model"""
        try:
            self.model = Word2Vec.load(model_path)
            print(f"Loaded model with vocabulary size: {len(self.model.wv.key_to_index)}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def get_word_vector(self, word):
        """Get the vector for a word if it exists in vocabulary"""
        try:
            return self.model.wv[word]
        except KeyError:
            print(f"Word '{word}' not found in vocabulary")
            return None

    def compute_cosine_similarity(self, word1, word2):
        """Compute cosine similarity between two words"""
        vec1 = self.get_word_vector(word1)
        vec2 = self.get_word_vector(word2)
        
        if vec1 is None or vec2 is None:
            return None
        
        # Reshape vectors for sklearn's cosine_similarity
        vec1 = vec1.reshape(1, -1)
        vec2 = vec2.reshape(1, -1)
        
        similarity = cosine_similarity(vec1, vec2)[0][0]
        return similarity

    def find_similar_words(self, word, n=10):
        """Find n most similar words to the given word"""
        try:
            similar_words = self.model.wv.most_similar(word, topn=n)
            return similar_words
        except KeyError:
            print(f"Word '{word}' not found in vocabulary")
            return None

    def test_word_pairs(self, word_pairs):
        """Test multiple word pairs and return results"""
        results = []
        for word1, word2 in word_pairs:
            similarity = self.compute_cosine_similarity(word1, word2)
            if similarity is not None:
                results.append({
                    'word1': word1,
                    'word2': word2,
                    'similarity': similarity
                })
        return results

    def visualize_similarities(self, results):
        """Create a heatmap visualization of word similarities"""
        # Extract unique words
        words = list(set([r['word1'] for r in results] + [r['word2'] for r in results]))
        n_words = len(words)
        
        # Create similarity matrix
        similarity_matrix = np.zeros((n_words, n_words))
        word_to_idx = {word: idx for idx, word in enumerate(words)}
        
        # Fill similarity matrix
        for result in results:
            i = word_to_idx[result['word1']]
            j = word_to_idx[result['word2']]
            similarity_matrix[i, j] = result['similarity']
            similarity_matrix[j, i] = result['similarity']  # Matrix is symmetric
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, 
                   xticklabels=words, 
                   yticklabels=words, 
                   annot=True, 
                   cmap='YlOrRd')
        plt.title('Word Similarity Heatmap')
        plt.tight_layout()
        plt.show()

def main():
    # Initialize tester with your trained model
    tester = Word2VecSimilarityTester('./Word2Vec/Bilingual/bilingual_tagalog_hiligaynon.model')
    #tester = Word2VecSimilarityTester('./Word2Vec/Tagalog/word2vec_300dim_20epochs.model')
    
    # Define word pairs to test (Tagalog-Hiligaynon pairs)
    word_pairs = [
        ('bahay', 'balay'),     # house
        ('tubig', 'tubig'),     # water
        ('salamat', 'salamat'), # thank you
        ('maganda', 'matahum'), # beautiful
        ('araw', 'adlaw'),      # sun/day
        # Add more word pairs as needed
    ]
    
    # Test word pairs
    print("\nTesting word pairs...")
    results = tester.test_word_pairs(word_pairs)
    
    # Display results in a table
    if results:
        table_data = [[r['word1'], r['word2'], f"{r['similarity']:.4f}"] for r in results]
        print("\nSimilarity Results:")
        print(tabulate(table_data, 
                      headers=['Word 1', 'Word 2', 'Similarity'],
                      tablefmt='grid'))
        
        # Visualize similarities
        #tester.visualize_similarities(results)
    
    # Example of finding similar words
    test_word = 'kaupod'  # or any word you want to test
    print(f"\nFinding words similar to '{test_word}':")
    similar_words = tester.find_similar_words(test_word)
    if similar_words:
        for word, score in similar_words:
            print(f"{word}: {score:.4f}")

if __name__ == "__main__":
    main()