import re

def count_sentences_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        sentences = re.split(r'[.!?]+', text)
        
        sentences = [s for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        print(f"Total sentence count: {sentence_count}")
        return sentence_count

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None

file_path = './dataset/Unlabeled Corpus/Merged Corpus.txt'
count_sentences_in_file(file_path)
