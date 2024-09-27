import re

def count_sentences_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Regular expression to match sentences based on punctuation marks
        sentences = re.split(r'[.!?]+', text)
        
        # Remove any empty strings from the resulting list
        sentences = [s for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        print(f"Total sentence count: {sentence_count}")
        return sentence_count

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None

# Example usage
file_path = 'Tagalog_Hiligaynon_Literary_Text.txt'
count_sentences_in_file(file_path)
