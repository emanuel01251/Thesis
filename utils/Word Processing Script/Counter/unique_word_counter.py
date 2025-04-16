import re

def count_unique_words_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        words = re.findall(r'\b\w+\b', text.lower())

        unique_words = set(words)
        unique_word_count = len(unique_words)
        
        print(f"Total unique word count: {unique_word_count}")
        return unique_word_count

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None

file_path = './Tagalog_Hiligaynon_Literary_Text.txt'
count_unique_words_in_file(file_path)