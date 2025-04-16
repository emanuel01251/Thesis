def count_words_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        words = text.split()
        word_count = len(words)
        
        print(f"Total word count: {word_count}")
        return word_count

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None

file_path = './Dataset/Unlabeled Corpus/Merged Corpus.txt'  # Replace with your file path
count_words_in_file(file_path)