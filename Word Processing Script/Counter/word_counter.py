def count_words_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Split text into words and count
        words = text.split()
        word_count = len(words)
        
        print(f"Total word count: {word_count}")
        return word_count

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None

# Example usage
file_path = './Dataset/Hiligaynon Corpus.txt'  # Replace with your file path
count_words_in_file(file_path)