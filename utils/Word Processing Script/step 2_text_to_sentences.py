import re

def split_text_into_sentences(file_path, output_file):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        """ text = re.sub(r'\n{3,}', '\n', text) """
        """ text = re.sub(r'\.\s*\s*\s*\s*', '. ', text) """
        sentences = re.split(r'(?<!\b[A-Z]\.)[.!?] +', text)  # prevent splitting after initials
        
        sentences = re.split(r'(?<=[.!?]) +', text)

        with open(output_file, 'w', encoding='utf-8') as outfile:
            for sentence in sentences:
                outfile.write(sentence.strip() + '\n')

        print(f"Sentences have been written to {output_file}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None

input_file = './dataset/output_file 51-53-dataset.txt'
output_file = './dataset/output_file 51-53-dataset.txt'
split_text_into_sentences(input_file, output_file)
