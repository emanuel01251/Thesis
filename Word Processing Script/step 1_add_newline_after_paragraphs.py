import re

def add_newline_after_whitespace_breaks(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    new_text = re.sub(r'\.\s*\s*\s*\s*', '.', text)
    new_text = re.sub(r'\n{2,}', '', text)
    new_text = re.sub(r'\n{1,}', '\n\n', text)

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(new_text)

input_file = './dataset/51-53-dataset.txt'
output_file = './dataset/output_file 51-53-dataset.txt'

add_newline_after_whitespace_breaks(input_file, output_file)
