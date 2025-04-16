import re

def remove_tags_from_sentence(tagged_sentence):
    regex_pattern = r'<(?:[^<>]+? )?([a-zA-Z0-9.,&"!?{}]+)>'
    
    words = re.findall(regex_pattern, tagged_sentence)

    clean_sentence = ' '.join(words)
    
    return clean_sentence

def process_input_file(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    clean_sentences = []
    
    for line in lines:
        clean_sentence = remove_tags_from_sentence(line.strip())
        clean_sentences.append(clean_sentence)

    return clean_sentences


import os

current_dir = os.getcwd()
current_dir += "./Dataset/Labeled Corpus/hil-eval-set.txt"
cleaned_sentences = process_input_file(current_dir)

output_file = "cleaned_sentences hil-eval-set.txt"
with open(output_file, 'w') as f:
    for sentence in cleaned_sentences:
        f.write(sentence + '\n')