import re

def remove_extra_spaces(input_file, output_file):
    with open(input_file, 'r') as infile:
        text = infile.read()
        
    #cleaned_text = re.sub(r'\n', '\n\n', text)
    cleaned_text = re.sub(r'\n\n', '\n', text)
    cleaned_text = cleaned_text.strip()

    with open(output_file, 'w') as outfile:
        outfile.write(cleaned_text)

    print(f"Cleaned text has been written to '{output_file}'.")

import os

if __name__ == "__main__":
    current_dir = os.getcwd()
    input_file = os.path.join(current_dir, "augmented hil-eval-set.txt")
    output_file = 'with_spaces augmented hil-eval-set.txt'
    
    remove_extra_spaces(input_file, output_file)