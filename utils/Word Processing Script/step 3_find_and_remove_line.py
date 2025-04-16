import re

def remove_newline_after_brgy(input_file, output_file):

    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
    
    text = re.sub(r'(Brgy\.)\s*\s*\s*\n', r'\1 ', text)
    text = re.sub(r'(Occ\.)\s*\s*\s*\n', r'\1 ', text)
    text = re.sub(r'(Jr\.)\s*\s*\s*\n', r'\1 ', text)
    text = re.sub(r'(Sr\.)\s*\s*\s*\n', r'\1 ', text)
    text = re.sub(r'(Adm\.)\s*\s*\s*\n', r'\1 ', text)
    text = re.sub(r'(Col\.)\s*\s*\s*\n', r'\1 ', text)
    text = re.sub(r'(Gen\.)\s*\s*\s*\n', r'\1 ', text)
    text = re.sub(r'(Lt\.)\s*\s*\s*\n', r'\1 ', text)
    text = re.sub(r'(\d\.)\s*\s*\s*\n', r'\1 ', text)
    text = re.sub(r'([A-Z])\s*\s*\s*\.', r'\1. ', text)
    text = re.sub(r'([a-z])\s*\s*\s*\.', r'\1. ', text)

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(text)

input_file = './dataset/output_file 51-53-dataset.txt'
output_file = './dataset/output_file 51-53-dataset.txt'

remove_newline_after_brgy(input_file, output_file)

print(f"Done. Check '{output_file}' for results.")