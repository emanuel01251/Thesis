import re

def remove_extra_spaces(input_file, output_file):
    with open(input_file, 'r') as infile:
        text = infile.read()
        
    cleaned_text = re.sub(r'\n', '\n\n', text)
    cleaned_text = cleaned_text.strip()

    with open(output_file, 'w') as outfile:
        outfile.write(cleaned_text)

    print(f"Cleaned text has been written to '{output_file}'.")

if __name__ == "__main__":
    input_file = './Dataset/train.txt'
    output_file = './Dataset/with spaces train.txt'
    
    remove_extra_spaces(input_file, output_file)