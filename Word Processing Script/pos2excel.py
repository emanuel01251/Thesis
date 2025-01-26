import pandas as pd
import re

def process_pos_file(input_file):
    # Initialize lists to store data
    data = []
    current_sentence_num = 1
    
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                # Get the full sentence without tags for reference
                full_sentence = ' '.join(word.split('>')[0].split()[1] for word in line.strip().split('<')[1:])
                
                # Process each word-tag pair
                tokens = line.strip().split('<')[1:]  # Split by '<' and remove empty first element
                
                # For each token in the sentence
                for i, token in enumerate(tokens):
                    if token.strip():
                        # Extract POS tag and word
                        match = re.match(r'(\w+)\s+([^>]+)>', token.strip())
                        if match:
                            pos_tag, word = match.groups()
                            
                            # Add to data list - only include number and sentence for first word
                            data.append({
                                'Number': current_sentence_num if i == 0 else '',
                                'Sentence': full_sentence if i == 0 else '',
                                'Word': word,
                                'POS Label': pos_tag,
                                'Comments': ''  # Empty column for comments
                            })
                
                current_sentence_num += 1
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Export to Excel
    df.to_excel('pos_annotation.xlsx', index=False)
    print("Excel file 'pos_annotation.xlsx' has been created successfully!")

# Usage
input_file = './Dataset/Labeled Corpus/hil-train-set-less-tagset.txt'  # Replace with your input file name
process_pos_file(input_file)