import re
from collections import defaultdict

# Function to count POS tags in the sentence
def count_tags_from_sentence(tagged_sentence):
    # This regex will capture the POS tag before the word
    regex_pattern = r'<([^<> ]+)[^<>]*>'
    
    # Find all matches of the POS tags
    pos_tags = re.findall(regex_pattern, tagged_sentence)

    return pos_tags

# Function to process the input file and count all POS tags
def process_input_file(input_file):
    pos_tag_counts = defaultdict(int)  # Dictionary to hold counts of each POS tag

    with open(input_file, 'r') as file:
        lines = file.readlines()

    # For each line in the file, extract the POS tags and count them
    for line in lines:
        pos_tags = count_tags_from_sentence(line.strip())
        for tag in pos_tags:
            pos_tag_counts[tag] += 1

    return pos_tag_counts

# Main code to run the program
if __name__ == "__main__":
    # Update this to the correct path of your input file
    input_file = "./Dataset/Labeled Corpus/eval-set-24.txt"

    # Process the file and get POS tag counts
    pos_tag_counts = process_input_file(input_file)

    # Output the POS tag counts
    print("POS Tag Counts:")
    for pos, count in pos_tag_counts.items():
        print(f"{pos}: {count}")
