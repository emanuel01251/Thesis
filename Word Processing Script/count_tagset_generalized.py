import re
from collections import defaultdict

# Define the generalized POS categories with their associated tags
generalized_pos = {
    "Noun": ["NNC", "NNP", "NNPA", "NNCA"],
    "Pronoun": ["PR", "PRS", "PRP", "PRSP", "PRO", "PRL", "PRC"],
    "Determiner": ["DT", "DTC", "DTP"],
    "Lexical Marker": ["LM"],
    "Conjunctions": ["CJN", "CCP", "CCU"],
    "Verb": ["VB", "VBW", "VBS", "VBH", "VBN", "VBTS", "VBTR", "VBTF"],
    "Adjective": ["JJ", "JJD", "JJC", "JJCC", "JJCS", "JJN"],
    "Adverb": ["RB", "RBD", "RBN", "RBK", "RBR", "RBQ", "RBT", "RBF", "RBW", "RBM", "RBL", "RBI"],
    "Digit, Rank, Count": ["CDB"],
    "Foreign Words": ["FW"],
    "Punctuation": ["PM", "PMP", "PME", "PMQ", "PMC", "PMSC", "PMS"]
}

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
    generalized_counts = defaultdict(int)  # Dictionary to hold counts of generalized categories

    with open(input_file, 'r') as file:
        lines = file.readlines()

    # For each line in the file, extract the POS tags and count them
    for line in lines:
        pos_tags = count_tags_from_sentence(line.strip())
        for tag in pos_tags:
            pos_tag_counts[tag] += 1

            # Generalize the tag and count it under the correct POS category
            for gen_pos, tags in generalized_pos.items():
                if tag in tags:
                    generalized_counts[gen_pos] += 1
                    break

    return pos_tag_counts, generalized_counts

# Main code to run the program
if __name__ == "__main__":
    # Update this to the correct path of your input file
    input_file = "./Dataset/Labeled Corpus/hil-eval-set.txt"

    # Process the file and get POS tag counts
    pos_tag_counts, generalized_counts = process_input_file(input_file)

    # Output the specific POS tag counts
    print("Specific POS Tag Counts:")
    for pos, count in pos_tag_counts.items():
        print(f"{pos}: {count}")

    # Output the generalized POS counts
    print("\nGeneralized POS Counts:")
    for gen_pos, count in generalized_counts.items():
        print(f"{gen_pos}: {count}")