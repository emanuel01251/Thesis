import os
import random
import re

# Define punctuation tags to exclude from replacement
PUNCTUATION_TAGS = {"PUNCT"}

# Function to extract tagsets and words from a labeled sentence
def extract_tagsets(sentence):
    return re.findall(r'<[^>]+>', sentence)

# Function to extract the tag and the word from a tagset, with proper error handling
def extract_words_from_tagsets(tagset):
    match = re.match(r'<(\w+)\s+([^>]+)>', tagset)  # Adjusted the regex
    if match:
        return match.group(1), match.group(2)  # Group 1 is the tag, Group 2 is the word
    else:
        print(f"Warning: Couldn't match the tagset pattern for: {tagset}")
        return None, None  # Handle cases where the tagset doesn't match the pattern

# Function to replace a tagset with another
def replace_tagset(original_sentence, original_tagset, replacement_tagset):
    return original_sentence.replace(original_tagset, replacement_tagset, 1)

# Function to augment a sentence by replacing tagsets
def augment_sentence(sentences, sentence, replacement_percentage=0.5, augment_count=1):
    original_tagsets = extract_tagsets(sentence)
    augmented_sentences = []

    # Find replacement tagsets from other sentences
    all_tagsets = {}
    
    for s in sentences:
        if s != sentence:  # avoid using the same sentence for replacement
            tagsets_in_sentence = extract_tagsets(s)
            for tagset in tagsets_in_sentence:
                tag, word = extract_words_from_tagsets(tagset)
                if tag and word and tag not in PUNCTUATION_TAGS:  # Exclude punctuation tags
                    if tag not in all_tagsets:
                        all_tagsets[tag] = []
                    all_tagsets[tag].append(tagset)

    # For each augmentation
    for _ in range(augment_count):
        augmented_sentence = sentence

        # Randomly select tagsets for replacement based on the percentage
        replace_count = max(1, int(len(original_tagsets) * replacement_percentage))
        tagsets_to_replace = random.sample(original_tagsets, replace_count)

        for tagset in tagsets_to_replace:
            tag, word = extract_words_from_tagsets(tagset)
            if tag in all_tagsets and all_tagsets[tag]:  # If we have other tagsets of the same tag type
                replacement_tagset = random.choice(all_tagsets[tag])
                augmented_sentence = replace_tagset(augmented_sentence, tagset, replacement_tagset)

        # Remove any trailing spaces from the augmented sentence
        augmented_sentences.append(augmented_sentence.rstrip())
    
    return augmented_sentences

# Function to augment the entire dataset
def augment_dataset(sentences, replacement_percentage=0.5, augment_count=1):
    augmented_data = []
    
    for sentence in sentences:
        # Original sentence is preserved
        augmented_data.append(sentence.strip())

        # Generate augmented sentences
        augmented_sentences = augment_sentence(sentences, sentence, replacement_percentage, augment_count)
        augmented_data.extend(augmented_sentences)
    
    return augmented_data

# Main process
def process_file(input_file_path, output_file_path, replacement_percentage=0.5, augment_count=1):
    # Read input data
    with open(input_file_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()

    # Augment the dataset
    augmented_data = augment_dataset(sentences, replacement_percentage, augment_count)

    # Write the augmented data to the output file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for sentence in augmented_data:
            file.write(sentence + '\n')

    print(f"Augmented sentences saved to {output_file_path}")

# Define paths for input and output files
current_dir = os.getcwd()
input_file_path = os.path.join(current_dir, "./Dataset/Labeled Corpus/hil-train-set-less-tagset.txt")  # Change this to your input file
output_file_path = os.path.join(current_dir, "./Dataset/Labeled Corpus/tssr-augmented-hil-train-set-less-tagset.txt")  # Output file

# Set augmentation parameters (e.g., 50% tagset replacement, 2 augmentations per sentence)
replacement_percentage = 1  # Change 50% of tagsets in each sentence
augment_count = 5

# Run the augmentation process
process_file(input_file_path, output_file_path, replacement_percentage, augment_count)
