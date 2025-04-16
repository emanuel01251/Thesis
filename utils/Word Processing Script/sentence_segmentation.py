from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Load pre-trained BERT model and tokenizer for Tagalog
tokenizer = AutoTokenizer.from_pretrained("jcblaise/bert-tagalog-base-cased")
model = AutoModelForTokenClassification.from_pretrained("jcblaise/bert-tagalog-base-cased")

# Use the pipeline for NER or sentence segmentation
#nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# Segment sentences
def segment_sentences(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = file.read()
        
    # Tokenize the text and perform sentence segmentation
    tokens = tokenizer.tokenize(data)
    sentences = tokenizer.convert_tokens_to_string(tokens).split('.')

    with open(output_file, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            file.write(sentence.strip() + '.\n\n')  # Add a double newline after each sentence

# Specify the input and output file paths
input_file = '51-100-dataset.txt'
output_file = 'output 51-100-dataset.txt'

# Run the function
segment_sentences(input_file, output_file)