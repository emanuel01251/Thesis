""" 
Program Title:
    TAGALONGGO POS Tagging Interface (app.py)

Created at:
    Tue Oct 1 2024
Last updated:
    Fri Nov 8 2024

Programmers:
    Emanuel Jamero
    John Nicolas Oandasan
    Vince Favorito
    Kyla Marie Alcantara

Purpose:
    To provide an interactive user interface for Part-of-Speech (POS) tagging of Tagalog-Ilonggo text. 
    Users can input their text and select from four different BERT-based POS tagging models—SSP, 
    SOP, SSP with Data Augmentation, and SOP with Data Augmentation. The interface allows users to 
    see how each selected model performs on the given text, displaying the POS tagging results. 
    This functionality is intended to facilitate comparative analysis and to make it easier for 
    users to experiment with different model configurations for POS tagging in a bilingual context.

 """

import gradio as gr
from how_to_use import create_how_to_use_page
from about_us import create_about_us_page
from pos_tags_reference import create_pos_tags_reference
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re

model_paths = {
    "SSP with Augmentation": "emanuel01251/BERT_SSP_DA_POS",
    "SOP with Augmentation": "emanuel01251/BERT_SOP_DA_POS",
    "SSP without Augmentation": "emanuel01251/BERT_SSP_POS",
    "SOP without Augmentation": "emanuel01251/BERT_SOP_POS",
    "Baseline": "emanuel01251/BERT-NSP-POS",
    "Baseline with Augmentation": "emanuel01251/BERT-NSP-DA-POS"
}

models = {name: AutoModelForTokenClassification.from_pretrained(path) for name, path in model_paths.items()}
tokenizers = {name: AutoTokenizer.from_pretrained(path) for name, path in model_paths.items()}

pos_tag_mapping = {
    '[PAD]': 0,
    'NN': 1,
    'PRNN': 2,
    'DET' : 3,
    'VB' : 4,
    'ADJ' : 5,
    'ADV' : 6,
    'NUM' : 7,
    'CONJ' : 8,
    'PUNCT' : 9,
    'FW' : 10,
    'LM' : 11,
}

num_labels = len(pos_tag_mapping)
id2label = {idx: tag for tag, idx in pos_tag_mapping.items()}
label2id = {tag: idx for tag, idx in pos_tag_mapping.items()}

special_symbols = ['-', '&', "\"", "[", "]", "/", "$", "(", ")", "%", ":", "'", '.', '?', ',']

def symbol2token(symbol):

    # Check if the symbol is a comma
    if symbol == ',':
        return '[PUNCT] '

    elif symbol == '.':
        return '[PUNCT] '

    # Check if the symbol is in the list of special symbols
    elif symbol in special_symbols:
        return '[PUNCT] '

    # If the symbol is not a comma or in the special symbols list, keep it as it is
    return symbol

def preprocess_untagged_sentence(sentence):
    # Define regex pattern to capture all special symbols
    special_symbols_regex = '|'.join([re.escape(sym) for sym in 
                                      ['-', '&', "\"", "[", "]", "/", "$", "(", ")", "%", ":", "'", '.', '?']])

    # Replace all special symbols with spaces around them
    sentence = re.sub(rf'({special_symbols_regex})', r' \1 ', sentence)

    # Remove extra whitespaces
    sentence = re.sub(r'\s+', ' ', sentence).strip()

    upper = sentence

    # Convert the sentence to lowercase
    sentence = sentence.lower()

    # Loop through the sentence and convert special symbols to tokens [PUNCT]
    new_sentence = ""
    i = 0
    while i < len(sentence):
        if any(sentence[i:].startswith(symbol) for symbol in special_symbols):
            # Check for ellipsis and replace with '[PUNCT]'
            if i + 2 < len(sentence) and sentence[i:i + 3] == '...':
                new_sentence += '[PUNCT]'
                i += 3
            # Check for single special symbols
            elif i + 1 == len(sentence):
                new_sentence += symbol2token(sentence[i])
                break
            elif sentence[i + 1] == ' ' and i == 0:
                new_sentence += symbol2token(sentence[i])
                i += 1
            elif sentence[i - 1] == ' ' and sentence[i + 1] == ' ':
                new_sentence += symbol2token(sentence[i])
                i += 1
            elif sentence[i - 1] != ' ':
                new_sentence += ''
            else:
                word_after_symbol = ""
                while i + 1 < len(sentence) and sentence[i + 1] != ' ' and not any(
                        sentence[i + 1:].startswith(symbol) for symbol in special_symbols):
                    word_after_symbol += sentence[i + 1]
                    i += 1
                new_sentence += word_after_symbol
        # Check for special symbols at the start of the sentence
        elif any(sentence[i:].startswith(symbol) for symbol in special_symbols):
            if i + 1 < len(sentence) and (sentence[i + 1] == ' ' and sentence[i - 1] != ' '):
                new_sentence += '[PUNCT] '
                i += 1
            elif i + 1 == len(sentence):
                new_sentence += '[PUNCT] '
                break
            else:
                word_after_symbol = ""
                while i + 1 < len(sentence) and sentence[i + 1] != ' ' and not any(
                        sentence[i + 1:].startswith(symbol) for symbol in special_symbols):
                    word_after_symbol += sentence[i + 1]
                    i += 1
                new_sentence += word_after_symbol
        else:
            new_sentence += sentence[i]
        i += 1

    print("Sentence after:", new_sentence.split())
    print("---")

    return new_sentence, upper

def clear_function(): 
    return "", "SSP with Augmentation"

import torch
import torch.nn.functional as F

def tag_sentence(input_sentence, selected_model):
    model = models[selected_model]  # Get the selected model
    tokenizer = tokenizers[selected_model]  # Get the corresponding tokenizer

    # Preprocess the input sentence
    sentence, upper = preprocess_untagged_sentence(input_sentence)
    
    # Tokenize the sentence
    encoded_sentence = tokenizer(sentence, padding="max_length", 
                                 truncation=True, max_length=128, return_tensors="pt")
    
    # Pass the encoded sentence to the model to get logits
    with torch.no_grad():
        model_output = model(**encoded_sentence)
    
    # Get the predicted tags
    logits = model_output.logits
    probabilities = F.softmax(logits, dim=-1)
    predicted_tags = torch.argmax(probabilities, dim=-1)
    
    # Convert predicted tags to their corresponding labels and scores
    scores = probabilities[0].max(dim=-1)[0].tolist()  # Get max scores for each token
    labels = [id2label[tag.item()] for tag in predicted_tags[0]]
    
    # Filter out the padding tokens and their scores
    filtered_pairs = [(labels[i], scores[i]) for i in range(len(labels)) if labels[i] != '[PAD]']
    
    return filtered_pairs  # Return the pairs of (label, score)

def tag_sentence_highlight(input_sentence, selected_model):
    model = models[selected_model]  # Get the selected model
    tokenizer = tokenizers[selected_model]  # Get the corresponding tokenizer

    # Preprocess the input sentence
    sentence, upper = preprocess_untagged_sentence(input_sentence)
    
    # Tokenize the sentence
    encoded_sentence = tokenizer(sentence, padding="max_length", 
                                 truncation=True, max_length=128, return_tensors="pt")
    
    # Pass the encoded sentence to the model to get logits
    with torch.no_grad():
        model_output = model(**encoded_sentence)
    
    # Get the predicted tags
    logits = model_output.logits
    probabilities = F.softmax(logits, dim=-1)
    predicted_tags = torch.argmax(probabilities, dim=-1)
    
    # Convert predicted tags to their corresponding labels
    labels = [id2label[tag.item()] for tag in predicted_tags[0] if id2label[tag.item()] != '[PAD]']
    
    return labels

def predict_tags(test_sentence, selected_model):
    sentence, upper = preprocess_untagged_sentence(test_sentence)
    words_list = upper.split()
    predicted_pairs = tag_sentence(test_sentence, selected_model)
    
    # Ensure we don't go out of bounds
    min_length = min(len(words_list), len(predicted_pairs))
    
    # Align words with their corresponding predicted tags and scores
    formatted_output = []
    for i in range(min_length):
        formatted_output.append({
            "word": words_list[i],
            "entity": predicted_pairs[i][0],
            "score": predicted_pairs[i][1]
        })
    
    """ # Handle any remaining words if necessary
    for i in range(min_length, len(words_list)):
        formatted_output.append({
            "word": words_list[i],
            "entity": "UNK",  # Unknown tag for unmatched words
            "score": 0.0
        }) """
    
    return formatted_output

def predict_tags_highlight(test_sentence, selected_model):
    sentence, upper = preprocess_untagged_sentence(test_sentence)
    words_list = upper.split()
    print("nice")
    print(words_list)
    predicted_tags = tag_sentence_highlight(test_sentence, selected_model)
    
    # Ensure we don't go out of bounds
    min_length = min(len(words_list), len(predicted_tags))
    
    # Align words with their corresponding predicted tags
    pairs = []
    for i in range(min_length):
        pairs.append((words_list[i], predicted_tags[i]))
    
    # Handle any remaining words if necessary
    for i in range(min_length, len(words_list)):
        pairs.append((words_list[i], "UNK"))  # Unknown tag for unmatched words
    
    return pairs


def create_custom_html(tagged_data, highlighted_output):
    if not tagged_data or not highlighted_output:
        return ""
    
    # Dictionary mapping tag abbreviations to full names
    pos_tags = {
        'VB': 'Verb',
        'NN': 'Noun',
        'PRNN': 'Pronoun',
        'DET': 'Determiner',
        'ADJ': 'Adjective',
        'ADV': 'Adverb',
        'NUM': 'Numerical',
        'CONJ': 'Conjunction',
        'PUNCT': 'Punctuation',
        'FW': 'Foreign Word'
    }
    
    # Extract colors from highlighted_output assuming it's a list of entities
    tag_colors = {}
    for entity in highlighted_output:
        if isinstance(entity, dict) and 'entity' in entity and 'color' in entity:
            tag_colors[entity['entity']] = entity['color']
    
    html = """
    <style>
        .tag-card {
            background: black;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 10px 0;
            transition: all 0.3s ease;
        }
        .tag-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .word-section {
            display: grid;
            grid-template-columns: 150px 120px minmax(300px, 1fr) 100px;
            align-items: center;
            gap: 20px;
            padding: 15px 30px;
            border-radius: 6px;
        }
        .word {
            font-size: 1.1em;
            font-weight: 600;
            color: #ffffff;
        }
        .tag {
            font-size: 0.9em;
            font-weight: 600;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            text-align: center;
            width: fit-content;
            cursor: help;
            position: relative;
        }
        .tag:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            bottom: 100%;
            padding: 5px 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border-radius: 4px;
            font-size: 0.9em;
            white-space: nowrap;
            z-index: 1000;
            margin-bottom: 5px;
        }
        .score {
            color: #ffffff;
            font-size: 1.1em;
            font-weight: 600;
        }
        .score-bar {
            width: 100%;
            height: 15px;
            background: #2a2a2a;
            border-radius: 3px;
            position: relative;
            overflow: hidden;
        }
        .score-fill {
            height: 100%;
            background: #2cdca7;
            border-radius: 3px;
            transition: width 0.3s ease;
        }
        .header {
            color: #ffffff;
            font-weight: 600;
            font-size: 1.1em;
        }
    </style>
    <div class="tag-card">
        <div class="word-section">
            <span class="header">Word</span>
            <span class="header">Tag Name</span>
            <span class="header">Confidence Score (%)</span>
            <span class="header"></span>
        </div>
    </div>
    """
    
    for item in tagged_data:
        word = item.get('word', '')
        tag = item.get('entity', '')
        score = item.get('score', 0)
        score2percent = score * 100
        score_percentage = min(score * 100, 100)
        
        tag_color = tag_colors.get(tag, '#702cdc')
        lighter_color = f"{tag_color}40"  # 25% opacity version for background
        
        # Get the full name for the tooltip, default to the tag if not found
        tooltip = pos_tags.get(tag, tag)
        
        html += f"""
        <div class="tag-card">
            <div class="word-section" style="background: {lighter_color};">
                <span class="word">{word}</span>
                <span class="tag" style="background: {tag_color};" data-tooltip="{tooltip}">{tag}</span>
                <div class="score-bar">
                    <div class="score-fill" style="width: {score_percentage}%;"></div>
                </div>
                <span class="score">{score2percent:.2f}%</span>
            </div>
        </div>
        """
    
    return html

custom_css = """
<style>
/* Input and Select Model Styling */
.input-container {
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.95), rgba(45, 45, 45, 0.95));
    border-radius: 12px;
    padding: 10px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

/* Input Box Styling */
.input-box textarea {
   background: linear-gradient(165deg, rgba(31, 41, 55, 0.95), rgba(17, 24, 39, 0.98)) !important;
   border: 1px solid rgba(108, 44, 220, 0.2) !important;
   border-radius: 14px !important;
   padding: 18px !important;
   box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1),
               0 8px 16px rgba(0, 0, 0, 0.1) !important;
   backdrop-filter: blur(4px) !important;
}

.input-box textarea:focus {
   border-color: #2cdca7 !important;
   box-shadow: 0 0 0 3px rgba(44, 220, 167, 0.15),
               inset 0 2px 4px rgba(0, 0, 0, 0.1) !important;
   transform: translateY(-1px) !important;
}

.input-box label {
    color: #2cdca7 !important;
    font-size: 24px !important;
    font-weight: 600 !important;
}

/* Select Model Dropdown Styling */
/* Updated Select Model styles */
.model-select {
    margin: 10px 0;
}


/* Container reset */
.model-select > .wrap {
    background: transparent !important;
    padding: 0 !important;
}

/* Label styling */
.model-select label {
    color: #2cdca7 !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    margin-bottom: 8px !important;
}

.model-select select {
   background: linear-gradient(165deg, rgba(31, 41, 55, 0.95), rgba(17, 24, 39, 0.98)) !important;
   border: 1px solid rgba(108, 44, 220, 0.2) !important;
   border-radius: 12px !important;
   box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
}

/* Dropdown button styling */
.model-select > select,
.model-select > .wrap > div > select {
    background: linear-gradient(165deg, rgba(31, 41, 55, 0.95), rgba(17, 24, 39, 0.98)) !important;
    border: 1px solid rgba(108, 44, 220, 0.2) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    color: white !important;
    font-size: 16px !important;
    padding: 12px !important;
}

/* Hover state */
.model-select select:hover,
.model-select > .wrap > .wrap > select:hover {
    border-color: #2cdca7 !important;
    background-color: rgba(31, 41, 55, 0.98) !important;
}

/* Focus state */
.model-select select:focus,
.model-select > .wrap > .wrap > select:focus {
    border-color: #702cdc !important;
    box-shadow: 0 0 0 2px rgba(108, 44, 220, 0.2) !important;
    outline: none !important;
}

/* Dropdown options */
.model-select select option,
.model-select > .wrap > .wrap > select option {
    background-color: #1a1a1a !important;
    color: white !important;
    padding: 12px !important;
}

/* Remove default dropdown arrow in IE */
.model-select select::-ms-expand {
    display: none !important;
}

/* Button Styling */
.primary-button, .secondary-button {
    padding: 10px 20px !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

.primary-button {
    background: linear-gradient(270deg, #702cdc, #2cdca7, #702cdc) !important;
    background-size: 200% 100% !important;
    animation: gradient 3s ease infinite !important;
    border: none !important;
    color: white !important;
}
.secondary-button {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 2px solid rgba(108, 44, 220, 0.3) !important;
    color: #944FBF !important;
}

.primary-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(44, 220, 167, 0.2) !important;
}

.secondary-button:hover {
    background: rgba(255, 255, 255, 0.15) !important;
    transform: translateY(-2px) !important;
}

@keyframes gradient {
    0% { background-position: 0% 50% }
    50% { background-position: 100% 50% }
    100% { background-position: 0% 50% }
}


/* Examples Section Styling */
.examples-section {
    margin-top: 10px !important;
    padding: 15px !important;
    border-radius: 8px !important;
}

.examples-section label {
    color: #2cdca7 !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* Tab Navigation Styling */
.tabs > .tab-nav {
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.95), rgba(45, 45, 45, 0.95)) !important;
    border-radius: 12px !important;
    padding: 8px !important;
}

.tabs > .tab-nav > button {
    font-family: 'Poppins', sans-serif !important;  /* Change font family */
    font-size: 16px !important;                     /* Change font size */
    font-weight: 600 !important;                    /* Change font weight */
    color: #ffffff !important;                      /* Change font color */
    text-transform: uppercase !important;           /* Make text uppercase */
    letter-spacing: 1px !important;                 /* Add letter spacing */
    padding: 12px 24px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

/* Selected tab styling */
.tabs > .tab-nav > button.selected {
    background: linear-gradient(90deg, #702cdc, #2cdca7) !important;
    color: white !important;
}

/* Hover effect for tabs */
.tabs > .tab-nav > button:hover:not(.selected) {
    background: rgba(255, 255, 255, 0.1) !important;
    transform: translateY(-2px) !important;
}
</style>
"""

# Modified process_and_display_cards function
def process_and_display_cards(sentence, model):
    # Get the tagged data
    tagged_data = predict_tags(sentence, model)
        
    # Get the highlighted output and extract just the entities
    highlighted_result = predict_tags_highlight(sentence, model)
    entities = highlighted_result.get('entities', []) if isinstance(highlighted_result, dict) else highlighted_result
        
    # Convert to card view HTML using the same colors
    html_output = create_custom_html(tagged_data, entities)
    return html_output

def create_main_interface():
    with gr.Blocks(theme='ParityError/Interstellar') as tagger:
        gr.HTML(custom_css)  # Add the custom CSS
        gr.Image(value="img/Tagalongo logo.png", show_label=False, container=False, height=150)
        
        with gr.Row(): 
            with gr.Column(): 
                with gr.Column():
                    sentence_input = gr.Textbox(
                        placeholder="Enter sentence here...",
                        label="INPUT",
                        elem_classes="input-box"
                    )
                    model_input_dropdown = gr.Dropdown(
                        choices=list(model_paths.keys()),
                        label="Select Model",
                        value="SSP with Augmentation",
                        elem_classes="model-select",
                        container=False  # Changed to False to prevent extra container
                    )
                    
                    with gr.Row():
                        clear_button = gr.Button("Clear", elem_classes="secondary-button")
                        submit_button = gr.Button("Submit", elem_classes="primary-button")

                    with gr.Column(elem_classes="examples-section"):
                        example_text = gr.Examples(
                            examples=[
                                ["Luyag ko mag-bakasyon sa iloilo kay damo sang magagandang lugar."],
                                ["Nagbakal ako ng bakal."]
                            ],
                            inputs=[sentence_input, model_input_dropdown]
                        )

            with gr.Column(min_width=900):  
                tagged_output = gr.HighlightedText(label="Tagged Texts:")
                tagged_output_cards = gr.HTML(label="Tagged Texts With Scores")

        # Modified process_and_display_cards function
        def process_and_display_cards(sentence, model):
            # Get the tagged data
            tagged_data = predict_tags(sentence, model)

            # Get the highlighted output and extract just the entities
            highlighted_result = predict_tags_highlight(sentence, model)
            entities = highlighted_result.get('entities', []) if isinstance(highlighted_result, dict) else highlighted_result

            # Convert to card view HTML using the same colors
            html_output = create_custom_html(tagged_data, entities)
            return html_output

        clear_button.click(
            fn=clear_function,
            outputs=[sentence_input, model_input_dropdown]
        )

        submit_button.click(
            fn=predict_tags_highlight,
            inputs=[sentence_input, model_input_dropdown],
            outputs=tagged_output
        )

        submit_button.click(
            fn=process_and_display_cards,
            inputs=[sentence_input, model_input_dropdown],
            outputs=tagged_output_cards
        )
        
    return tagger

demo = gr.TabbedInterface(
    interface_list=[
        create_main_interface(),
        create_pos_tags_reference(),
        create_how_to_use_page(),
        create_about_us_page()
    ],
    tab_names=["🏠 Home", "📝 List of POS", "❔ How to Use", "👥 About Us"],
    css=custom_css
)

if __name__ == "__main__":
    demo.launch(favicon_path="favicon.png", share=True)