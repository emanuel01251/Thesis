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
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re

model_paths = {
    "SSP with Augmentation": "./BERT-SSP-DA-POS/BERTPOS_test",
    "SOP with Augmentation": "./BERT-SOP-DA-POS/BERTPOS_test",
    "SSP without Augmentation": "./BERT-SSP-POS/BERTPOS_test",
    "SOP without Augmentation": "./BERT-SOP-POS/BERTPOS_test",
    "NSP baseline": "./BERT-NSP-POS/BERTPOS_test"
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
}

num_labels = len(pos_tag_mapping)
id2label = {idx: tag for tag, idx in pos_tag_mapping.items()}
label2id = {tag: idx for tag, idx in pos_tag_mapping.items()}

special_symbols = ['-', '&', "\"", "[", "]", "/", "$", "(", ")", "%", ":", "'", '.', '?', ',']

def symbol2token(symbol):

    # Check if the symbol is a comma
    if symbol == ',':
        return '[PMC] '

    elif symbol == '.':
        return '[PMP] '

    # Check if the symbol is in the list of special symbols
    elif symbol in special_symbols:
        return '[PMS] '

    # If the symbol is not a comma or in the special symbols list, keep it as it is
    return symbol

def preprocess_untagged_sentence(sentence):
    # Define regex pattern to capture all special symbols
    special_symbols_regex = '|'.join([re.escape(sym) for sym in 
                                      ['-', '&', "\"", "[", "]", "/", "$", "(", ")", "%", ":", "'", '.']])

    # Replace all special symbols with spaces around them
    sentence = re.sub(rf'({special_symbols_regex})', r' \1 ', sentence)

    # Remove extra whitespaces
    sentence = re.sub(r'\s+', ' ', sentence).strip()

    upper = sentence

    # Convert the sentence to lowercase
    sentence = sentence.lower()

    # Loop through the sentence and convert special symbols to tokens [PMS], [PMC], or [PMP]
    new_sentence = ""
    i = 0
    while i < len(sentence):
        if any(sentence[i:].startswith(symbol) for symbol in special_symbols):
            # Check for ellipsis and replace with '[PMS]'
            if i + 2 < len(sentence) and sentence[i:i + 3] == '...':
                new_sentence += '[PMS]'
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
                new_sentence += '[PMS] '
                i += 1
            elif i + 1 == len(sentence):
                new_sentence += '[PMS] '
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
    
    # Handle any remaining words if necessary
    for i in range(min_length, len(words_list)):
        formatted_output.append({
            "word": words_list[i],
            "entity": "UNK",  # Unknown tag for unmatched words
            "score": 0.0
        })
    
    return formatted_output

def predict_tags_highlight(test_sentence, selected_model):
    sentence, upper = preprocess_untagged_sentence(test_sentence)
    words_list = upper.split()
    predicted_tags = tag_sentence_highlight(test_sentence, selected_model)

    # Align words with their corresponding predicted tags
    pairs = list(zip(words_list, predicted_tags))
    return pairs


html_table = """
<table style="width: 100%; border: 1px solid gray; border-collapse: collapse;">
    <thead>
        <tr>
            <th style="border: 1px solid white; padding: 8px;">Part of Speech</th>
            <th style="border: 1px solid white; padding: 8px;">Tags</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Common Noun</td>
            <td style="border: 1px solid white; padding: 8px;">NNC</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Proper Noun</td>
            <td style="border: 1px solid white; padding: 8px;">NNP</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Proper Noun Abbreviation</td>
            <td style="border: 1px solid white; padding: 8px;">NNPA</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Common Noun Abbreviation</td>
            <td style="border: 1px solid white; padding: 8px;">NNCA</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">as Subject (Palagyo)/Personal Pronouns Singular</td>
            <td style="border: 1px solid white; padding: 8px;">PRS</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Personal Pronouns</td>
            <td style="border: 1px solid white; padding: 8px;">PRP</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Possessive Subject (Paari)</td>
            <td style="border: 1px solid white; padding: 8px;">PRSP</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Pointing to an Object Demonstrative/(Paturo)/Pamatlig</td>
            <td style="border: 1px solid white; padding: 8px;">PRO</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Question/Interrogative (Pananong)/Singular</td>
            <td style="border: 1px solid white; padding: 8px;">PRQ</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Location (Panlunan)</td>
            <td style="border: 1px solid white; padding: 8px;">PRL</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Comparison (Panulad)</td>
            <td style="border: 1px solid white; padding: 8px;">PRC</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Indefinite</td>
            <td style="border: 1px solid white; padding: 8px;">PRI</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Determiner</td>
            <td style="border: 1px solid white; padding: 8px;">DT</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Determiner (Pantukoy) for Common Noun Plural</td>
            <td style="border: 1px solid white; padding: 8px;">DTC</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Determiner (Pantukoy) for Proper Noun</td>
            <td style="border: 1px solid white; padding: 8px;">DTP</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Lexical Marker</td>
            <td style="border: 1px solid white; padding: 8px;">LM</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Conjunctions (Pang-ugnay)</td>
            <td style="border: 1px solid white; padding: 8px;">CJN</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Ligatures (Pang-angkop)</td>
            <td style="border: 1px solid white; padding: 8px;">CCP</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Preposition (Pang-ukol)</td>
            <td style="border: 1px solid white; padding: 8px;">CCU</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Verb (Pandiwa)</td>
            <td style="border: 1px solid white; padding: 8px;">VB</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Neutral/Infinitive</td>
            <td style="border: 1px solid white; padding: 8px;">VBW</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Auxiliary, Modal/Pseudo-verbs</td>
            <td style="border: 1px solid white; padding: 8px;">VBS</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Existential</td>
            <td style="border: 1px solid white; padding: 8px;">VBH</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Non-existential</td>
            <td style="border: 1px solid white; padding: 8px;">VBN</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Time Past (Perfective)</td>
            <td style="border: 1px solid white; padding: 8px;">VBTS</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Time Present (Imperfective)</td>
            <td style="border: 1px solid white; padding: 8px;">VBTR</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Time Future (Contemplative)</td>
            <td style="border: 1px solid white; padding: 8px;">VBTF</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Adjective (Pang-uri)</td>
            <td style="border: 1px solid white; padding: 8px;">JJ</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Describing (Panlarawan)</td>
            <td style="border: 1px solid white; padding: 8px;">JJD</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Used for Comparison (same level) (Pahambing Magkatulad)</td>
            <td style="border: 1px solid white; padding: 8px;">JJC</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Comparison Comparative (more) (Palamang)</td>
            <td style="border: 1px solid white; padding: 8px;">JJCC</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Comparison Superlative (most) (Pasukdol)</td>
            <td style="border: 1px solid white; padding: 8px;">JJCS</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Describing Number (Pamilang)</td>
            <td style="border: 1px solid white; padding: 8px;">JJN</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Adverb (Pang-Abay)</td>
            <td style="border: 1px solid white; padding: 8px;">RB</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Describing “How” (Pamaraang)</td>
            <td style="border: 1px solid white; padding: 8px;">RBD</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Number (Panggaano/Panukat)</td>
            <td style="border: 1px solid white; padding: 8px;">RBN</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Conditional (Kondisyunal)</td>
            <td style="border: 1px solid white; padding: 8px;">RBK</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Referential (Pangkaukulan)</td>
            <td style="border: 1px solid white; padding: 8px;">RBR</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Question (Pananong)</td>
            <td style="border: 1px solid white; padding: 8px;">RBQ</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Agree (Panang-ayon)</td>
            <td style="border: 1px solid white; padding: 8px;">RBT</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Disagree (Pananggi)</td>
            <td style="border: 1px solid white; padding: 8px;">RBF</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Frequency (Pamanahon)</td>
            <td style="border: 1px solid white; padding: 8px;">RBW</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Possibility (Pang-agam)</td>
            <td style="border: 1px solid white; padding: 8px;">RBM</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Place (Panlunan)</td>
            <td style="border: 1px solid white; padding: 8px;">RBL</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Enclitics (Paningit)</td>
            <td style="border: 1px solid white; padding: 8px;">RBI</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Cardinal Number (Bilang)</td>
            <td style="border: 1px solid white; padding: 8px;">CD</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Digit, Rank, Count</td>
            <td style="border: 1px solid white; padding: 8px;">CDB</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Foreign Words</td>
            <td style="border: 1px solid white; padding: 8px;">FW</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Punctuation (Pananda)</td>
            <td style="border: 1px solid white; padding: 8px;">PM</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Period</td>
            <td style="border: 1px solid white; padding: 8px;">PMP</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Exclamation Point</td>
            <td style="border: 1px solid white; padding: 8px;">PME</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Question Mark</td>
            <td style="border: 1px solid white; padding: 8px;">PMQ</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Comma</td>
            <td style="border: 1px solid white; padding: 8px;">PMC</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Semi-colon</td>
            <td style="border: 1px solid white; padding: 8px;">PMSC</td>
        </tr>
        <tr>
            <td style="border: 1px solid white; padding: 8px;">Symbols</td>
            <td style="border: 1px solid white; padding: 8px;">PMS</td>
        </tr>
    </tbody>
</table>
"""

def create_custom_html(tagged_data, highlighted_output):
    if not tagged_data or not highlighted_output:
        return ""
    
    # Extract colors from highlighted_output assuming it's a list of entities
    tag_colors = {}
    for entity in highlighted_output:  # Removed .get() since it's a list
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
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 6px;
        }
        .word {
            font-size: 1.1em;
            font-weight: 600;
            color: #2c3e50;
            margin-right: 15px;
            min-width: 120px;
            padding: 4px 8px;
            border-radius: 4px;
        }
        .tag {
            font-size: 1.1em;
            font-weight: 600;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
            margin-right: 15px;
        }
        .score {
            color: #7f8c8d;
            font-size: 1.2em;
            font-weight: 600;
            margin-left: 50px;
        }
        .score-bar {
            width: 500px;
            height: 15px;
            background: #ecf0f1;
            border-radius: 3px;
            margin-left: 50px;
            position: relative;
        }
        .score-fill {
            height: 100%;
            background: #2cdca7;
            border-radius: 3px;
            transition: width 0.3s ease;
        }
    </style>
    <div class="tag-card">
        <div class="word-section">
            <span class="score">Word</span>
            <span class="score">Tag Name</span>
            <span class="score">Confidence Score (%)</span>
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
        
        html += f"""
        <div class="tag-card">
            <div class="word-section" style="background: {lighter_color};">
                <span class="word">{word}</span>
                <span class="tag" style="background: {tag_color};">{tag}</span>
                <div class="score-bar">
                    <div class="score-fill" style="width: {score_percentage}%;"></div>
                </div>
                <span class="score">{score2percent:.2f}%</span>
            </div>
        </div>
        """
    
    return html

# Modified Gradio interface
with gr.Blocks(theme='ParityError/Interstellar') as tagger:
    gr.Markdown("<div style='margin-top: 20px; text-align: center; font-size: 2em; font-weight: bold;'>TAGALONGGO: A Part-of-Speech (POS) Tagger For Tagalog-Ilonggo Texts Using Bilingual BERT</div>")

    with gr.Row(): 
        with gr.Column(): 
            sentence_input = gr.Textbox(
                placeholder="Enter sentence here...",
                label="Input",
                elem_classes="input-box"
            )
            model_input_dropdown = gr.Dropdown(
                choices=list(model_paths.keys()),
                label="Select Model",
                value="SSP with Augmentation"
            )
            
            with gr.Row():
                clear_button = gr.Button("Clear", elem_classes="secondary-button")
                submit_button = gr.Button("Submit", elem_classes="primary-button")

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
    
    gr.HTML(html_table)

tagger.launch(favicon_path="favicon.png", share=True)