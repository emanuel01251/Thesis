import torch
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np

def predict_masked_words(model_path, vocab_file, sentence, top_k=5):
    """
    Predict masked words in a sentence using a custom BERT model.
    
    Args:
        model_path (str): Path to the custom BERT model
        sentence (str): Input sentence with [MASK] token
        top_k (int): Number of top predictions to return
    
    Returns:
        list: Top k predictions with their probabilities
    """
    # Load tokenizer and model
    tokenizer = BertTokenizer(vocab_file=vocab_file)
    model = BertForMaskedLM.from_pretrained(model_path)
    model.eval()

    # Tokenize input
    inputs = tokenizer.encode_plus(
        sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # Get mask token index
    mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits

    # Get probabilities for masked position
    mask_predictions = predictions[0, mask_index]
    probs = torch.nn.functional.softmax(mask_predictions, dim=-1)

    # Get top k predictions
    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
    
    # Convert to words and probabilities
    results = []
    for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
        predicted_token = tokenizer.convert_ids_to_tokens([idx])[0]
        results.append({
            'token': predicted_token,
            'probability': float(prob)
        })
    
    return results

# Example usage
if __name__ == "__main__":
    # Replace with your model path
    ssp_model_path = "./BERT-SSP/output_model_merged/checkpoint-331500/"
    vocab_file = "./Tokenizer/corpus-merged/uncased-vocab.txt"
    
    # Test sentence with mask
    test_sentence = "Ang data nga nadula sa hard draib nga sang Pearson Driving [MASK] Ltd , isa ka pribado nga kontraktor sa ahensya sang programa sang pagmamaneho sang UK ."
    
    print(f"Testing sentence: {test_sentence}")
    predictions = predict_masked_words(ssp_model_path, vocab_file, test_sentence)
    
    print("\nTop 5 predictions for masked word:")
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. {pred['token']}: {pred['probability']:.4f}")