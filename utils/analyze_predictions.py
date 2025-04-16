def analyze_predictions(file_path):
    total_labels = 0
    correct_predictions = 0
    current_sentence = None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check if this is a sentence header
            if line.startswith("Sentence"):
                current_sentence = line
                continue
                
            # Skip separator lines
            if line.startswith("-"):
                continue
                
            # Skip header line
            if line.startswith("Token"):
                continue
                
            # Skip full sentence line
            if line.startswith("Full sentence:"):
                continue
                
            # Process prediction lines
            if len(line.split()) >= 4:  # Make sure we have enough columns
                token, true_label, pred_label, match = line.split()[:4]
                total_labels += 1
                if match == "✓":
                    correct_predictions += 1
    
    except FileNotFoundError:
        print(f"Error: Could not find file {file_path}")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None
    
    accuracy = (correct_predictions / total_labels) * 100 if total_labels > 0 else 0
    
    return {
        'total_labels': total_labels,
        'correct_predictions': correct_predictions,
        'accuracy': accuracy
    }

if __name__ == "__main__":
    results = analyze_predictions("sample.txt")
    
    if results:
        print("\nPrediction Analysis Results")
        print("=" * 30)
        print(f"Total Labels: {results['total_labels']}")
        print(f"Correct Predictions: {results['correct_predictions']}")
        print(f"Accuracy: {results['accuracy']:.2f}%")