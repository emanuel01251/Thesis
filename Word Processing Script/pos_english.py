import re
from nltk.corpus import words
import nltk
from typing import List, Tuple, Dict
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

class POSTagAnalyzer:
    def __init__(self):
        # Create a set of English words for faster lookup
        self.english_words = set(words.words())
        
        # Define common Filipino/Tagalog/Ilonggo words that might be mistaken as English
        self.excluded_words = {
            'si', 'sang', 'may', 'ka', 'ko', 'ako', 'pa', 
            'labis', 'para', 'ara', 'ang', 'ng', 'mga', 'sa',
            'at', 'ay', 'na', 'in', 'an', 'man', 'are', 'as', 'ni', 'con', 'kung', 'gusto', 'pang', 'tanan', 'kon', 'kay', 'mas', 'away',
            'iba', 'sina', 'aga', 'panay', 'sur', 'antes' 'Base', 'base', 'gid', 'dali', 'sing', 'tig', 'antes', 'lain'
        }
        
    def parse_tagged_text(self, line: str) -> List[Tuple[str, str]]:
        """
        Parse the tagged text format and return list of (word, tag) tuples
        Example input: '<DET Ang> <NN Portugal>'
        """
        # Find all patterns like '<TAG word>'
        pattern = r'<(\w+)\s+([^>]+)>'
        return re.findall(pattern, line)
    
    def is_english_word(self, word: str) -> bool:
        """
        Check if a word is in the English dictionary and not in excluded words list
        """
        word_lower = word.lower()
        return (word_lower in self.english_words and 
                word_lower not in self.excluded_words and 
                len(word) > 1)  # Exclude single letters
    
    def analyze_file(self, file_path: str) -> Dict[str, List[Tuple[str, str]]]:
        """
        Analyze the file and return dictionary of findings
        Returns: {
            'nn_matches': [(word, line_number, original_line)],
            'fw_matches': [(word, line_number, original_line)],
            'possible_mismatches': [(word, current_tag, line_number, original_line)]
        }
        """
        results = {
            'nn_matches': [],
            'fw_matches': [],
            'possible_mismatches': []
        }
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                    
                tagged_words = self.parse_tagged_text(line)
                
                for tag, word in tagged_words:
                    # Skip punctuation and numbers
                    if (word.isdigit() or 
                        all(char in '.,!?;:' for char in word) or 
                        word in self.excluded_words):
                        continue
                        
                    if self.is_english_word(word):
                        if tag == 'NN':
                            results['nn_matches'].append((word, line_num, line))
                        elif tag == 'FW':
                            results['fw_matches'].append((word, line_num, line))
                        else:
                            # If it's English but tagged as something else
                            results['possible_mismatches'].append((word, tag, line_num, line))

        return results

def write_results_to_file(results: Dict[str, List[Tuple[str, str]]], input_file_path: str):
    """Write the analysis results to a text file"""
    # Generate timestamp for the output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"analysis_results_{timestamp}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== POS Tag Analysis Results ===\n")
        f.write(f"Input file analyzed: {input_file_path}\n")
        f.write(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write NN matches
        f.write("\nEnglish words tagged as nouns (NN):\n")
        if results['nn_matches']:
            for word, line_num, original_line in results['nn_matches']:
                f.write(f"- {word} (line {line_num})\n")
                #f.write(f"  Context: {original_line}\n")
        else:
            f.write("None found\n")
        
        # Write FW matches
        f.write("\nEnglish words tagged as foreign words (FW):\n")
        if results['fw_matches']:
            for word, line_num, original_line in results['fw_matches']:
                f.write(f"- {word} (line {line_num})\n")
                #f.write(f"  Context: {original_line}\n")
        else:
            f.write("None found\n")
        
        # Write possible mismatches
        f.write("\nPossible mismatches (English words with other tags):\n")
        if results['possible_mismatches']:
            for word, tag, line_num, original_line in results['possible_mismatches']:
                f.write(f"- {word} (tagged as {tag}, line {line_num})\n")
                #f.write(f"  Context: {original_line}\n")
        else:
            f.write("None found\n")
            
        # Write summary statistics
        f.write("\nSummary Statistics:\n")
        f.write(f"Total English nouns (NN): {len(results['nn_matches'])}\n")
        f.write(f"Total English foreign words (FW): {len(results['fw_matches'])}\n")
        f.write(f"Total possible mismatches: {len(results['possible_mismatches'])}\n")
    
    return output_file

def main():
    # Initialize the analyzer
    analyzer = POSTagAnalyzer()
    
    # Get input file path from user
    file_path = "./Dataset/Labeled Corpus/hil-train-set-less-tagset.txt"
    
    try:
        # Analyze the file
        print("Analyzing file...")
        results = analyzer.analyze_file(file_path)
        
        # Write results to file
        output_file = write_results_to_file(results, file_path)
        print(f"\nAnalysis complete! Results have been saved to: {output_file}")
        
        # Print a brief summary to console
        print("\nQuick Summary:")
        print(f"Found {len(results['nn_matches'])} English nouns (NN)")
        print(f"Found {len(results['fw_matches'])} English words tagged as foreign (FW)")
        print(f"Found {len(results['possible_mismatches'])} possible mismatches")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()