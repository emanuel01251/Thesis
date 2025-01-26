import re
from collections import defaultdict
from typing import Dict, List, Tuple
from datetime import datetime
import os

def read_file_content(filename: str, debug: bool = False) -> str:
    """
    Reads and returns file content with debug option.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            if debug:
                print("\nFirst 500 characters of file content:")
                print(content[:500])
                print("\n" + "="*50 + "\n")
            return content
    except UnicodeDecodeError:
        try:
            with open(filename, 'r', encoding='latin-1') as file:
                content = file.read()
                if debug:
                    print("\nFirst 500 characters of file content (latin-1 encoding):")
                    print(content[:500])
                    print("\n" + "="*50 + "\n")
                return content
        except Exception as e:
            print(f"Error with alternative encoding: {str(e)}")
            return ""
    except FileNotFoundError:
        print(f"File not found: {filename}")
        print("Current working directory:", os.getcwd())
        return ""
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return ""

def find_multiword_tags(filename: str, debug: bool = False) -> Tuple[Dict[str, List[str]], int]:
    """
    Analyzes a text file to find POS tags containing multiple words.
    """
    multiword_tags = defaultdict(list)
    total_count = 0
    
    # Read file content
    text = read_file_content(filename, debug)
    if not text:
        return {}, 0

    # Pattern to match tags with their content: <TAG word(s)>
    pattern = r'<(\w+)\s+([^>]+)>'
    
    matches = re.findall(pattern, text)
    if debug:
        print(f"\nFound {len(matches)} total tagged expressions")
        print("First 3 matches:", matches[:3])
    
    for tag, content in matches:
        # Clean up the tag and content
        tag = tag.strip()
        content = content.strip()
        
        # Check if content contains multiple words
        words = content.split()
        if len(words) >= 2:
            multiword_tags[tag].append(content)
            total_count += 1
            if debug:
                print(f"Found multi-word: <{tag}> {content}")
                    
    return multiword_tags, total_count

def export_multiword_results(multiword_tags: Dict[str, List[str]], total_count: int, input_filename: str) -> str:
    """
    Exports the multi-word analysis results to a text file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_base = os.path.splitext(os.path.basename(input_filename))[0]
    output_filename = f"{input_base}_multiword_analysis_{timestamp}.txt"
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("Multi-word POS Tag Analysis Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total multi-word expressions found: {total_count}\n")
            f.write(f"Number of unique tags with multi-word expressions: {len(multiword_tags)}\n\n")
            
            f.write("Detailed Analysis by Tag:\n")
            f.write("-" * 20 + "\n\n")
            
            for tag in sorted(multiword_tags.keys()):
                expressions = multiword_tags[tag]
                f.write(f"{tag} ({len(expressions)} expressions):\n")
                for expr in sorted(expressions):
                    f.write(f"  • {expr}\n")
                f.write("\n")
                    
        return output_filename
    except Exception as e:
        print(f"Error exporting results: {str(e)}")
        return ""

def display_multiword_results(multiword_tags: Dict[str, List[str]], total_count: int):
    """
    Displays the multi-word analysis results in console.
    """
    if not multiword_tags:
        print("\nNo multi-word expressions were found in the file.")
        return
        
    print("\nMulti-word POS Tag Analysis Results:")
    print("=" * 50)
    
    print(f"\nTotal multi-word expressions found: {total_count}")
    print(f"Number of unique tags with multi-word expressions: {len(multiword_tags)}\n")
    
    print("Detailed Analysis by Tag:")
    print("-" * 20)
    
    for tag in sorted(multiword_tags.keys()):
        expressions = multiword_tags[tag]
        print(f"\n{tag} ({len(expressions)} expressions):")
        for expr in sorted(expressions):
            print(f"  • {expr}")

def main():
    # Use the specific dataset path
    filename = "./Dataset/Labeled Corpus/tl-hil-eval-set-less-tagset.txt"
    
    # Run with debug mode on
    print(f"Attempting to analyze file: {filename}")
    multiword_tags, total_count = find_multiword_tags(filename, debug=True)
    
    # Display results in console
    display_multiword_results(multiword_tags, total_count)
    
    # Export results to file if any found
    if multiword_tags:
        output_file = export_multiword_results(multiword_tags, total_count, filename)
        if output_file:
            print(f"\nResults have been exported to: {output_file}")
        else:
            print("\nError: Could not export results to file.")

if __name__ == "__main__":
    main()