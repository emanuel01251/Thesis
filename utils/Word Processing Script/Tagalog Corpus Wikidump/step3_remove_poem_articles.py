def remove_poem_articles(input_file, output_file, min_line_length=30, min_lines_for_poem=3):
    """
    Reads an input file, removes articles that are detected as poems based on
    having multiple short lines of text.
    
    Parameters:
    - input_file: Path to the input text file.
    - output_file: Path to save the filtered output text file.
    - min_line_length: The maximum average line length considered for identifying a poem.
    - min_lines_for_poem: Minimum number of consecutive short lines required to consider the article a poem.
    """
    filtered_articles = []
    buffer = ""
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            if line.startswith("Title:"):
                # Process the previous article
                if buffer:
                    title, content = buffer.split("\n", 1) if "\n" in buffer else (buffer, "")
                    
                    # Check if it's a poem by counting short lines
                    lines = content.split("\n")
                    short_line_count = sum(1 for l in lines if len(l.strip()) > 0 and len(l) <= min_line_length)
                    
                    # Remove the article if it has enough short lines (likely a poem)
                    if short_line_count < min_lines_for_poem:
                        filtered_articles.append(buffer)
                
                # Start a new buffer for the current article
                buffer = line
            else:
                buffer += line

        # Process the last article
        if buffer:
            title, content = buffer.split("\n", 1) if "\n" in buffer else (buffer, "")
            lines = content.split("\n")
            short_line_count = sum(1 for l in lines if len(l.strip()) > 0 and len(l) <= min_line_length)
            if short_line_count < min_lines_for_poem:
                filtered_articles.append(buffer)

    # Write filtered articles to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("\n".join(filtered_articles))
    
    print(f"Filtered articles saved to {output_file}")


# Example usage
input_file = './dataset/output merged_file.txt'  # Replace with your actual input file path
output_file = './dataset/remove_poem_output_file merged_file.txt'  # Replace with your desired output file path

# This will remove articles with more than 3 short lines (less than or equal to 30 characters each)
remove_poem_articles(input_file, output_file, min_line_length=30, min_lines_for_poem=3)
