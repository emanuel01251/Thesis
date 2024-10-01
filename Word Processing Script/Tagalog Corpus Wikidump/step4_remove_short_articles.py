def remove_short_articles(input_file, output_file, min_content_length=50):
    """
    Reads an input file, removes articles that have very short or blank content.
    The article is defined as starting with 'Title:' and ending with the next 'Title:'.
    Articles with content below `min_content_length` will be removed.
    
    Parameters:
    - input_file: Path to the input text file.
    - output_file: Path to save the filtered output text file.
    - min_content_length: Minimum content length required for an article to be kept.
    """
    filtered_articles = []
    buffer = ""
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            if line.startswith("Title:"):
                # When we reach a new title, process the previous article if exists
                if buffer:
                    # Split buffer into title and content
                    parts = buffer.split("\n", 1)
                    title = parts[0]
                    content = parts[1] if len(parts) > 1 else ""
                    
                    # Check if the article's content length meets the threshold
                    if len(content.strip()) >= min_content_length:
                        filtered_articles.append(buffer)
                
                # Start a new buffer for the current article
                buffer = line  # Add the current title to the buffer
            else:
                buffer += line  # Add content under the current title
    
        # Process the last article in the file
        if buffer:
            parts = buffer.split("\n", 1)
            title = parts[0]
            content = parts[1] if len(parts) > 1 else ""
            if len(content.strip()) >= min_content_length:
                filtered_articles.append(buffer)
    
    # Write filtered articles to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("\n".join(filtered_articles))
    
    print(f"Filtered articles saved to {output_file}")


# Example usage
input_file = './dataset/remove_poem_output_file merged_file.txt'  # Replace with your actual input file path
output_file = './dataset/remove_short_articles remove_poem_output_file merged_file.txt'  # Replace with your desired output file path

# This will remove articles with content less than 50 characters
remove_short_articles(input_file, output_file, min_content_length=2000)