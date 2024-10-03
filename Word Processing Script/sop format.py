def format_to_sop_input_with_paragraphs(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    doc_id = 1
    output_lines = []
    
    title = ""
    article = []
    
    for line in lines:
        # Strip line breaks but preserve spaces
        stripped_line = line.rstrip()

        if stripped_line.startswith("Title:"):
            # Process the previous article (if any)
            if title and article:
                output_lines.append(f'<doc id="{doc_id}" url="url" title="{title}">')
                output_lines.extend(article)
                output_lines.append('</doc>\n')
                doc_id += 1
                article = []  # Reset for the next article

            # Get the new title, removing the "Title:" prefix
            title = stripped_line.replace("Title:", "").strip()

        else:
            # Add each line to the article, preserving empty lines for paragraph separation
            if stripped_line or article:
                article.append(stripped_line)

    # Process the final article
    if title and article:
        output_lines.append(f'<doc id="{doc_id}" url="url" title="{title}">')
        output_lines.extend(article)
        output_lines.append('</doc>\n')

    # Write the formatted content to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("\n".join(output_lines))

# Specify input and output file paths
input_file_path = './dataset/Unlabeled Corpus/Merged Corpus.txt'  # Replace with your actual file path
output_file_path = './dataset/Unlabeled Corpus/SOP Merged Corpus.txt'

# Run the function to convert the text
format_to_sop_input_with_paragraphs(input_file_path, output_file_path)

# Output path to the formatted file
output_file_path
