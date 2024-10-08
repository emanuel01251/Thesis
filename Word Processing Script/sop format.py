def format_to_sop_input_with_paragraphs(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    doc_id = 1
    output_lines = []
    
    title = ""
    article = []
    
    for line in lines:
        stripped_line = line.rstrip()

        if stripped_line.startswith("Title:"):
            if title and article:
                output_lines.append(f'<doc id="{doc_id}" url="url" title="{title}">')
                output_lines.extend(article)
                output_lines.append('</doc>\n')
                doc_id += 1
                article = []

            title = stripped_line.replace("Title:", "").strip()

        else:
            if stripped_line or article:
                article.append(stripped_line)

    if title and article:
        output_lines.append(f'<doc id="{doc_id}" url="url" title="{title}">')
        output_lines.extend(article)
        output_lines.append('</doc>\n')

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("\n".join(output_lines))

input_file_path = './Data Augmentation/EDA/Unlabeled Corpus/New Hiligaynon Corpus augmented_output.txt'
output_file_path = './Data Augmentation/EDA/Unlabeled Corpus/SOP New Hiligaynon Corpus augmented_output.txt'

format_to_sop_input_with_paragraphs(input_file_path, output_file_path)

