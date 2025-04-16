import re

def remove_doc_tags_but_keep_title(input_file, output_file):
    # Regular expression pattern to match the <doc> tag and capture the title attribute
    doc_tag_pattern = re.compile(r'<doc.*?title="(.*?)".*?>|</doc>')

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Search for <doc> tag and capture the title
            match = re.search(doc_tag_pattern, line)
            if match:
                title = match.group(1)  # Extract the title
                outfile.write(f"Title: {title}\n")  # Write the title in the output file
            else:
                # If no <doc> tag is found, just write the line (regular content)
                outfile.write(line)

    print(f"Processed text with <doc> tags removed but titles kept. Output saved to {output_file}")

# Example usage
input_file = './dataset/merged_file.txt'  # Replace with the path to your input file
output_file = './dataset/output merged_file.txt'  # Replace with the path to your output file

remove_doc_tags_but_keep_title(input_file, output_file)
