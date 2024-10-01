from bs4 import BeautifulSoup

def remove_html_tags_from_file(input_file, output_file):
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    # Use BeautifulSoup to remove HTML tags
    soup = BeautifulSoup(html_content, "html.parser")
    cleaned_text = soup.get_text()

    # Write the cleaned text to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)

# Specify input and output file paths
input_file = 'Hiligaynon_Literary_Text.txt'  # Replace with your input text file
output_file = 'output_file Hiligaynon_Literary_Text.txt'  # Replace with your desired output file

# Call the function to process the files
remove_html_tags_from_file(input_file, output_file)

print(f"HTML tags removed. Cleaned text saved in '{output_file}'")