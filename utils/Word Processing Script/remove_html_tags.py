from bs4 import BeautifulSoup

def remove_html_tags_from_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    soup = BeautifulSoup(html_content, "html.parser")
    cleaned_text = soup.get_text()

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)

input_file = 'Hiligaynon_Literary_Text.txt'
output_file = 'output_file Hiligaynon_Literary_Text.txt'

remove_html_tags_from_file(input_file, output_file)

print(f"HTML tags removed. Cleaned text saved in '{output_file}'")