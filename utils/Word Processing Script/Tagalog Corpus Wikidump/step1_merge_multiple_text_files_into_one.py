import os

def merge_text_files(input_folder, output_file):
    # Open the output file in write mode
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Traverse through all files in the input folder
        for filename in os.listdir(input_folder):
            if filename.endswith('.txt'):  # Only process text files
                file_path = os.path.join(input_folder, filename)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())  # Write file content to output
                    outfile.write('\n\n')  # Add a double newline between files for separation
    print(f"All text files merged into {output_file}")

# Example usage
input_folder = './wikiextractor/extracted_tlwiki/AA'  # Replace with your folder path
output_file = './dataset/merged_file.txt'  # Replace with the desired output file path

merge_text_files(input_folder, output_file)
