def count_words_in_text(text):
    """
    Helper function to count the number of words in a given text.
    """
    words = text.split()
    return len(words)

def trim_text_to_word_limit(input_file, output_file, word_limit=1100000):
    """
    Reads an input file, counts words up to the specified word_limit,
    and removes any text beyond the limit while maintaining the 'Title:' pattern.
    """
    current_word_count = 0
    keep_text = []
    should_stop = False

    with open(input_file, 'r', encoding='utf-8') as infile:
        buffer = ""
        for line in infile:
            # Check if a new title is encountered (starts with 'Title:')
            if line.startswith("Title:"):
                # If there is any content in the buffer from the previous title, process it
                if buffer:
                    buffer_word_count = count_words_in_text(buffer)
                    if current_word_count + buffer_word_count > word_limit:
                        # If adding this content exceeds the word limit, truncate it
                        remaining_words = word_limit - current_word_count
                        truncated_content = ' '.join(buffer.split()[:remaining_words])
                        keep_text.append(truncated_content)
                        should_stop = True
                        break
                    else:
                        keep_text.append(buffer)
                        current_word_count += buffer_word_count
                
                # Reset buffer to store the new title and its content
                buffer = line  # Store the title

            else:
                # Accumulate content under the current title
                buffer += line

        # Check the last buffered section
        if not should_stop and buffer:
            buffer_word_count = count_words_in_text(buffer)
            if current_word_count + buffer_word_count <= word_limit:
                keep_text.append(buffer)
            else:
                remaining_words = word_limit - current_word_count
                truncated_content = ' '.join(buffer.split()[:remaining_words])
                keep_text.append(truncated_content)

    # Write the result to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(keep_text))

    print(f"Text trimmed to {word_limit} words and saved to {output_file}")

# Example usage
input_file = './dataset/remove_short_articles remove_poem_output_file merged_file.txt'  # Replace with the path to your input file
output_file = './dataset/trimmed remove_short_articles remove_poem_output_file merged_file.txt'  # Replace with the desired output file path

trim_text_to_word_limit(input_file, output_file)