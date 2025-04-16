def add_title_prefix(titles_file, corpus_file, output_file):
    # Read the titles from the titles file
    with open(titles_file, 'r', encoding='utf-8') as f:
        titles = [line.strip() for line in f.readlines()]

    # Read the Hiligaynon Corpus
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus = f.read()

    # Go through each title and replace the first occurrence in the corpus
    for title in titles:
        # Make sure we don't prepend "Title: " multiple times by checking for it
        if title in corpus and f"Title: {title}" not in corpus:
            corpus = corpus.replace(title, f"Title: {title}", 1)

    # Write the modified corpus to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(corpus)

    print(f"Titles have been updated in '{output_file}'")

# Define file paths
titles_file = './dataset/list of titles.txt'  # Replace with your titles file path
corpus_file = './dataset/Unlabeled Corpus/Hiligaynon Corpus.txt'  # Replace with your corpus file path
output_file = './dataset/Unlabeled Corpus/Hiligaynon Corpus Updated.txt'  # The updated corpus with "Title: "

# Call the function to update the titles in the corpus
add_title_prefix(titles_file, corpus_file, output_file)
