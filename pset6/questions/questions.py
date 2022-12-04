"""
AI that answers questions based on given corpus of data.
Most relevant file(s) are first retrieved, followed by the relevant sentence(s).
Uses tf-idf values to rank files and sentences.
Usage: python questions.py corpus --> input query at command line.
"""

import nltk
import sys
import os
import string
import math

# Constants that determine how many files and sentences are used to answer query
FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {filename: tokenize(files[filename]) for filename in files}
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentence_words = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            # Tokenize passage to individual sentences
            for sentence in nltk.sent_tokenize(passage):
                # Tokenize sentence to individual words (tokens)
                tokens = tokenize(sentence)
                if tokens:
                    sentence_words[sentence] = tokens

    # Compute IDF values across sentences
    sentence_idfs = compute_idfs(sentence_words)

    # Determine top sentence matches
    matches = top_sentences(query, sentence_words, sentence_idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = dict()
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            # open() uses the default platform encoding - CP1252 for windows, but wiki pages are encoded UTF-8
            with open(filepath, encoding="utf-8") as f:
                files[filename] = f.read()
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.

    Removes punctuation and stopwords in string.punctuation and nltk.corpus.stopwords.words("english").
    """
    # Tokenize document
    words = nltk.word_tokenize(document.lower())

    # Punctuation and stopwords to be removed
    punctuation = string.punctuation
    stopwords = nltk.corpus.stopwords.words("english")

    # Remove punctuation and stopwords
    words_to_remove = set()
    for word in words:
        if word in punctuation or word in stopwords:
            words_to_remove.add(word)

    for word in words_to_remove:
        words.remove(word)

    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.

    IDF(word) = log(TotalDocuments / NumDocumentsContaining(word))
    """
    idfs = dict()

    # Iterate through each word in each document, adding word and its idf value to `idfs`
    for words in documents.values():
        for word in words:
            if word not in idfs:
                idfs[word] = math.log(
                    len(documents) / num_docs_with_word(documents, word)
                )

    return idfs


def num_docs_with_word(documents, word):
    """
    Helper function for `compute_idfs`.
    Counts the number of documents containing `word`.
    `documents` is a dict mapping names to a list of words.
    """
    count = 0
    for words in documents.values():
        if word in words:
            count += 1
    return count


def top_files(query, file_words, word_idfs, n):
    """
    Given a `query` (a set of words), `file_words` (a dictionary mapping names of
    files to a list of their words), and `word_idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # List of filenames
    filenames = list(file_words.keys())

    # Store tf-idf values for each word in each file
    tfidfs = {file: {} for file in filenames}

    # Compute td-idf values
    for file, words in file_words.items():
        for word in words:
            if word not in tfidfs[file]:
                term_frequency = words.count(word)
                tfidfs[file][word] = term_frequency * word_idfs[word]

    # Compute and store sum of tf-idf values for all words in `query` for each file
    query_tfidfs = dict()
    for file in filenames:
        query_sum = 0
        for word in query:
            # Only consider query words which are in the given file
            try:
                query_sum += tfidfs[file][word]
            except KeyError:
                pass

        # Update `query_tfidfs`
        query_tfidfs[file] = query_sum

    # Sort files from highest tfidf rank to lowest
    filenames.sort(key=lambda x: query_tfidfs[x], reverse=True)

    # Return list of `n` top files
    return filenames[:n]


def top_sentences(query, sentence_words, word_idfs, n):
    """
    Given a `query` (a set of words), `sentence_words` (a dictionary mapping
    sentences to a list of their words), and `word_idfs` (a dictionary mapping
    words to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # List of sentences
    sentences = list(sentence_words.keys())

    # Store sum of idf values for all words in `query` and query term density (qtd) for each sentence
    query_scores = dict()
    for sentence, words in sentence_words.items():
        idf_sum = 0
        matches = 0

        # Only add idf values for query words that also appear in sentence
        for word in query:
            if word in words:
                idf_sum += word_idfs[word]
                matches += 1

        # Compute query term density (proportion of words in sentence that are also words in query)
        qtd = float(matches) / len(words)

        # Update `query_scores`
        query_scores[sentence] = {"idf": idf_sum, "qtd": qtd}

    # Sort sentences from highest idf rank to lowest
    # If there are ties for idf, rank by qtd
    sentences.sort(
        key=lambda x: (query_scores[x]["idf"], query_scores[x]["qtd"]), reverse=True
    )

    # Return list of top `n` sentences
    return sentences[:n]


if __name__ == "__main__":
    main()
