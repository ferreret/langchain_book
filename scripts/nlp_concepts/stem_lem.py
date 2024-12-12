import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import spacy
from textblob import Word

# =====================================================================================================
# NLTK
print("*"*25)
print("Bellow example of Stemming using NLTK package")
nltk.download('punkt')

# Create a PorterStemmer instance
stemmer = PorterStemmer()

# Example words for stemming
words = ["jumps", "jumping", "jumper", "flies", "flying"]

# Perform stemming on each word
stemmed_words = [stemmer.stem(word) for word in words]

# Print the original and stemmed words
for word, stemmed_word in zip(words, stemmed_words):
    print(f"{word} -> {stemmed_word}")

# =====================================================================================================
# NLTK
print("*"*25)
print("Bellow example of Lemmatization using NLTK package")
nltk.download('wordnet')

# Create a WordNetLemmatizer instance
lemmatizer = WordNetLemmatizer()

# Example words for lemmatization
words = ["jumps", "jumping", "jumper", "flies", "flying"]

# Perform lemmatization on each word
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

# Print the original and lemmatized words
for word, lemmatized_word in zip(words, lemmatized_words):
    print(f"{word} -> {lemmatized_word}")

# =====================================================================================================
# Spacy
print("*"*25)
print("Bellow example of Lemmatization using Spacy package")

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Example words for lemmatization
words = ["jumps", "jumping", "jumper", "flies", "flying"]

# Perform lemmatization on each word
lemmatized_words = [token.lemma_ for token in nlp(" ".join(words))]

# Print the original and lemmatized words
for word, lemmatized_word in zip(words, lemmatized_words):
    print(f"{word} -> {lemmatized_word}")

# =====================================================================================================
# TextBlob
print("*"*25)
print("Bellow example of Lemmatization using TextBlob package")

# Example words for lemmatization
words = ["jumps", "jumping", "jumper", "flies", "flying"]

# Perform lemmatization on each word
lemmatized_words = [Word(word).lemmatize("v") for word in words] # v for verb

# Print the original and lemmatized words
for word, lemmatized_word in zip(words, lemmatized_words):
    print(f"{word} -> {lemmatized_word}")