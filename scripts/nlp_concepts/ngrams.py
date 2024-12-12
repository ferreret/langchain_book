from nltk.util import ngrams
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer

# ====================================================================================================
# NLTK
print("*" * 25)
print("Bellow example of N Grams is using NLTK package")
text = "This is an example sentence for creating n-grams."
n = 2
bigrams = list(ngrams(text.split(), n))
print(bigrams)

print("Ahora hago un ejemplo en español")
text = "Esta es una oración de ejemplo para crear n-gramas."
n = 2
bigrams = list(ngrams(text.split(), n))
print(bigrams)

# ====================================================================================================
# SpaCy
print("*" * 25)
print("Bellow example of N Grams is using SpaCy package")
# It is to download english package. Not required to run everytime
# !python3 -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")
text = "This is an example sentence for creating n-grams."
n = 2 # Specify the n-gram you want to create
tokens = [token.text for token in nlp(text)]
ngrams_spacy = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
print(ngrams_spacy)

# ====================================================================================================
# TextBlob
# print("*" * 25)
# print("Bellow example of N Grams is using TextBlob package")
# # This is to download required corpora. Not required to run everytime
# # Run below code from terminal after activationg virtual environment"
# # !python3 -m textblob.download_corpora

# text = "This is an example sentence for creating n-grams."
# n = 2
# blob = TextBlob(text)
# bigrams = list(blob.ngrams(n))
# print(bigrams)

# ====================================================================================================
# Scikit-learn
print("*" * 25)
print("Bellow example of N Grams is using Scikit-learn package")
text = ["This is an example sentence for creating n-grams."]
n = 2
vectorizer = CountVectorizer(ngram_range=(n, n))
X = vectorizer.fit_transform(text)
# Get the n-gram feature names
feature_names = vectorizer.get_feature_names_out()
# Print the n-grams
for feature_name in feature_names:
    print(feature_name)

# ====================================================================================================
# Hugging Face Transformers
print("*" * 25)
print("Bellow example of N Grams is using Hugging Face Transformers package")

# Define your text
text = "This is an example sentence for creating n-grams with Hugging Face Transformers."

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the text
tokens = tokenizer.tokenize(text)

# Generate bigrams
bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

# Generate trigrams
trigrams = [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(len(tokens) - 2)]

# Print the bigrams and trigrams
for bigram in bigrams:
    print(bigram)

for trigram in trigrams:
    print(trigram)