import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
# from gensim.corpora import Dictionary
from collections import defaultdict
# from gensim.models import TfidfModel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ================================================================================================================================
# Download required data
nltk.download("punkt")

# ================================================================================================================================
# NLTK
print("*" * 25)
print("Bellow example of Bag Of Words is using NLTK package")

text = """This is a sample document. Another document with somewords. Repeating document with some words.
        A third document for illustration. repeating illustration."""

words = word_tokenize(text)
fdist = FreqDist(words)

fdist.pprint()

# ================================================================================================================================
# Scikit Learn
print("*" * 25)
print("Bellow example of Bag Of Words is using Scikit Learn package Count Method")

documents = ["This is a sample document.", "Another document with some words. Repeating document with some words.",
             "A third document for illustration. Repeating illustration."]

# Join the list of document into a single string
corpus = " ".join(documents)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([corpus])

# Get the feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Get the word frequencies from the CountVectorizer's array
word_frequencies = X.toarray()[0]

# Print words with their frequencies
for word, frequency in zip(feature_names, word_frequencies):
    print(f"Word: {word}: Frequency:{frequency}")

# ================================================================================================================================
# Scikit Learn with TFIDF
print("*" * 25)
print("Bellow example of Bag Of Words is using Scikit Learn package TFIDF Method")

documents = [
    "This is a sample document.",
    "Another document with some words. Repeating document with some words.",
    "A third document for illustration. Repeating illustration."
]

# Join the list of document into a single string
corpus = " ".join(documents)

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform([corpus])

# Get the feature names (words)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Get the word frequencies from the CountVectorizer's array
tfidf_values = X.toarray()[0]

# Print words with their TF-IDF values
for word, tfidf in zip(feature_names, tfidf_values):
    print(f"Word: {word}: TF-IDF:{tfidf}")