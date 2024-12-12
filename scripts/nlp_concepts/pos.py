import nltk
import spacy
from textblob import TextBlob

# =====================================================================================================
# NLTK
print("*"*25)
print("Bellow example of POS tagging using NLTK package")

# Download the Punkt tokenizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

text = "This is an example sentence for part-of-speech tagging."
words = nltk.word_tokenize(text)
tagged_words = nltk.pos_tag(words)

for word, tag in tagged_words:
    print(f"{word} -> {tag}")

# =====================================================================================================
# Spacy
print("*"*25)
print("Bellow example of POS tagging using Spacy package")

nlp = spacy.load("en_core_web_sm")

text = "This is an example sentence for part-of-speech tagging."
doc = nlp(text)

for token in doc:
    print(f"{token.text} -> {token.pos_}")

# =====================================================================================================
# TextBlob
print("*"*25)
print("Bellow example of POS tagging using TextBlob package")

text = "This is an example sentence for part-of-speech tagging."
blob = TextBlob(text)

for word, tag in blob.tags:
    print(f"{word} -> {tag}")