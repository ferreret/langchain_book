import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from transformers import AutoTokenizer
from textblob import TextBlob

# ====================================================================================================
# NLTK
print("*" * 25)
print("Bellow example of Tokens is using NLTK package")

# Download required dataset. Not required to run everytime
nltk.download("punkt")
nltk.download("punkt_tab")
text = "This is an example sentence. Tokenize it."

# Word tokenization
words = word_tokenize(text)
print("Word tokens:", words)

# Sentence tokenization
sentences = sent_tokenize(text)
print("Sentence tokens:", sentences)

# ====================================================================================================
# Spacy
print("*" * 25)
print("Bellow example of Tokens is using Spacy package")
# It is to download english package. Not required to run everytime
# Run below code from terminal after activating virtual environment
# !python3 -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")
text = "This is an example sentence. Tokenize it."
doc = nlp(text)

# Word tokenization
words = [token.text for token in doc]
print("Word tokens:", words)

# Sentence tokenization
sentences = [sent.text for sent in doc.sents]
print("Sentence tokens:", sentences)

# ====================================================================================================
# Builtin Methods
print("*" * 25)
print("Bellow example of Tokens is using Builtin Package")
text = "This is an example sentence. Tokenize it."

# Word tokenization
words = text.split()
print("Word tokens:", words)
# Sentence tokenization
sentences = text.split(".")
# Remove 3rd element wich is empty string. Also remove leading and trailing whitespaces
sentences = [sentence.strip() for sentence in sentences if sentence]
print("Sentence tokens:", sentences)

# ====================================================================================================
# Hugging Face Transformers
print("*" * 25)
print("Bellow example of Tokens is using Hugging Face Transformers package")

# Use pretrained model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "This is an example sentence. Tokenize it."

# Word tokenization
words = tokenizer.tokenize(text)
print("Word tokens:", words)

# We tokenize the text into sentence-level tokens by adding special tokens [CLS] and [SEP] to the output
# [CLS] is used to represent the start of the sentence and [SEP] is used to represent the end of the sentence
# Sentence tokenization
sent_tokens = tokenizer.encode(text, add_special_tokens=True)
print("Sentence tokens:", sent_tokens)

# Optionally, you can convert the sentence tokens into actual sentences
sentences = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(sent_tokens))
print("Sentences:", sentences)

# ====================================================================================================
# TextBlob
print("*" * 25)
print("Bellow example of Tokens is using TextBlob package")
text = "This is an example sentence. Tokenize it."
blob = TextBlob(text)

# Word tokenization
words = blob.words
print("Word tokens:", words)

# Sentence tokenization
sentences = blob.sentences
print("Sentence tokens:", sentences)