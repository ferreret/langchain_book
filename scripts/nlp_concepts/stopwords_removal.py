import nltk
from nltk.corpus import stopwords
import spacy
# from gensim.parsing.preprocessing import remove_stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# In case you get an error "ImportError: cannot import name `triu`from `scipy.linalg`",
# when importing Gensim, please install specific version of scipy
# !pip install scipy==1.12

# ====================================================================================================
# NLTK
print("*" * 25)
print("Bellow example of Stopwords Removal is using NLTK package")

nltk.download("stopwords")  # Download necessary data (if not already downloaded)

text = "This is an example sentence with some stop words."

words = text.split()
filtered_words = [
    word for word in words if word.lower() not in stopwords.words("english")
]
print("Filtered words:", filtered_words)

# ====================================================================================================
# Spacy
print("*" * 25)
print("Bellow example of Stopwords Removal is using Spacy package")

nlp = spacy.load("en_core_web_sm")
text = "This is an example sentence with some stop words."

doc = nlp(text)

filtered_words = [token.text for token in doc if not token.is_stop]
print("Filtered words:", filtered_words)

# ====================================================================================================
# Gensim
# print("*" * 25)
# print("Bellow example of Stopwords Removal is using Gensim package")

# text = "This is an example sentence with some stop words."

# filtered_words = remove_stopwords(text)
# print("Filtered words:", filtered_words)

# ====================================================================================================
# Scikit Learn
print("*" * 25)
print("Bellow example of Stopwords Removal is using Scikit Learn package")

text = "This is an example sentence with some stop words."
words = text.split()
filtered_words = [word for word in words if word.lower() not in ENGLISH_STOP_WORDS]

print("Filtered words:", filtered_words)