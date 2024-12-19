import spacy
from transformers import DistilBertTokenizer, DistilBertModel

# ====================================================================================  
# Spacy
print("*" * 25)
print("Bellow example of Word Embeddings using Spacy package")

# Load the pre-trained English model
nlp = spacy.load("en_core_web_sm")

# Process a text to get word embeddings
doc = nlp("This is an example sentence for word embeddings. Word embeddings capture semantic relationships. Gensin is a popular library for word embeddings.")
word_vector = doc[0].vector
print(word_vector)


# ====================================================================================
# Huggingface
# Huggingface section
print("*" * 25)
print("Bellow example of Word Embeddings using Huggingface package")

# Load the pre-trained DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize a sentence - correct format for the model
text = "Hugging Face's Transformers library is fantastic!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Load the pre-trained DistilBERT model
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Get the word embeddings
outputs = model(**inputs)

# Access word embeddings for the first token
word_embeddings = outputs.last_hidden_state[0]

# Convert to numpy array for display
word_embeddings_np = word_embeddings.detach().numpy()
print(word_embeddings_np)