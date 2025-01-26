import pickle

# Load the dataset with UTF-8 encoding
with open('../shakespeare.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenization
tokens = text.split()

# Vocabulary creation
vocab = sorted(set(tokens))

# Save the vocabulary to a pickle file
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)