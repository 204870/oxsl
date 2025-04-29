import gensim
from gensim.models import Word2Vec
import os
from scipy.io import savemat

# replace filename depending on dataset
txt_files = []
f = open("list_yolocats")
for i in f:
    txt_files.append(i[:-1])
f.close()

# Read all text files and tokenize sentences into words
sentences = []
for file in txt_files:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            # Split each line into tokens (words)
            tokens = line.strip().split()
            sentences.append(tokens[0])

# Train the model
model = Word2Vec(
    sentences,       # Tokenized sentences
    vector_size=8000, # Dimension of embeddings
    window=5,        # Context window size
    min_count=1,     # Ignore words with frequency < min_count
    workers=4        # Parallel threads
)

# Extract the embedding matrix and vocabulary
vocab = model.wv.index_to_key  # List of words
word_vectors = model.wv.vectors  # Numpy array (shape: [vocab_size, vector_size])

print("Embedding array shape:", word_vectors.shape)
print("Example word:", vocab[0], "→ Vector:", word_vectors[0])

# replace filename depending on dataset
savemat("../../word_vectors_yolo.mat", {
    "vocab": vocab,          # Vocabulary (list of words)
    "word_vectors": word_vectors  # Embedding matrix
})
