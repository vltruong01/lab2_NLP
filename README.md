# lab2_NLP
 
![alt text](image.png)

Task 1:
Dataset Chosen: The Complete Works of William Shakespeare (Plain Text UTF-8)
The dataset consists of the complete works of William Shakespeare, including 
all his plays, sonnets, and poems. This text-rich dataset is suitable for language 
modeling as it contains a diverse range of vocabulary, styles, and contexts. 
The dataset is sourced from Project Gutenberg, a reputable public repository of free eBooks.

Source: Project Gutenberg (https://www.gutenberg.org/ebooks/100)

Task 2:
Steps to Preprocess the Text Data
Load the Dataset: Read the text data from shakespeare.txt.
Tokenization: Split the text into tokens (words or characters).
Vocabulary Creation: Create a vocabulary of unique tokens.
Sequence Creation: Convert the text into sequences of fixed length.
Encoding: Encode the sequences into numerical format using the vocabulary.
Padding: Pad the sequences to ensure uniform length.
Train-Test Split: Split the data into training and validation sets.

Model Architecture and Training Process
Model Architecture:
Embedding Layer: Converts input tokens into dense vectors of fixed size.
LSTM Layers: Two LSTM layers to capture temporal dependencies in the data.
Dense Layer: Fully connected layer to output probabilities for each token in the vocabulary.
Activation Function: Softmax activation function to get the probability distribution.

Training Process:
Loss Function: Categorical Cross-Entropy to measure the difference between predicted and actual token distributions.
Optimizer: Adam optimizer to update model weights.
Batch Size: Number of samples per gradient update.
Epochs: Number of times the entire dataset is passed through the model.(5 Epochs)
Validation: Monitor the model's performance on a validation set to prevent overfitting.

Task 3: Web Deployed (FLask)