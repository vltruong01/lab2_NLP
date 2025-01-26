from flask import Flask, request, render_template
import torch
import torch.nn as nn
import pickle

# Define the LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

# Load the vocabulary
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)  # Set vocab_size to the actual size of the vocabulary
token_to_idx = {token: idx for idx, token in enumerate(vocab)}
idx_to_token = {idx: token for token, idx in token_to_idx.items()}

# Model parameters
embed_size = 128
hidden_size = 256
num_layers = 2

# Initialize and load the model
model = LSTMModel(vocab_size, embed_size, hidden_size, num_layers)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Initialize Flask app
app = Flask(__name__)

# Define the text generation function
def generate_text(prompt, max_seq_len=30):
    model.eval()
    tokens = prompt.split()
    input_seq = [token_to_idx.get(token, 0) for token in tokens]
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0)
    
    generated_tokens = tokens[:]
    for _ in range(max_seq_len - len(tokens)):
        with torch.no_grad():
            output = model(input_tensor)
            next_token_idx = torch.argmax(output[0, -1]).item()
            next_token = idx_to_token[next_token_idx]
            generated_tokens.append(next_token)
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_idx]], dtype=torch.long)], dim=1)
    
    generated_text = ' '.join(generated_tokens)
    print(f"Generated text: {generated_text}")  # Debugging statement
    return generated_text

# Define the home route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt = request.form['prompt']
        generated_text = generate_text(prompt)
        print(f"Prompt: {prompt}")  # Debugging statement
        print(f"Generated Text: {generated_text}")  # Debugging statement
        return render_template('index.html', prompt=prompt, generated_text=generated_text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)