import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Using ResNet-18 for a lighter model
        resnet = models.resnet18(pretrained=True)
        # Freeze pretrained layers
        for param in resnet.parameters():
            param.requires_grad_(False)
        # Remove final classification layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # Linear layer to transform to embed_size
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)          # (batch, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (batch, 512)
        features = self.embed(features)        # (batch, embed_size)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        # Embedding layer that turns word indices into embeddings
        self.embed = nn.Embedding(vocab_size, embed_size)
        # LSTM to process the sequence of embeddings
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        # Final linear layer to project hidden state to vocabulary scores
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """
        Forward pass through the decoder.
        - features: tensor of shape (batch, embed_size) from CNN encoder
        - captions: tensor of shape (batch, seq_length) of token indices
        Returns:
        - outputs: tensor of shape (batch, seq_length, vocab_size)
        """
        # Embed captions (excluding <end> token at the end)
        embeddings = self.embed(captions[:, :-1])  # (batch, seq_len-1, embed_size)
        # Unsqueeze features to become first input token
        features = features.unsqueeze(1)           # (batch, 1, embed_size)
        # Concatenate image features and caption embeddings
        inputs = torch.cat((features, embeddings), dim=1)  # (batch, seq_len, embed_size)

        # Pass through LSTM
        lstm_out, _ = self.lstm(inputs)           # (batch, seq_len, hidden_size)
        # Project LSTM outputs to vocabulary space
        outputs = self.linear(lstm_out)           # (batch, seq_len, vocab_size)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        """
        Generate captions for given image features using greedy search.
        - inputs: initial input features of shape (1, 1, embed_size)
        - states: initial LSTM states
        - max_len: maximum length of generated caption
        Returns:
        - predicted_ids: list of token ids
        """
        predicted_ids = []
        for _ in range(max_len):
            lstm_out, states = self.lstm(inputs, states)       # lstm_out: (1, 1, hidden_size)
            outputs = self.linear(lstm_out.squeeze(1))         # outputs: (1, vocab_size)
            _, predicted = outputs.max(1)                     # predicted: (1,)
            predicted_ids.append(predicted.item())
            # Prepare next input
            inputs = self.embed(predicted)                    # (1, embed_size)
            inputs = inputs.unsqueeze(1)                      # (1, 1, embed_size)
        return predicted_ids

