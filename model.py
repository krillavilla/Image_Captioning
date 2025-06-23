import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Using ResNet-50 for richer feature representations
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Freeze pretrained layers
        for param in resnet.parameters():
            param.requires_grad_(False)
        # Remove final classification layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # Linear layer to transform to embed_size
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)          # (batch, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (batch, 2048)
        features = self.embed(features)        # (batch, embed_size)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        features = features.unsqueeze(1)
        inputs = torch.cat((features, embeddings), dim=1)
        lstm_out, _ = self.lstm(inputs)
        outputs = self.linear(lstm_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        predicted_ids = []
        for _ in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            outputs = self.linear(lstm_out.squeeze(1))
            _, predicted = outputs.max(1)
            predicted_ids.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)
        return predicted_ids
