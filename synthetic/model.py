import torch
import torch.nn as nn

class FeedForwardNet(nn.Module):
    def __init__(self, input_dim):
        super(FeedForwardNet, self).__init__()
        
        # Architecture: Input -> 200 -> 100 -> 50 -> 1
        self.layer1 = nn.Linear(input_dim, 200)
        self.layer2 = nn.Linear(200, 100)
        self.layer3 = nn.Linear(100, 50)
        self.output = nn.Linear(50, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Standard forward pass. Returns raw logits (no sigmoid) 
        for use with BCEWithLogitsLoss.
        """
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.output(x) 
        return x

    def get_logits(self, x):
        """
        Explicitly returns logits for downstream analysis.
        Identical to forward(), but semantically distinct.
        """
        return self.forward(x)

    def get_embeddings(self, x):
        """
        Returns the 50-dimensional latent vector from the penultimate layer.
        Useful for TDA/clustering analysis before the final classification.
        """
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return x
    
    def predict_proba(self, x):
        """
        Returns the actual probability (0-1) by applying Sigmoid to logits.
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)
