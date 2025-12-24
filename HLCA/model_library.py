import torch
import torch.nn as nn

# --- Model Definition ---
class BinaryClassifier(nn.Module):
    def __init__(self, num_genes):
        super(BinaryClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(num_genes, 250),
            nn.ReLU(),
            nn.Linear(250, 50),
            nn.ReLU(),
            nn.Linear(50, 1) 
        )

    def forward(self, x):
        return self.network(x)


# --- QOI Function ---
def probability_qoi(inputs, model):
    """
    Quantity of Interest: Predicted Probability (Sigmoid of logits).
    """
    logits = model(inputs)
    return torch.sigmoid(logits)
