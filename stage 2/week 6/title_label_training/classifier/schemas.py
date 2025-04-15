import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np


class TitleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(TitleClassifier, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def predict_proba(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs.numpy()

    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1)
