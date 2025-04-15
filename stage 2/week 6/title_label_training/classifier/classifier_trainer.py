from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from classifier.schemas import TitleClassifier


def encode_labels(labels: List[str]) -> Tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    return encoder.fit_transform(labels), encoder


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 2) -> float:
    topk_preds = torch.topk(logits, k=k, dim=1).indices
    correct = topk_preds.eq(targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    epochs: int = 200,
    patience: int = 30,
    min_delta: float = 1e-4,
    learning_rate: float = 0.001,
) -> TitleClassifier:
    model = TitleClassifier(input_dim, hidden_dims, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        inputs = torch.tensor(X_train, dtype=torch.float32)
        targets = torch.tensor(y_train, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'𖧨 Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

        # Early stopping
        if loss.item() < best_loss - min_delta:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f'\n⏹️ Early stopping at epoch {epoch + 1}, best loss: {best_loss:.4f}'
            )
            break

    return model, {
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'output_dim': output_dim,
        'model_state_dict': model.state_dict(),
    }


def evaluate_classifier(
    model: TitleClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    encoder: LabelEncoder,
) -> None:
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32)
        targets = torch.tensor(y_test, dtype=torch.long)
        preds = model(inputs)
        pred_labels = torch.argmax(preds, dim=1).numpy()

        acc = accuracy_score(y_test, pred_labels)
        print(f'\n✅ Test Accuracy (Top-1): {acc * 100:.2f}%')

        top2_acc = top_k_accuracy(preds, targets, k=2)
        top3_acc = top_k_accuracy(preds, targets, k=3)
        print(f'🌟 Top-2 Accuracy: {top2_acc * 100:.2f}%')
        print(f'🌟 Top-3 Accuracy: {top3_acc * 100:.2f}%')

        print('\n📊 Classification Report:')
        print(classification_report(y_test, pred_labels, target_names=encoder.classes_))
