import argparse
import csv
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class TitleClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def load_data(csv_path: str) -> Tuple[List[List[str]], List[str]]:
    texts, labels = [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            labels.append(row[0])
            texts.append(row[1:])
    return texts, labels


def tokens_to_vectors(model: Doc2Vec, tokenized_texts: List[List[str]]) -> np.ndarray:
    return np.array([model.infer_vector(tokens) for tokens in tokenized_texts])


def encode_labels(labels: List[str]) -> Tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    return encoder.fit_transform(labels), encoder


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 100,
    epochs: int = 1000,
) -> TitleClassifier:
    model = TitleClassifier(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        inputs = torch.tensor(X_train, dtype=torch.float32)
        targets = torch.tensor(y_train, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')
    return model


def evaluate_classifier(
    model: TitleClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    encoder: LabelEncoder,
) -> None:
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32)
        preds = model(inputs)
        pred_labels = torch.argmax(preds, dim=1).numpy()

        acc = accuracy_score(y_test, pred_labels)
        print(f'\n✅ Test Accuracy: {acc * 100:.2f}%')
        print('\n📊 Classification Report:')
        print(classification_report(y_test, pred_labels, target_names=encoder.classes_))


def run_with_model(model_path: str, texts: List[List[str]], labels: List[str]) -> None:
    print(f'\n📦 載入 Doc2Vec 模型：{model_path}')
    d2v_model = Doc2Vec.load(model_path)
    vectors = tokens_to_vectors(d2v_model, texts)
    y, encoder = encode_labels(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        vectors, y, test_size=0.2, stratify=y, random_state=42
    )
    model = train_classifier(
        X_train,
        y_train,
        input_dim=d2v_model.vector_size,
        output_dim=len(encoder.classes_),
    )
    evaluate_classifier(model, X_test, y_test, encoder)


def main():
    parser = argparse.ArgumentParser(description='Doc2Vec + Classifier Pipeline')
    parser.add_argument(
        '--csv',
        type=str,
        default='sample_50k.csv',
        help='Path to tokenized labeled CSV file',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/d2v_v50_e100_m2_w3.model',
        help='Path to .model file or folder containing models',
    )
    parser.add_argument(
        '--train_doc2vec',
        action='store_true',
        help='Train a new Doc2Vec model if not loading',
    )
    args = parser.parse_args()

    texts, labels = load_data(args.csv)

    if os.path.isdir(args.model):
        print(f'📁 偵測到資料夾：{args.model}，將對所有 .model 檔案進行分類測試')
        for fname in os.listdir(args.model):
            if fname.endswith('.model'):
                run_with_model(os.path.join(args.model, fname), texts, labels)
    else:
        run_with_model(args.model, texts, labels)


if __name__ == '__main__':
    main()
