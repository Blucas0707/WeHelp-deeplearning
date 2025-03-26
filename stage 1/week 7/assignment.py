import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, DataLoader


class WeightDataset(Dataset):
    def __init__(self, filename: str):
        self.data = self.read_csv(filename)
        self.standardize()

    def read_csv(self, filename: str):
        data_list = []

        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)

            for row in reader:
                gender = 0 if row[0] == 'Female' else 1
                height = float(row[1])
                weight = float(row[2])
                data_list.append([gender, height, weight])
        return np.array(data_list)

    def standardize(self):
        heights = self.data[:, 1]
        weights = self.data[:, 2]
        self.height_mean, self.height_std = np.mean(heights), np.std(heights)
        self.weight_mean, self.weight_std = np.mean(weights), np.std(weights)
        self.data[:, 1] = (heights - self.height_mean) / self.height_std
        self.data[:, 2] = (weights - self.weight_mean) / self.weight_std

    def __getitem__(self, index):
        gender, height, weight = self.data[index]
        return torch.tensor([gender, height], dtype=torch.float32), torch.tensor(
            [weight], dtype=torch.float32
        )

    def __len__(self):
        return len(self.data)


class TitanicDataset(Dataset):
    def __init__(self, filename: str):
        self.data = self.read_csv(filename)
        self.standardize()

    def read_csv(self, filename: str):
        data_list = []

        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)

            for row in reader:
                pclass = int(row[2])
                sex = 0 if row[4] == 'female' else 1
                age = float(row[5]) if row[5] else np.nan
                sibsp = int(row[6])
                parch = int(row[7])
                fare = float(row[9]) if row[9] else np.nan
                survived = int(row[1])
                data_list.append([pclass, sex, age, sibsp, parch, fare, survived])

            data_array = np.array(data_list)
            age_median = np.nanmedian(data_array[:, 2])
            fare_median = np.nanmedian(data_array[:, 5])

            for row in data_array:
                if np.isnan(row[2]):
                    row[2] = age_median
                if np.isnan(row[5]):
                    row[5] = fare_median

        return np.array(data_array)

    def standardize(self):
        age = self.data[:, 2]
        fare = self.data[:, 5]

        age_mean, age_std = np.mean(age), np.std(age)
        fare_mean, fare_std = np.mean(fare), np.std(fare)

        self.data[:, 2] = (age - age_mean) / age_std
        self.data[:, 5] = (fare - fare_mean) / fare_std

    def __getitem__(self, index):
        pclass, sex, age, sibsp, parch, fare, survived = self.data[index]
        return torch.tensor(
            [pclass, sex, age, sibsp, parch, fare], dtype=torch.float32
        ), torch.tensor([survived], dtype=torch.float32)

    def __len__(self):
        return len(self.data)


class WeightNN(torch.nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class TitanicNN(torch.nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: _Loss,
    optimizer: torch.optim.Optimizer,
    epochs: int = 100,
):
    for _ in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


def test_weight_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: _Loss,
    weight_std: float,
    weight_mean: float,
):
    model.eval()
    test_loss = 0.0
    mae_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            outputs_real = outputs * weight_std + weight_mean
            targets_real = targets * weight_std + weight_mean
            mae_loss += torch.mean(torch.abs(outputs_real - targets_real)).item()

    rmse_real = np.sqrt((test_loss / len(dataloader)) * (weight_std**2))
    mae_real = mae_loss / len(dataloader)
    print(f'Test Loss (RMSE): {rmse_real:.4f}, MAE: {mae_real:.4f}\n')


def test_titanic_model(model: nn.Module, dataloader: DataLoader, criterion: _Loss):
    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            predicted_labels = (outputs >= 0.5).float()
            correct_predictions += (predicted_labels == targets).sum().item()
            total_samples += targets.size(0)

    avg_test_loss = test_loss / len(dataloader)
    accuracy = correct_predictions / total_samples * 100
    print(f'Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%\n')


class NeuralNetworkTaskHandler:
    @staticmethod
    def exec_task_weight_prediction():
        dataset = WeightDataset('gender-height-weight.csv')
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

        model = WeightNN(input_size=2, hidden_sizes=[16, 8], output_size=1)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        train_model(model, train_loader, criterion, optimizer)
        print('Weight Prediction Task Training completed')
        test_weight_model(
            model,
            test_loader,
            criterion,
            dataset.weight_std,
            dataset.weight_mean,
        )

    @staticmethod
    def exec_task_titantic_prediction():
        dataset = TitanicDataset('titanic.csv')
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

        model = TitanicNN(input_size=6, hidden_sizes=[16, 8], output_size=1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        train_model(model, train_loader, criterion, optimizer)
        print('Titanic Prediction Task Training completed')
        test_titanic_model(model, test_loader, criterion)


if __name__ == '__main__':
    NeuralNetworkTaskHandler.exec_task_weight_prediction()
    NeuralNetworkTaskHandler.exec_task_titantic_prediction()
