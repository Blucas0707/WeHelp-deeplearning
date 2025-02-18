from abc import ABC, abstractmethod
import numpy as np
import csv
import math
import torch


class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def get_deviation(self, x):
        pass


class Linear(ActivationFunction):
    def __call__(self, x):
        return x

    def get_deviation(self, x):
        return np.ones_like(x)


class ReLU(ActivationFunction):
    def __call__(self, x):
        return np.maximum(0, x)

    def get_deviation(self, x):
        return np.where(x > 0, 1, 0)


class Sigmoid(ActivationFunction):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def get_deviation(self, x):
        return self(x) * (1 - self(x))


class LossFunction(ABC):
    @abstractmethod
    def get_total_loss(self, output_values: np.ndarray, expected_values: np.ndarray):
        pass

    @abstractmethod
    def get_output_losses(self, output_values: np.ndarray, expected_values: np.ndarray):
        pass


class MSE(LossFunction):
    def get_total_loss(self, output_values: np.ndarray, expected_values: np.ndarray):
        output_values = output_values.reshape(expected_values.shape)
        return np.mean(np.square(expected_values - output_values), axis=0)

    def get_output_losses(self, output_values: np.ndarray, expected_values: np.ndarray):
        return 2 * (output_values - expected_values) / output_values.size


class BCE(LossFunction):
    def get_total_loss(self, output_values: np.ndarray, expected_values: np.ndarray):
        return -np.sum(
            expected_values * np.log(output_values)
            + (1 - expected_values) * np.log(1 - output_values)
        )

    def get_output_losses(self, output_values: np.ndarray, expected_values: np.ndarray):
        return (output_values - expected_values) / (output_values * (1 - output_values))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def standard_deviation(values: list[float], mean_value: float):
    variance = sum((x - mean_value) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def z_score_standardize(values: list[float]) -> list[float]:
    mean_value = mean(values)
    std_dev = standard_deviation(values, mean_value)

    if std_dev == 0:
        return [0] * len(values)

    return [(x - mean_value) / std_dev for x in values]


def read_gender_height_weight_csv(filename: str):
    data_list = []

    with open(filename, "r") as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            gender = 0 if row[0] == "Female" else 1
            height = float(row[1])
            weight = float(row[2])
            data_list.append([gender, height, weight])

    return np.array(data_list)


def get_standardized_height_weight_data(data: np.ndarray):
    genders = data[:, 0]
    heights = z_score_standardize(data[:, 1].tolist())
    weights = z_score_standardize(data[:, 2].tolist())

    standardized_data = np.array([genders, heights, weights]).T
    return standardized_data


def read_titantic_csv(filename: str) -> np.ndarray:
    data_list = []

    with open(filename, "r") as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            pclass = int(row[2])
            sex = 0 if row[4] == "female" else 1
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


def get_standardized_titanic_data(data: np.ndarray) -> np.ndarray:
    """對 Titanic 數據進行標準化 (Z-score)，標準化 `Age` 和 `Fare`"""
    pclass = data[:, 0]  # 艙等 (保持不變)
    sex = data[:, 1]  # 性別 (保持不變)

    # 標準化年齡和票價
    age = z_score_standardize(data[:, 2])
    fare = z_score_standardize(data[:, 5])

    sibsp = data[:, 3]  # 兄弟姊妹/配偶數 (保持不變)
    parch = data[:, 4]  # 父母/子女數 (保持不變)

    survived = data[:, 6]  # 存活結果 (標籤，保持不變)

    standardized_data = np.array([pclass, sex, age, sibsp, parch, fare, survived]).T
    return standardized_data


def get_split_train_test_data(data_list: list[np.ndarray], train_ratio: float):
    training_size = int(train_ratio * len(data_list))
    return data_list[:training_size], data_list[training_size:]


class BaseNeuralNetwork(ABC):
    def __init__(
        self,
        hidden_weights: list[np.ndarray],
        hidden_biases: list[np.ndarray],
        hidden_activations: list[ActivationFunction],
        output_weight: np.ndarray,
        output_bias: np.ndarray,
        output_activation: ActivationFunction,
    ):
        self.hidden_weights = hidden_weights
        self.hidden_biases = hidden_biases
        self.hidden_activations = hidden_activations
        self.output_weight = output_weight
        self.output_bias = output_bias
        self.output_activation = output_activation

    @abstractmethod
    def forward(self, input_values: np.ndarray):
        pass

    @abstractmethod
    def backward(self, output_losses: np.ndarray):
        pass

    @abstractmethod
    def zero_grad(self, learning_rate: float):
        pass


class NeuralNetwork(BaseNeuralNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_values: np.ndarray):
        self.hidden_layer_inputs = []
        self.hidden_layer_outputs = [input_values]

        hidden_layer = len(self.hidden_weights)
        for idx in range(hidden_layer):
            hidden_layer_input = (
                np.dot(self.hidden_layer_outputs[-1], self.hidden_weights[idx].T)
                + self.hidden_biases[idx].T
            )  # (batch_size, 8)

            self.hidden_layer_inputs.append(hidden_layer_input)
            self.hidden_layer_outputs.append(
                self.hidden_activations[idx](hidden_layer_input)
            )

        self.output_layer_input = (
            np.dot(self.hidden_layer_outputs[-1], self.output_weight.T)
            + self.output_bias.T
        )

        self.output_layer_output = self.output_activation(self.output_layer_input)
        return self.output_layer_output.reshape(-1, 1)  # (batch_size, 1)

    def backward(self, output_losses: np.ndarray):
        # # Output layer
        output_layer_delta = output_losses * self.output_activation.get_deviation(
            self.output_layer_input
        )

        self.output_layer_weight_gradient = np.dot(
            output_layer_delta.T, self.hidden_layer_outputs[-1]
        )

        self.output_bias_gradient = np.sum(output_layer_delta, axis=0, keepdims=True)

        # Hidden layer
        self.hidden_layer_weight_gradients = []
        self.hidden_layer_bias_gradients = []

        hidden_layer_delta = np.dot(output_layer_delta, self.output_weight)

        for i in reversed(range(len(self.hidden_weights))):
            hidden_layer_delta *= self.hidden_activations[i].get_deviation(
                self.hidden_layer_inputs[i]
            )

            self.hidden_layer_weight_gradients.append(
                np.dot(hidden_layer_delta.T, self.hidden_layer_outputs[i])
            )
            self.hidden_layer_bias_gradients.append(
                np.sum(hidden_layer_delta, axis=0, keepdims=True)
            )

            if i > 0:
                hidden_layer_delta = np.dot(hidden_layer_delta, self.hidden_weights[i])

        self.hidden_layer_weight_gradients = self.hidden_layer_weight_gradients[::-1]
        self.hidden_layer_bias_gradients = self.hidden_layer_bias_gradients[::-1]

    def zero_grad(self, learning_rate: float, batch_size: int):
        for i in range(len(self.hidden_weights)):
            self.hidden_weights[i] -= learning_rate * (
                self.hidden_layer_weight_gradients[i] / batch_size
            )
            self.hidden_biases[i] -= learning_rate * (
                self.hidden_layer_bias_gradients[i].T / batch_size
            )

        self.output_weight -= learning_rate * (
            self.output_layer_weight_gradient / batch_size
        )
        self.output_bias -= learning_rate * (self.output_bias_gradient / batch_size)


class NeuralNetworkTaskHandler:
    @staticmethod
    def exec_task_weight_prediction():
        print("\n=======================WEIGHT PREDICTION=========================\n")
        nn = NeuralNetwork(
            hidden_weights=[
                np.random.randn(16, 2),
                np.random.randn(8, 16),
            ],
            hidden_biases=[
                np.random.randn(16, 1),
                np.random.randn(8, 1),
            ],
            hidden_activations=[ReLU(), Linear()],
            output_weight=np.random.randn(1, 8),
            output_bias=np.random.randn(1, 1),
            output_activation=Linear(),
        )

        traning_data_ratio = 0.9
        learning_rate = 0.0075
        batch_size = 100
        epochs = 100

        traning_data = read_gender_height_weight_csv("gender-height-weight.csv")
        standarized_data = get_standardized_height_weight_data(traning_data)
        training_data, testing_data = get_split_train_test_data(
            standarized_data, traning_data_ratio
        )

        patience = 30
        loss = best_loss = float("inf")
        no_improve_count = 0

        for epoch in range(epochs):
            if loss < best_loss:
                best_loss = loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                print(f"Early Stopping at Epoch {epoch}")
                break

            np.random.shuffle(training_data)

            for i in range(0, len(training_data), batch_size):
                batch_training_data = training_data[i : i + batch_size]
                inputs = np.array(
                    [row[:2] for row in batch_training_data]
                )  # (batch_size, 2)
                expected_values = np.array(
                    [row[2] for row in batch_training_data]
                ).reshape(-1, 1)  # (batch_size, 1)

                output_values = nn.forward(inputs)

                loss = MSE().get_total_loss(output_values, expected_values)

                output_losses = MSE().get_output_losses(output_values, expected_values)

                nn.backward(output_losses)
                nn.zero_grad(learning_rate, batch_size)

        all_weights = traning_data[:, 2]
        weight_mean = mean(all_weights)
        weight_std = standard_deviation(all_weights, weight_mean)

        testing_inputs = np.array([row[:2] for row in testing_data])
        testing_expected_values = np.array([row[2] for row in testing_data]).reshape(
            -1, 1
        )

        predicted_values = nn.forward(testing_inputs)
        testing_weight_loss = MSE().get_total_loss(
            predicted_values, testing_expected_values
        )

        predicted_weights = predicted_values * weight_std + weight_mean
        expected_weights = testing_expected_values * weight_std + weight_mean
        MSE_testing_weight_loss = np.sqrt(testing_weight_loss) * weight_std

        MAE_loss = np.mean(np.abs(expected_weights - predicted_weights))

        print(f"MSE_loss={MSE_testing_weight_loss}, {MAE_loss=}\n")

    @staticmethod
    def exec_task_titantic_prediction():
        print("\n=======================TITANIC PREDICTION=========================\n")
        nn = NeuralNetwork(
            hidden_weights=[
                np.random.randn(16, 6),
                np.random.randn(8, 16),
            ],
            hidden_biases=[
                np.random.randn(16, 1),
                np.random.randn(8, 1),
            ],
            hidden_activations=[ReLU(), ReLU()],
            output_weight=np.random.randn(1, 8),
            output_bias=np.random.randn(1, 1),
            output_activation=Sigmoid(),
        )

        traning_data_ratio = 0.8
        learning_rate = 0.01
        batch_size = 64
        epochs = 100

        raw_traning_data = read_titantic_csv("titanic.csv")
        standardized_training_data = get_standardized_titanic_data(raw_traning_data)

        training_data, testing_data = get_split_train_test_data(
            standardized_training_data, traning_data_ratio
        )

        for _ in range(epochs):
            np.random.shuffle(training_data)

            for i in range(0, len(training_data), batch_size):
                batch_training_data = training_data[i : i + batch_size]
                inputs = np.array(
                    [row[:6] for row in batch_training_data]
                )  # (batch_size, 5)

                expected_values = np.array(
                    [row[6] for row in batch_training_data]
                ).reshape(-1, 1)  # (batch_size, 1)

                output_values = nn.forward(inputs)

                # loss = BCE().get_total_loss(output_values, expected_values)

                output_losses = BCE().get_output_losses(output_values, expected_values)

                nn.backward(output_losses)
                nn.zero_grad(learning_rate, batch_size)

        testing_inputs = np.array([row[:6] for row in testing_data])
        testing_expected_values = np.array([row[6] for row in testing_data]).reshape(
            -1, 1
        )

        threshold = 0.5
        predicted_values = nn.forward(testing_inputs)

        predicted_labels = (predicted_values >= threshold).astype(int)  # 轉成 0 or 1
        correct_predictions = np.sum(predicted_labels == testing_expected_values)
        accuracy = correct_predictions / len(testing_expected_values)

        print(f"Testing Accuracy: {accuracy * 100:.2f}%\n")

    @staticmethod
    def exec_task_3_pytorch():
        # Q1
        data = torch.tensor([[2, 3, 1], [5, -2, 1]])
        print(f"Q1: {data.shape=}, {data.dtype=}\n")

        # Q2
        random_tensor = torch.rand(3, 4, 2)
        print(f"Q2: {random_tensor.shape=}, {random_tensor=}\n")

        # Q3
        ones_tensor = torch.ones(2, 1, 5)
        print(f"Q3: {ones_tensor.shape=}, {ones_tensor=}\n")

        # Q4
        tensor_a = torch.tensor([[1, 2, 4], [2, 1, 3]])
        tensor_b = torch.tensor([[5], [2], [1]])
        print(f"Q4: {torch.matmul(tensor_a, tensor_b)=}\n")

        # Q5
        tensor_a = torch.tensor([[1, 2], [2, 3], [-1, 3]])
        tensor_b = torch.tensor([[5, 4], [2, 1], [1, -5]])
        print(f"Q5: {tensor_a * tensor_b=}\n")


if __name__ == "__main__":
    NeuralNetworkTaskHandler.exec_task_weight_prediction()
    NeuralNetworkTaskHandler.exec_task_titantic_prediction()
    NeuralNetworkTaskHandler.exec_task_3_pytorch()
