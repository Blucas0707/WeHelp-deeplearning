from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x):
        pass


class Linear(ActivationFunction):
    def __call__(self, x):
        return x


class ReLU(ActivationFunction):
    def __call__(self, x):
        return np.maximum(0, x)


class Sigmoid(ActivationFunction):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))


class Softmax(ActivationFunction):
    def __call__(self, x: np.ndarray):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


class LossFunction(ABC):
    @abstractmethod
    def __call__(self, x):
        pass


class MSE(LossFunction):
    def __call__(self, expected_values: np.ndarray, output_values: np.ndarray):
        return np.mean((expected_values - output_values) ** 2)


class BCE(LossFunction):
    def __call__(
        self, expected_binary_values: np.ndarray, output_binary_values: np.ndarray
    ):
        return -np.sum(
            expected_binary_values * np.log(output_binary_values)
            + (1 - expected_binary_values) * np.log(1 - output_binary_values)
        )


class CCE(LossFunction):
    def __call__(
        self,
        expected_probability_values: np.ndarray,
        output_probability_values: np.ndarray,
    ):
        return -np.sum(expected_probability_values * np.log(output_probability_values))


@dataclass
class NetworkConfig:
    hidden_weights: list[np.ndarray]
    hidden_biases: list[np.ndarray]
    hidden_activation: ActivationFunction
    output_weights: np.ndarray
    output_bias: np.ndarray
    output_activation: ActivationFunction


class NeuralNetwork(ABC):
    def __init__(self, config: NetworkConfig):
        self.config = config

        self._validate_config()

    @abstractmethod
    def _validate_config(self):
        pass

    @abstractmethod
    def forward(self, input_values: np.ndarray):
        pass


class RegressionNetwork(NeuralNetwork):
    def __init__(self, config: NetworkConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        if len(self.config.hidden_biases) != len(self.config.hidden_weights):
            raise ValueError(
                'Number of hidden biases must match number of hidden weights'
            )

    def forward(self, input_values: np.ndarray):
        hidden_layer = input_values
        for idx, hidden_weight in enumerate(self.config.hidden_weights):
            hidden_layer = (
                np.dot(hidden_weight, hidden_layer) + self.config.hidden_biases[idx]
            )
            hidden_layer = self.config.hidden_activation(hidden_layer)

        output_layer = (
            np.dot(self.config.output_weights, hidden_layer) + self.config.output_bias
        )

        return self.config.output_activation(output_layer)


class BinaryClassificationNetwork(NeuralNetwork):
    def __init__(self, config: NetworkConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        if len(self.config.hidden_biases) != len(self.config.hidden_weights):
            raise ValueError(
                'Number of hidden biases must match number of hidden weights'
            )

    def forward(self, input_values: np.ndarray):
        hidden_layer = input_values
        for idx, hidden_weight in enumerate(self.config.hidden_weights):
            hidden_layer = (
                np.dot(hidden_weight, hidden_layer) + self.config.hidden_biases[idx]
            )
            hidden_layer = self.config.hidden_activation(hidden_layer)

        output_layer = (
            np.sum(np.dot(self.config.output_weights, hidden_layer))
            + self.config.output_bias
        )
        return self.config.output_activation(output_layer)


class MultiLabelClassificationNetwork(NeuralNetwork):
    def __init__(self, config: NetworkConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        if len(self.config.hidden_biases) != len(self.config.hidden_weights):
            raise ValueError(
                'Number of hidden biases must match number of hidden weights'
            )

    def forward(self, input_values: np.ndarray):
        hidden_layer = input_values
        for idx, hidden_weight in enumerate(self.config.hidden_weights):
            hidden_layer = (
                np.dot(hidden_weight, hidden_layer) + self.config.hidden_biases[idx]
            )
            hidden_layer = self.config.hidden_activation(hidden_layer)

        output_layer = (
            np.dot(self.config.output_weights, hidden_layer) + self.config.output_bias
        )
        return self.config.output_activation(output_layer)


class MultiClassClassificationNetwork(NeuralNetwork):
    def __init__(self, config: NetworkConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        if len(self.config.hidden_biases) != len(self.config.hidden_weights):
            raise ValueError(
                'Number of hidden biases must match number of hidden weights'
            )

    def forward(self, input_values: np.ndarray):
        hidden_layer = input_values
        for idx, hidden_weight in enumerate(self.config.hidden_weights):
            hidden_layer = (
                np.dot(hidden_weight, hidden_layer) + self.config.hidden_biases[idx]
            )
            hidden_layer = self.config.hidden_activation(hidden_layer)

        output_layer = (
            np.dot(self.config.output_weights, hidden_layer) + self.config.output_bias
        )
        return self.config.output_activation(output_layer)


class NeuralNetworkTaskHandler:
    @staticmethod
    def exec_task_1_regression():
        network_config = NetworkConfig(
            hidden_weights=[np.array([[0.5, 0.2], [0.6, -0.6]])],
            hidden_biases=[np.array([0.3, 0.25])],
            hidden_activation=ReLU(),
            output_weights=[np.array([[0.8, -0.5], [0.4, 0.5]])],
            output_bias=[np.array([0.6, -0.25])],
            output_activation=Linear(),
        )

        nn = RegressionNetwork(config=network_config)
        output_values = nn.forward(np.array([1.5, 0.5]))
        expected_values = np.array([0.8, 1])
        print('Total Loss:', MSE()(expected_values, output_values))

        output_values = nn.forward(np.array([0, 1]))
        expected_values = np.array([0.5, 0.5])
        print('Total Loss:', MSE()(expected_values, output_values))

    @staticmethod
    def exec_task_2_binary_classification():
        network_config = NetworkConfig(
            hidden_weights=[np.array([[0.5, 0.2], [0.6, -0.6]])],
            hidden_biases=[np.array([0.3, 0.25])],
            hidden_activation=ReLU(),
            output_weights=np.array([[0.8, 0], [0, 0.4]]),
            output_bias=-0.5,
            output_activation=Sigmoid(),
        )

        nn = BinaryClassificationNetwork(config=network_config)
        output_values = nn.forward(np.array([0.75, 1.25]))
        expected_values = np.array([1])
        print('Total Loss:', BCE()(expected_values, output_values))

        output_values = nn.forward(np.array([-1, 0.5]))
        expected_values = np.array([0])
        print('Total Loss:', BCE()(expected_values, output_values))

    @staticmethod
    def exec_task_3_multi_label_classification():
        network_config = NetworkConfig(
            hidden_weights=[np.array([[0.5, 0.2], [0.6, -0.6]])],
            hidden_biases=[np.array([0.3, 0.25])],
            hidden_activation=ReLU(),
            output_weights=np.array([[0.8, -0.4], [0.5, 0.4], [0.3, 0.75]]),
            output_bias=np.array([0.6, 0.5, -0.5]),
            output_activation=Sigmoid(),
        )

        nn = MultiLabelClassificationNetwork(config=network_config)
        output_values = nn.forward(np.array([1.5, 0.5]))
        expected_values = np.array([1, 0, 1])
        print('Total Loss:', BCE()(expected_values, output_values))

        output_values = nn.forward(np.array([0, 1]))
        expected_values = np.array([1, 1, 0])
        print('Total Loss:', BCE()(expected_values, output_values))

    @staticmethod
    def exec_task_4_multi_class_classification():
        network_config = NetworkConfig(
            hidden_weights=[np.array([[0.5, 0.2], [0.6, -0.6]])],
            hidden_biases=[np.array([0.3, 0.25])],
            hidden_activation=ReLU(),
            output_weights=np.array([[0.8, -0.4], [0.5, 0.4], [0.3, 0.75]]),
            output_bias=np.array([0.6, 0.5, -0.5]),
            output_activation=Softmax(),
        )

        nn = MultiClassClassificationNetwork(config=network_config)
        output_values = nn.forward(np.array([1.5, 0.5]))
        expected_values = np.array([1, 0, 0])
        print('Total Loss:', CCE()(expected_values, output_values))

        output_values = nn.forward(np.array([0, 1]))
        expected_values = np.array([0, 0, 1])
        print('Total Loss:', CCE()(expected_values, output_values))


if __name__ == '__main__':
    NeuralNetworkTaskHandler.exec_task_1_regression()
    NeuralNetworkTaskHandler.exec_task_2_binary_classification()
    NeuralNetworkTaskHandler.exec_task_3_multi_label_classification()
    NeuralNetworkTaskHandler.exec_task_4_multi_class_classification()
