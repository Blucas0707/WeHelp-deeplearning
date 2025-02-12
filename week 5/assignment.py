from abc import ABC, abstractmethod
import numpy as np


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
        return np.mean((expected_values - output_values) ** 2)

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
                np.dot(self.hidden_weights[idx], self.hidden_layer_outputs[-1])
                + self.hidden_biases[idx]
            )
            self.hidden_layer_inputs.append(hidden_layer_input)
            self.hidden_layer_outputs.append(
                self.hidden_activations[idx](hidden_layer_input)
            )

        self.output_layer_input = (
            np.dot(self.output_weight, self.hidden_layer_outputs[-1]) + self.output_bias
        )

        self.output_layer_output = self.output_activation(self.output_layer_input)
        return self.output_layer_output

    def backward(self, output_losses: np.ndarray):
        # Output layer
        output_layer_delta = output_losses * self.output_activation.get_deviation(
            self.output_layer_input
        )
        self.output_layer_weight_gradient = np.dot(
            output_layer_delta, self.hidden_layer_outputs[-1].T
        )
        self.output_bias_gradient = output_layer_delta

        # Hidden layer
        self.hidden_layer_weight_gradients = []
        self.hidden_layer_bias_gradients = []

        hidden_layer_delta = np.dot(self.output_weight.T, output_layer_delta)
        for i in reversed(range(len(self.hidden_weights))):
            hidden_layer_delta = hidden_layer_delta * self.hidden_activations[
                i
            ].get_deviation(self.hidden_layer_inputs[i])
            self.hidden_layer_weight_gradients.append(
                np.dot(hidden_layer_delta, self.hidden_layer_outputs[i].T)
            )
            self.hidden_layer_bias_gradients.append(hidden_layer_delta)

            if i > 0:
                hidden_layer_delta = np.dot(
                    self.hidden_weights[i].T, hidden_layer_delta
                )
        self.hidden_layer_weight_gradients = self.hidden_layer_weight_gradients[::-1]
        self.hidden_layer_bias_gradients = self.hidden_layer_bias_gradients[::-1]

    def zero_grad(self, learning_rate: float):
        for i in range(len(self.hidden_weights)):
            self.hidden_weights[i] -= (
                learning_rate * self.hidden_layer_weight_gradients[i]
            )
            self.hidden_biases[i] -= learning_rate * self.hidden_layer_bias_gradients[i]

        self.output_weight -= learning_rate * self.output_layer_weight_gradient
        self.output_bias -= learning_rate * self.output_bias_gradient


class NeuralNetworkTaskHandler:
    @staticmethod
    def exec_task_1_regression(epoch: int = 1):
        nn = NeuralNetwork(
            hidden_weights=[
                np.array([[0.5, 0.2], [0.6, -0.6]]),
                np.array([0.8, -0.5]).reshape(1, 2),
            ],
            hidden_biases=[
                np.array([[0.3], [0.25]]),
                np.array([0.6]).reshape(1, 1),
            ],
            hidden_activations=[ReLU(), Linear()],
            output_weight=np.array([[0.6], [-0.3]]),
            output_bias=np.array([[0.4], [0.75]]),
            output_activation=Linear(),
        )
        expected_values = np.array([[0.8], [1]])
        loss_function = MSE()
        learning_rate = 0.01

        for _ in range(epoch):
            output_values = nn.forward(np.array([[1.5], [0.5]]))
            loss = loss_function.get_total_loss(output_values, expected_values)
            output_losses = loss_function.get_output_losses(
                output_values, expected_values
            )

            nn.backward(output_losses)
            nn.zero_grad(learning_rate)

        print(f"%%%%% Epoch = {epoch} %%%%%%\n")

        if epoch == 1:
            print(f"{nn.hidden_weights=}\n{nn.output_weight=}\n")
        else:
            loss = loss_function.get_total_loss(output_values, expected_values)
            print(f"{loss=}\n")

    @staticmethod
    def exec_task_2_binaryclassification(epoch: int = 1):
        nn = NeuralNetwork(
            hidden_weights=[
                np.array([[0.5, 0.2], [0.6, -0.6]]),
            ],
            hidden_biases=[np.array([[0.3], [0.25]])],
            hidden_activations=[ReLU()],
            output_weight=np.array([0.8, 0.4]).reshape(1, 2),
            output_bias=np.array([-0.5]).reshape(1, 1),
            output_activation=Sigmoid(),
        )
        expected_values = np.array([[1]])
        loss_function = BCE()
        learning_rate = 0.1

        for i in range(epoch):
            output_values = nn.forward(np.array([[0.75], [1.25]]))
            if i == 0:
                loss = loss_function.get_total_loss(output_values, expected_values)

            output_losses = loss_function.get_output_losses(
                output_values, expected_values
            )
            nn.backward(output_losses)
            nn.zero_grad(learning_rate)

        print(f"%%%%% Epoch = {epoch} %%%%%%\n")

        if epoch == 1:
            print(f"{nn.hidden_weights=}\n{nn.output_weight=}\n")
        else:
            loss = loss_function.get_total_loss(output_values, expected_values)
            print(f"{loss=}\n")


if __name__ == "__main__":
    print("----- Task 1: Regression -----")
    NeuralNetworkTaskHandler.exec_task_1_regression()
    NeuralNetworkTaskHandler.exec_task_1_regression(1000)
    print("\n----- Task 2: Binary Classification -----")
    NeuralNetworkTaskHandler.exec_task_2_binaryclassification()
    NeuralNetworkTaskHandler.exec_task_2_binaryclassification(1000)
