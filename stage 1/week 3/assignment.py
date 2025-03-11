from dataclasses import dataclass
import numpy as np


@dataclass
class NetworkConfig:
    hidden_weights: list[np.ndarray]
    hidden_biases: list[np.ndarray]
    output_weights: np.ndarray
    output_bias: np.ndarray


class Network:
    def __init__(self, inputs: np.ndarray, config: NetworkConfig):
        self.inputs = inputs
        self.config = config

        self._validate_config()

    def _validate_config(self):
        if len(self.config.hidden_biases) != len(self.config.hidden_weights):
            raise ValueError(
                "Number of hidden biases must match number of hidden weights"
            )

    def forword(self):
        hidden_layer = self.inputs
        for idx, hidden_weight in enumerate(self.config.hidden_weights):
            hidden_layer = (
                np.dot(hidden_weight, hidden_layer) + self.config.hidden_biases[idx]
            )

        return (
            np.dot(self.config.output_weights, hidden_layer) + self.config.output_bias
        )


class NeuralNetworkTaskHandler:
    @staticmethod
    def exec_network_1():
        network_config = NetworkConfig(
            hidden_weights=[np.array([[0.5, 0.2], [0.6, -0.6]])],
            hidden_biases=[np.array([0.3, 0.25])],
            output_weights=np.array([0.8, 0.4]),
            output_bias=-0.5,
        )
        network_1 = Network(inputs=np.array([1.5, 0.5]), config=network_config)
        print(f"{network_1.forword():.2f}")

        network_2 = Network(inputs=np.array([0, 1]), config=network_config)
        print(network_2.forword())

    @staticmethod
    def exec_network_2():
        network_config = NetworkConfig(
            hidden_weights=[
                np.array([[0.5, 1.5], [0.6, -0.8]]),
                np.array([[0.6, -0.8], [0, 0]]),
            ],
            hidden_biases=[
                np.array([0.3, 1.25]),
                np.array([0.3, 0]),
            ],
            output_weights=np.array([[0.5, 0], [-0.4, 0]]),
            output_bias=np.array([0.2, 0.5]),
        )
        network_1 = Network(inputs=np.array([0.75, 1.25]), config=network_config)
        print(network_1.forword())

        network_2 = Network(inputs=np.array([-1, 0.5]), config=network_config)
        print(network_2.forword())


if __name__ == "__main__":
    NeuralNetworkTaskHandler.exec_network_1()
    NeuralNetworkTaskHandler.exec_network_2()
