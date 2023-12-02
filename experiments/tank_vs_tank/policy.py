import numpy as np
import torch

# torch.manual_seed(0)
# np.random.seed(0)


class Policy(torch.nn.Module):
    """
    A basic classifier neural network with 1 hidden layer.
    """

    def __init__(
        self,
        nb_of_inputs: int = 2,
        nb_of_outputs: int = 4,
        nb_of_hidden_neurons: int = 5,
    ):
        super(Policy, self).__init__()
        self.nb_of_inputs = nb_of_inputs
        self.nb_of_outputs = nb_of_outputs
        self.nb_of_hidden_neurons = nb_of_hidden_neurons

        self.fc1 = torch.nn.Linear(self.nb_of_inputs, self.nb_of_hidden_neurons)
        self.fc2 = torch.nn.Linear(self.nb_of_hidden_neurons, self.nb_of_outputs)

        self.nb_of_param = self._get_nb_of_params()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(x))
        # x = torch.tanh(self.fc2(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        x = torch.argmax(x)
        return x

    def set_weight(self, weights: list):
        """
        Given a flat weight vector, set the weights of the neural network.
        """
        assert len(weights) == (self.nb_of_inputs * self.nb_of_hidden_neurons) + (
            self.nb_of_hidden_neurons * self.nb_of_outputs
        )
        self.fc1.weight.data = torch.Tensor(
            np.array(weights[: self.nb_of_inputs * self.nb_of_hidden_neurons]).reshape(
                self.nb_of_hidden_neurons, self.nb_of_inputs
            )
        )
        self.fc2.weight.data = torch.Tensor(
            np.array(weights[self.nb_of_inputs * self.nb_of_hidden_neurons :]).reshape(
                self.nb_of_outputs, self.nb_of_hidden_neurons
            )
        )

    def _get_nb_of_params(self) -> list:
        """
        Return the weights of the neural network as a flat vector.
        """
        # for param in self.named_parameters():
        total_nb_of_params = 0
        for param in self.named_parameters():
            try:
                total_nb_of_params += param[1].shape[0] * param[1].shape[1]
            except:
                total_nb_of_params += param[1].shape[0]

        return total_nb_of_params

    def set_weight(self, weights: list):
        """
        Get weights from outside and set them to the neural network.
        """
        assert len(weights) == self.nb_of_param
        start_idx = 0
        for param in self.named_parameters():
            try:
                end_idx = start_idx + param[1].shape[0] * param[1].shape[1]
                param[1].data = torch.Tensor(weights[start_idx:end_idx]).reshape(
                    param[1].shape
                )
                start_idx = end_idx
            except:
                end_idx = start_idx + param[1].shape[0]
                param[1].data = torch.Tensor(weights[start_idx:end_idx]).reshape(
                    param[1].shape
                )
                start_idx = end_idx

    def _str__(self) -> str:
        return (
            "Policy: nb_of_inputs={}, nb_of_outputs={}, nb_of_hidden_neurons={}".format(
                self.nb_of_inputs, self.nb_of_outputs, self.nb_of_hidden_neurons
            )
        )


def main():
    demo_policy = Policy(nb_of_inputs=2, nb_of_outputs=4, nb_of_hidden_neurons=5)

    output = demo_policy.forward(torch.Tensor([1, 2]))
    print(output)
    print(demo_policy.nb_of_param)

    demo_policy.set_weight(np.random.rand(demo_policy.nb_of_param))
    # demo_policy.set_weight([1]*demo_policy.nb_of_param)

    output = demo_policy.forward(torch.Tensor([1, 2]))
    print(output.numpy())


if __name__ == "__main__":
    main()
