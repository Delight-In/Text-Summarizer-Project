import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Hidden to output
        self.bh = np.zeros((hidden_size, 1))  # Hidden bias
        self.by = np.zeros((output_size, 1))  # Output bias
        self.hidden_state = np.zeros((hidden_size, 1))  # Initial hidden state

    def forward(self, inputs):
        # Inputs is expected to be of shape (input_size, sequence_length)
        h = self.hidden_state  # Initialize hidden state
        outputs = []

        for x in inputs.T:  # Iterate over time steps
            x = x.reshape(-1, 1)  # Reshape to column vector
            h = self._tanh(self.Wxh @ x + self.Whh @ h + self.bh)  # Update hidden state
            y = self.Why @ h + self.by  # Output
            outputs.append(y)

        return np.hstack(outputs), h  # Return outputs and last hidden state

    def _tanh(self, x):
        return np.tanh(x)

# Example usage
input_size = 3    # Size of the input
hidden_size = 5   # Size of the hidden layer
output_size = 2   # Size of the output

rnn = RNN(input_size, hidden_size, output_size)

# Create a random input sequence with shape (input_size, sequence_length)
inputs = np.random.rand(input_size, 4)  # Sequence length of 4

outputs, last_hidden_state = rnn.forward(inputs)

print("Outputs shape:", outputs.shape)  # Should be (output_size, sequence_length)
print("Last hidden state shape:", last_hidden_state.shape)  # Should be (hidden_size, 1)
