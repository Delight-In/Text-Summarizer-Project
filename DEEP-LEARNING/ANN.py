import numpy as np

# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

class SimpleANN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize weights
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

    def forward(self, X):
        # Forward pass
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden)
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
        
        self.final_input = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        self.final_output = sigmoid(self.final_input)
        
        return self.final_output

    def backward(self, X, y, output):
        # Backward pass
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)

        hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * sigmoid_derivative(self.hidden_layer_output)

        # Update weights
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_layer_delta) * self.learning_rate

    def train(self, X, y, epochs=10000):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        output = self.forward(X)
        return output

# Example usage
if __name__ == "__main__":
    # XOR problem
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    # Create the ANN
    ann = SimpleANN(input_size=2, hidden_size=2, output_size=1)

    # Train the ANN
    ann.train(X, y, epochs=10000)

    # Predictions
    predictions = ann.predict(X)
    print("Predictions after training:")
    print(predictions)
