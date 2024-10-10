import numpy as np

class Conv2D:
    def __init__(self, filters, kernel_size, stride=1, padding=0):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = np.random.randn(filters, kernel_size, kernel_size) * 0.1  # Random weights
        self.b = np.random.randn(filters) * 0.1  # Random biases

    def forward(self, X):
        # Add padding to the input
        X_padded = np.pad(X, ((self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)
        
        h, w = X.shape
        out_height = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (w - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        output = np.zeros((self.filters, out_height, out_width))
        
        for f in range(self.filters):
            for i in range(out_height):
                for j in range(out_width):
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    w_start = j * self.stride
                    w_end = w_start + self.kernel_size
                    output[f, i, j] = np.sum(X_padded[h_start:h_end, w_start:w_end] * self.W[f]) + self.b[f]
        
        return output

class ReLU:
    def forward(self, X):
        return np.maximum(0, X)

class MaxPooling2D:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        n, h, w = X.shape
        out_height = (h - self.pool_size) // self.stride + 1
        out_width = (w - self.pool_size) // self.stride + 1
        
        output = np.zeros((n, out_height, out_width))
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                output[:, i, j] = np.max(X[:, h_start:h_end, w_start:w_end], axis=(1, 2))
        
        return output

# Example usage
input_image = np.random.rand(1, 8, 8)  # A random 8x8 image
conv_layer = Conv2D(filters=2, kernel_size=3, padding=1)
relu_layer = ReLU()
pooling_layer = MaxPooling2D(pool_size=2, stride=2)

conv_out = conv_layer.forward(input_image[0])
relu_out = relu_layer.forward(conv_out)
pooled_out = pooling_layer.forward(relu_out)

print("Pooled Output Shape:", pooled_out.shape)
