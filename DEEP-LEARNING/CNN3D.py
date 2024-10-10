import numpy as np

class Conv3D:
    def __init__(self, filters, kernel_size, stride=1, padding=0):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = np.random.randn(filters, kernel_size, kernel_size, kernel_size) * 0.1  # Random weights
        self.b = np.random.randn(filters) * 0.1  # Random biases

    def forward(self, X):
        # Add padding to the input
        X_padded = np.pad(X, ((self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)
        
        d, h, w = X.shape
        out_depth = (d - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_height = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (w - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        output = np.zeros((self.filters, out_depth, out_height, out_width))
        
        for f in range(self.filters):
            for i in range(out_depth):
                for j in range(out_height):
                    for k in range(out_width):
                        d_start = i * self.stride
                        d_end = d_start + self.kernel_size
                        h_start = j * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = k * self.stride
                        w_end = w_start + self.kernel_size
                        output[f, i, j, k] = np.sum(X_padded[d_start:d_end, h_start:h_end, w_start:w_end] * self.W[f]) + self.b[f]
        
        return output

class ReLU3D:
    def forward(self, X):
        return np.maximum(0, X)

class MaxPooling3D:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        n, d, h, w = X.shape
        out_depth = (d - self.pool_size) // self.stride + 1
        out_height = (h - self.pool_size) // self.stride + 1
        out_width = (w - self.pool_size) // self.stride + 1
        
        output = np.zeros((n, out_depth, out_height, out_width))
        
        for i in range(out_depth):
            for j in range(out_height):
                for k in range(out_width):
                    d_start = i * self.stride
                    d_end = d_start + self.pool_size
                    h_start = j * self.stride
                    h_end = h_start + self.pool_size
                    w_start = k * self.stride
                    w_end = w_start + self.pool_size
                    output[:, i, j, k] = np.max(X[:, d_start:d_end, h_start:h_end, w_start:w_end], axis=(1, 2, 3))
        
        return output

# Example usage
input_volume = np.random.rand(1, 8, 8, 8)  # A random 8x8x8 volume
conv3d_layer = Conv3D(filters=2, kernel_size=3, padding=1)
relu3d_layer = ReLU3D()
pooling3d_layer = MaxPooling3D(pool_size=2, stride=2)

conv3d_out = conv3d_layer.forward(input_volume[0])
relu3d_out = relu3d_layer.forward(conv3d_out)
pooled3d_out = pooling3d_layer.forward(relu3d_out)

print("Pooled 3D Output Shape:", pooled3d_out.shape)
