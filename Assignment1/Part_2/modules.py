import numpy as np


class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Initializes a linear (fully connected) layer.
        - Weights should be initialized to small random values (e.g., using a normal distribution).
        - Biases should be initialized to zeros.
        Formula: output = x * weight + bias
        """
        # Initialize weights and biases with the correct shapes.
        self.params = {
            'weight': np.random.randn(in_features, out_features) * 0.01,
            'bias': np.zeros(out_features)
        }
        self.grads = {
            'weight': np.zeros_like(self.params['weight']),
            'bias': np.zeros_like(self.params['bias'])
        }
        self.x = None  # Save the input during the forward and use it during the backward

    def forward(self, x):
        """
        Performs the forward pass using the formula: output = xW + b
        """
        self.x = x  # Save the input
        return np.dot(x, self.params['weight']) + self.params['bias']

    def backward(self, dout):
        """
        Backward pass to calculate gradients of loss w.r.t. weights and inputs.
        """
        self.grads['weight'] = np.dot(self.x.T, dout)  # Use the saved input here
        self.grads['bias'] = np.sum(dout, axis=0)
        dx = np.dot(dout, self.params['weight'].T)
        return dx


class ReLU(object):
    def __init__(self):
        self.x = None

    def forward(self, x):
        """
        Applies the ReLU activation function element-wise to the input.
        Formula: output = max(0, x)
        """
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        """
        Computes the gradient of the ReLU function.
        Hint: Gradient is 1 for x > 0, otherwise 0.
        """
        dx = dout * (self.x > 0)
        return dx


class SoftMax(object):
    def forward(self, x):
        """
        Applies the softmax function to the input to obtain output probabilities.
        Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
        """
        # Max trick: minus max{x_i} from all exponents
        exp_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    def backward(self, dout):  # Combined with CE
        """
        The backward pass for softmax is often directly integrated with CrossEntropy for simplicity.
        """
        return dout


class CrossEntropy(object):
    def forward(self, x, y):
        """
        Computes the CrossEntropy loss between predictions and true labels.
        Formula: L = -sum(y_i * log(p_i)), where p is the softmax probability of the correct class y.
        """
        loss = -np.sum(y * np.log(x + 1e-10)) / x.shape[0]  # Avoid log(0) issue
        return loss

    def backward(self, x, y):  # Combined with SoftMax
        """
        Computes the gradient of CrossEntropy loss with respect to the input.
        Hint: For softmax output followed by cross-entropy loss, the gradient simplifies to: p - y.
        """
        return (x - y) / x.shape[0]

