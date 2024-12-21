import numpy as np


class Perceptron(object):  # Part_1 Task 2

    def __init__(self, n_inputs, max_epochs=100, learning_rate=0.01, save_para_epoch=10):
        """
        Initializes the perceptron object.
        - n_inputs: Number of inputs.
        - max_epochs: Maximum number of training cycles.
        - learning_rate: Magnitude of weight changes at each training cycle.
        - weights: Initialize weights (including bias).
        """
        self.n_inputs = n_inputs  # Fill in: Initialize number of inputs
        self.max_epochs = max_epochs  # Fill in: Initialize maximum number of epochs
        self.learning_rate = learning_rate  # Fill in: Initialize learning rate
        self.weights = np.zeros(self.n_inputs + 1)  # Fill in: Initialize weights with zeros
        # Debug & Plot
        self.save_para_epoch = save_para_epoch
        self.weights_history = []

    def forward(self, input_vec):
        """
        Predicts label from input.
        Args:
            input_vec (np.ndarray): Input array of training data, input vec must be all samples
        Returns:
            int: Predicted label (1 or -1) or Predicted labels.
        """
        return np.sign(np.dot(self.weights, np.insert(input_vec, 0, 1)))

    def train(self, training_inputs, labels):
        """
        Trains the perceptron.
        Args:
            training_inputs (list of np.ndarray): List of numpy arrays of training points.
            labels (np.ndarray): Array of expected output values for the corresponding point in training_inputs.
        """
        self.weights_history = []
        # we need max_epochs to train our model
        for epoch in range(self.max_epochs):
            """
                What we should do in one epoch ? 
                you are required to write code for 
                1.do forward pass
                2.calculate the error
                3.compute parameters' gradient 
                4.Using gradient descent method to update parameters(not Stochastic gradient descent!,
                please follow the algorithm procedure in "perceptron_tutorial.pdf".)
            """
            N = len(labels)
            total_gradient = np.zeros(self.n_inputs + 1)
            for training_input, label in zip(training_inputs, labels):
                # a. Compute the predictions of Perceptron of the whole training set.
                prediction = self.forward(training_input)

                # b. Compute the gradient of the loss function with respect to w:
                if prediction != label:
                    total_gradient += (-1/N) * np.insert(training_input, 0, 1) * label

            # c. Update w
            self.weights = self.weights - self.learning_rate * total_gradient

            if epoch % self.save_para_epoch == 0 or epoch == self.max_epochs - 1:
                self.weights_history.append(self.weights)

