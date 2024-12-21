import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from mlp_numpy import MLP
from modules import CrossEntropy
from sklearn.datasets import make_moons


# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-3
MAX_EPOCHS_DEFAULT = 1500  # adjust if you use batch or not
EVAL_FREQ_DEFAULT = 10


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.
    
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding
    
    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """
    p = np.argmax(predictions, axis=1)
    y = np.argmax(targets, axis=1)
    correct = np.sum(p == y)
    return (correct / len(y)) * 100


def train(dnn_hidden_units, learning_rate, max_steps, eval_freq, sgd_batch_size=0, fig_title=True):
    """
    Performs training and evaluation of MLP model.
    
    Args:
        sgd_batch_size: SGD batch size, set to 0 means use batch gradient descent
        dnn_hidden_units: Comma separated list of number of units in each hidden layer
        learning_rate: Learning rate for optimization
        max_steps: Number of epochs to run trainer
        eval_freq: Frequency of evaluation on the test set
        Add necessary arguments such as the data, your model...
    """

    # Process parameters
    n_samples = 1000
    test_size = 0.2
    dnn_hidden_units = list(map(int, dnn_hidden_units.split(',')))

    # Load your data here
    X, y = make_moons(n_samples=n_samples)

    # One-hot encoding
    y_one_hot = np.zeros((y.size, y.max() + 1))
    y_one_hot[np.arange(y.size), y] = 1
    y = y_one_hot

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Initialize your MLP model and loss function (CrossEntropy) here
    model = MLP(2, dnn_hidden_units, 2)
    loss = CrossEntropy()

    # Draw the loss/acc curve
    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []

    for step in range(max_steps):
        # Train
        if sgd_batch_size:  # SGD
            N = X_train.shape[0]
            indices = np.random.permutation(N)
            for i in range(0, N, sgd_batch_size):
                X_batch, y_batch = (X_train[indices[i: min(i + sgd_batch_size, N)]],
                                    y_train[indices[i: min(i + sgd_batch_size, N)]])

                # 1. Forward pass
                train_pred = model.forward(X_batch)

                # 2. Compute loss
                train_loss = loss.forward(train_pred, y_batch)

                # 3. Backward pass
                loss_grad = loss.backward(train_pred, y_batch)
                model.backward(loss_grad)

                # 4. Update weights
                for layer in model.layers:
                    if hasattr(layer, 'params'):
                        layer.params['weight'] -= learning_rate * layer.grads['weight']
                        layer.params['bias'] -= learning_rate * layer.grads['bias']

        else:  # batch gradient descent
            # Implement the training loop
            # 1. Forward pass
            train_pred = model.forward(X_train)

            # 2. Compute loss
            train_loss = loss.forward(train_pred, y_train)

            # 3. Backward pass (compute gradients)
            loss_grad = loss.backward(train_pred, y_train)
            model.backward(loss_grad)

            # 4. Update weights
            for layer in model.layers:
                if hasattr(layer, 'params'):
                    layer.params['weight'] -= learning_rate * layer.grads['weight']
                    layer.params['bias'] -= learning_rate * layer.grads['bias']

        # Test
        if step % eval_freq == 0 or step == max_steps - 1:
            # Forward pass on the test set, compute loss and accuracy
            train_pred = model.forward(X_train)
            test_pred = model.forward(X_test)
            train_loss = loss.forward(train_pred, y_train)
            test_loss = loss.forward(test_pred, y_test)
            train_acc = accuracy(train_pred, y_train)
            test_acc = accuracy(test_pred, y_test)

            # Output results
            print(f"Step: {step}, "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Test Loss: {test_loss:.4f}, "
                  f"Train Accuracy: {train_acc:.2f}%, "
                  f"Test Accuracy: {test_acc:.2f}%")

            # Save results
            train_loss_history.append(train_loss)
            test_loss_history.append(test_loss)
            train_acc_history.append(train_acc)
            test_acc_history.append(test_acc)

    print("Training complete!")

    plt.figure(figsize=(8, 6))
    if fig_title:
        plt.title("Loss and Accuracy over Time")
    plt.xlabel("Steps")
    test_step = [i for i in range(0, max_steps, eval_freq)]
    if test_step[-1] != max_steps - 1:
        test_step.append(max_steps - 1)
    # Loss
    plt.plot(test_step, train_loss_history, label="Train Loss", color='b')
    plt.plot(test_step, test_loss_history, label="Test Loss", color='g')
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    # Accuracy
    ax2 = plt.gca().twinx()  # Get the current axis and create a twin on the right
    ax2.plot(test_step, train_acc_history, label="Train Accuracy", color='r', linestyle='--')
    ax2.plot(test_step, test_acc_history, label="Test Accuracy", color='orange', linestyle='--')
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(loc="upper right")
    plt.show()


def main():
    """
    Main function.
    """
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--sgd_batch_size', type=int, default=0,
                        help='SGD batch size')
    FLAGS = parser.parse_known_args()[0]
    
    train(FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq, FLAGS.sgd_batch_size)


if __name__ == '__main__':
    main()

