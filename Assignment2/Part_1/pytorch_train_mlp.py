import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from pytorch_mlp import MLP
from torch.utils.data import DataLoader

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    pred_labels = predictions.argmax(dim=1)
    correct = (pred_labels == targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy


def train(model, train_dataset, test_dataset, learning_rate, max_steps, eval_freq):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should evaluate the model on the whole test set each eval_freq iterations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    accuracy_history = []
    loss_history = []

    for epoch in range(max_steps):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % eval_freq == 0 or epoch == max_steps - 1:
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += criterion(output, target).item()
                    correct += (output.argmax(dim=1) == target).sum().item()

            test_loss /= len(test_loader.dataset)
            accuracy_score = correct / len(test_loader.dataset)
            accuracy_history.append(float(accuracy_score))
            loss_history.append(float(test_loss))
            print(f"Epoch {epoch}, Test loss: {test_loss:.4f}, Accuracy: {accuracy_score:.4f}")

    return accuracy_history, loss_history


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()

    # make_moon setting
    n_hidden = list(map(int, FLAGS.dnn_hidden_units.split(',')))
    model = MLP(2, n_hidden, 2)
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                   torch.tensor(y_train, dtype=torch.long))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                                  torch.tensor(y_test, dtype=torch.long))
    train(model, train_dataset, test_dataset, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq)

