import argparse
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from cnn_model import CNN

# Default constants
LEARNING_RATE_DEFAULT = 1e-3
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 20
EVAL_FREQ_DEFAULT = 1
DATA_DIR_DEFAULT = './data'

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
    p = np.argmax(predictions, axis=1)
    y = np.argmax(targets, axis=1)
    correct = np.sum(p == y)
    return correct / len(y)


def train():
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    learning_rate = LEARNING_RATE_DEFAULT
    max_steps = MAX_EPOCHS_DEFAULT
    batch_size = BATCH_SIZE_DEFAULT
    eval_freq = EVAL_FREQ_DEFAULT
    data_dir = DATA_DIR_DEFAULT
    if FLAGS is not None:
        learning_rate = FLAGS.learning_rate
        max_steps = FLAGS.max_steps
        batch_size = FLAGS.batch_size
        eval_freq = FLAGS.eval_freq
        data_dir = FLAGS.data_dir

    '''
    Data Loading
    '''
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    '''
    The model, the criterion and the optimizer
    '''
    model = CNN(3, 10)  # CIFAR-10
    num_epochs = max_steps
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    '''
    Train
    '''

    loss_history = []
    accuracy_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        if (epoch + 1) % eval_freq == 0 or epoch == num_epochs - 1:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = correct / total
            loss_history.append(epoch_loss)
            accuracy_history.append(val_accuracy)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    '''
    Test
    '''
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.4f}')

    return loss_history, accuracy_history


def main():
    """
    Main function
    """
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()

