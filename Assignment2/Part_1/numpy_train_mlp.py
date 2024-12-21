import numpy as np
from modules import CrossEntropy

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-3
MAX_EPOCHS_DEFAULT = 1500  # adjust if you use batch or not
EVAL_FREQ_DEFAULT = 10


def accuracy(predictions, targets):
    p = np.argmax(predictions, axis=1)
    y = np.argmax(targets, axis=1)
    correct = np.sum(p == y)
    return correct / len(y)


def train(model, train_dataset, test_dataset, learning_rate, max_steps, eval_freq):
    sgd_batch_size = 16
    loss = CrossEntropy()
    test_loss_history = []
    test_acc_history = []

    X_train = train_dataset.tensors[0].numpy()
    y_train = train_dataset.tensors[1].numpy()
    y_one_hot = np.zeros((y_train.size, y_train.max() + 1))
    y_one_hot[np.arange(y_train.size), y_train] = 1
    y_train = y_one_hot
    X_test = test_dataset.tensors[0].numpy()
    y_test = test_dataset.tensors[1].numpy()
    y_one_hot = np.zeros((y_test.size, y_test.max() + 1))
    y_one_hot[np.arange(y_test.size), y_test] = 1
    y_test = y_one_hot

    for step in range(max_steps):
        # Train
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

        # Test
        if (step + 1) % eval_freq == 0 or step == max_steps - 1:
            # Forward pass on the test set, compute loss and accuracy
            test_pred = model.forward(X_test)
            test_loss = loss.forward(test_pred, y_test)
            test_loss /= len(y_test)
            test_acc = accuracy(test_pred, y_test)

            # Output results
            print(f"Epoch {step}, Test loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

            # Save results
            test_loss_history.append(test_loss)
            test_acc_history.append(test_acc)

    return test_acc_history, test_loss_history

