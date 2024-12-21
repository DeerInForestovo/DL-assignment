from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from utils import AverageMeter, accuracy


def train(model, data_loader, optimizer, criterion, device, config):
    model.train()
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        optimizer.zero_grad()
        loss.backward()

        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

        optimizer.step()

        acc = accuracy(outputs, batch_targets)
        losses.update(loss.item(), batch_inputs.size(0))
        accuracies.update(acc, batch_inputs.size(0))

        # if step % 1000 == 0:
        #     print(f'[{step}/{len(data_loader)}]', losses, accuracies)
    return losses.avg, accuracies.avg


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, config):
    model.eval()
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)

        acc = accuracy(outputs, batch_targets)
        losses.update(loss.item(), batch_inputs.size(0))
        accuracies.update(acc, batch_inputs.size(0))

        # if step % 1000 == 0:
        #     print(f'[{step}/{len(data_loader)}]', losses, accuracies)
    return losses.avg, accuracies.avg


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    # Initialize the model that we are going to use
    model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes)
    model.to(device)

    # Initialize the dataset and data loader
    dataset = PalindromeDataset(config.input_length, config.data_size)
    # Split dataset into train and validation sets
    train_size = int(config.portion_train * len(dataset))
    # Create data loaders for training and validation
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dloader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Set up the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)
    # scheduler = ...

    val_loss, val_acc = None, None
    for epoch in range(config.max_epoch):
        # Train the model for one epoch
        train_loss, train_acc = train(model, train_dloader, optimizer, criterion, device, config)
        print(f'epoch {epoch}, training loss {train_loss}, training accuracy {train_acc}')
        # Evaluate the trained model on the validation set
        val_loss, val_acc = evaluate(model, val_dloader, criterion, device, config)
        print(f'epoch {epoch}, evaluating loss {val_loss}, evaluating accuracy {val_acc}')

    print('Done training.')
    return val_loss, val_acc


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=19,
                        help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1,
                        help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128,
                        help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='Learning rate')
    parser.add_argument('--max_epoch', type=int,
                        default=1000, help='Number of epochs to run for')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--data_size', type=int,
                        default=1000000, help='Size of the total dataset')
    parser.add_argument('--portion_train', type=float, default=0.8,
                        help='Portion of the total dataset used for training')

    config = parser.parse_args()
    # Train the model
    main(config)

