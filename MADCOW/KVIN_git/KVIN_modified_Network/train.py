import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from dataset.dataset import *
from utility.utils import *
from model import *




def train(net, trainloader, config, criterion, optimizer,testloader):
    print_header()
    for epoch in range(config.epochs):  # Loop over dataset multiple times
        avg_error, avg_loss, num_batches = 0.0, 0.0, 0.0
        start_time = time.time()
        net = net.cuda()
        #test(net, testloader, config)
        for i, data in enumerate(trainloader):  # Loop over batches of data
            # Get input batch
            X, S1, S2, gamma, labels = data
            if X.size()[0] != config.batch_size:
                continue  # Drop those data, if not enough for a batch
            #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            X = X.cuda()
            S1 = S1.cuda()
            S2 = S2.cuda()
            gamma = gamma.cuda()
            labels = labels.cuda()

            #S1=S1.view(-1,1)
            #S2 = S2.view(-1, 1)


            # Wrap to autograd.Variable
            X, S1 = Variable(X), Variable(S1)
            S2, labels = Variable(S2), Variable(labels)
            gamma = Variable(gamma)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = net(X, S1, S2, gamma, config)

            # Loss
            loss = criterion(outputs, labels.squeeze(1))
            #print(loss)
            #print(outputs)
            # Backward pass
            loss.backward()
            #print("Now Backward Propagation : "+str(i))
            # Update params
            optimizer.step()
            # Calculate Loss and Error
            loss_batch, error_batch = get_stats(loss, outputs, labels)
            avg_loss += loss_batch
            avg_error += error_batch
            #print(loss)
            #print(error_batch)
            num_batches += 1
        time_duration = time.time() - start_time
        # test(net, testloader, config)
        # Print epoch logs
        print_stats(epoch, avg_loss, avg_error, num_batches, time_duration)
    print('\nFinished training. \n')


def test(net, testloader, config):
    total, correct = 0.0, 0.0
    for i, data in enumerate(testloader):
        if i==1000:
            break
        # Get inputs
        X, S1, S2, gamma, labels = data
        if X.size()[0] != config.batch_size:
            continue  # Drop those data, if not enough for a batch
        # automaticlly select device, device agnostic
        X = X.cuda()
        S1 = S1.cuda()
        S2 = S2.cuda()
        gamma = gamma.cuda()
        labels = labels.cuda()
        net = net.cuda()
        # Wrap to autograd.Variable
        X, S1, S2, gamma = Variable(X), Variable(S1), Variable(S2), Variable(gamma)
        # Forward pass
        outputs = net(X, S1, S2, gamma, config)
        # Select actions with max scores(logits)
        # Unwrap autograd.Variable to Tensor

        # Compute test accuracy
        correct += abs(outputs.cpu().data.numpy() - labels.cpu().data.squeeze(1).numpy()).sum()
        total += labels.size()[0]
    print('Test Accuracy: {:.2f}%'.format(100 * (correct / total)))


if __name__ == '__main__':
    # Automatic swith of GPU mode if available
    # use_GPU = torch.cuda.is_available()
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datafile',
        type=str,
        default='dataset/gridworld_16x16.npz',
        help='Path to data file')
    parser.add_argument('--imsize', type=int, default=16, help='Size of image')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.005,
        help='Learning rate, [0.01, 0.005, 0.002, 0.001]')
    parser.add_argument(
        '--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument(
        '--k', type=int, default=20, help='Number of Value Iterations')
    parser.add_argument(
        '--l_i', type=int, default=2, help='Number of channels in input layer')
    parser.add_argument(
        '--l_h',
        type=int,
        default=150,
        help='Number of channels in first hidden layer')
    parser.add_argument(
        '--l_q',
        type=int,
        default=3,
        help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument(
        '--batch_size', type=int, default=128, help='Batch size')
    config = parser.parse_args()
    # Get path to save trained model
    save_path = "trained/vin_{0}x{0}.pth".format(config.imsize)
    # Instantiate a VIN model
    net = VIN(config)
    # Loss
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    # Optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=config.lr, eps=1e-6)
    # Dataset transformer: torchvision.transforms
    transform = None
    # Define Dataset
    trainset = GridworldData(
        config.datafile, imsize=config.imsize, train=True, transform=transform)
    testset = GridworldData(
        config.datafile,
        imsize=config.imsize,
        train=False,
        transform=transform)
    # Create Dataloader
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    # Train the model
    train(net, trainloader, config, criterion, optimizer,testloader)
    # Test accuracy
    test(net, testloader, config)
    # Save the trained model parameters
    torch.save(net.state_dict(), save_path)
