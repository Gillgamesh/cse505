# This example clusters points in concentric circles into two classes.
#
# https://www.datahubbs.com/deep-learning-101-first-neural-network-with-pytorch/
# https://medium.com/mlearning-ai/how-to-create-two-circles-in-sklearn-and-make-predictions-on-it-691a94e64f81

import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas

import z3
from z3 import *
import lantern




def plot_circles(X, Y, pause_for_plot=False):
    df = pandas.DataFrame(dict(x=X[:,0], y=X[:,1], label=Y))
    colors = {0: 'red', 1: 'blue'}
    fig, ax = plt.subplots(figsize=(12,8))
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.title('Circle Data')
    if pause_for_plot:
        plt.show()


def build_net(X, Y):
    # Split into test and training data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=73)

    # Define network dimensions
    n_input_dim = X_train.shape[1]
    # Layer size
    n_hidden = 4 # Number of hidden nodes
    n_output = 1 # Number of output nodes. Use "1" for binary classifier

    # Build your network
    net = nn.Sequential(
        nn.Linear(n_input_dim, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_output),
        nn.ReLU())

    return X_train, X_test, Y_train, Y_test, net


def train_net(X_train, X_test, Y_train, Y_test, net):
    # Set hyperparameters: Loss function, learning rate, optimizer
    loss_func = nn.L1Loss()
    learning_rate = 0.003
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    train_loss = []
    train_accuracy = []
    iters = 500
    Y_train_t = torch.FloatTensor(Y_train).reshape(-1, 1)
    for i in range(iters):
        X_train_t = torch.FloatTensor(X_train)
        y_hat = net(X_train_t)
        loss = loss_func(y_hat, Y_train_t)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        y_hat_class = np.where(y_hat.detach().numpy()<0.5, 0, 1)
        accuracy = np.sum(Y_train.reshape(-1,1)==y_hat_class) / len(Y_train)
        train_accuracy.append(accuracy)
        train_loss.append(loss.item())

    return train_accuracy, train_loss, net

def plot_training(train_accuracy, train_loss, net, pause_for_plot=False):
    fig, ax = plt.subplots(2, 1, figsize=(12,8))
    ax[0].plot(train_loss)
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training Loss')

    ax[1].plot(train_accuracy)
    ax[1].set_ylabel('Classification Accuracy')
    ax[1].set_title('Training Accuracy')

    plt.tight_layout()
    if pause_for_plot:
        plt.show()

def plot_barrier(X, Y):
    # Pass test data
    X_test_t = torch.FloatTensor(X)
    y_hat_test = net(X_test_t)
    y_hat_test_class = np.where(y_hat_test.detach().numpy()<0.5, 0, 1)
    test_accuracy = np.sum(Y.reshape(-1,1)==y_hat_test_class) / len(Y)
    print("Test Accuracy {:.2f}".format(test_accuracy))

    # Plot the decision boundary
    # Determine grid range in x and y directions
    x_min, x_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
    y_min, y_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1

    # Set grid spacing parameter
    spacing = min(x_max - x_min, y_max - y_min) / 100

    # Create grid
    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                np.arange(y_min, y_max, spacing))

    # Concatenate data to match input
    data = np.hstack((XX.ravel().reshape(-1,1),
                    YY.ravel().reshape(-1,1)))

    # Pass data to predict method
    data_t = torch.FloatTensor(data)
    db_prob = net(data_t)

    clf = np.where(db_prob<0.5,0,1)

    Z = clf.reshape(XX.shape)

    plt.figure(figsize=(12,8))
    plt.contourf(XX, YY, Z, cmap=plt.cm.Accent, alpha=0.5)
    plt.scatter(X[:,0], X[:,1], c=Y,
                cmap=plt.cm.Accent)
    plt.show()


if __name__ == '__main__':
    print("Using PyTorch Version %s" %torch.__version__)

    np.random.seed(6)
    torch.manual_seed(0)

    X, Y = make_circles(1000, noise=0.06, factor=0.6)
    plot_circles(X, Y, pause_for_plot=False)

    X_train, X_test, Y_train, Y_test, net = build_net(X, Y)
    print(net)

    train_accuracy, train_loss, net = train_net(X_train, X_test, Y_train, Y_test, net)
    print(net)

    plot_training(train_accuracy, train_loss, net, pause_for_plot=False)

    plot_barrier(X, Y)



    ## Z3

    constraints, in_vars, out_vars = lantern.as_z3(net)
    print("Z3 constraints, input variables, output variables (Real-sorted):")
    #print(constraints)
    print("Inputs:", in_vars)
    print("Outputs:", out_vars)

    in_pt_x = in_vars[0]
    in_pt_y = in_vars[1]
    out_class = out_vars[0]

    # Point in inner ring; should be class 1
    in_c = And(
            (in_pt_x * in_pt_x) + (in_pt_y * in_pt_y) >= 0.16,   # radius >= 0.4
            (in_pt_x * in_pt_x) + (in_pt_y * in_pt_y) <= 0.5625, # radius <= 0.75
    )
    out_c = out_class == 0 # classified as class 0 (outer ring)

    s = Solver()
    s.add(constraints)
    s.add(in_c)
    s.add(out_c)
    sat_check = s.check()
    if sat_check == sat:
        print("Uh oh! Successfully found a counterexample:")

        lin0_in_0 = Real("_lin0_in__0")
        lin0_in_1 = Real("_lin0_in__1")
        relu3_out_0 = Real("_relu3_out__0")

        lin0_in_0_interp = s.model().get_interp(lin0_in_0)
        lin0_in_1_interp = s.model().get_interp(lin0_in_1)
        relu3_out_0_interp = s.model().get_interp(relu3_out_0)

        counterexample_coords = [ eval(str(lin0_in_0_interp)),  eval(str(lin0_in_1_interp))]
        counterexample_class = relu3_out_0_interp

        print("Counterexample: ({}, {}) -> {}".format(counterexample_coords[0], counterexample_coords[1], counterexample_class))

        # Confirm that the counter-example is indeed mapped to the wrong class.
        counterexample_array = np.ndarray(shape=(1,2), dtype=float, buffer=np.array(counterexample_coords))
        counterexample_input_tensor = torch.FloatTensor(counterexample_array)
        counterexample_output_tensor = net(counterexample_input_tensor)
        counterexample_output_class = np.where(counterexample_output_tensor.detach().numpy()<0.5, 0, 1)
        print(counterexample_output_class)

    else:
        print("Hooray! Failed to find a counterexample.")