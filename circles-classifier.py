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

print("Using PyTorch Version %s" %torch.__version__)

np.random.seed(6)
torch.manual_seed(0)

X, Y = make_circles(1000, noise=0.06, factor=0.6)


# Split into test and training data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
    test_size=0.25, random_state=73)

# Plot
df = pandas.DataFrame(dict(x=X[:,0], y=X[:,1], label=Y))
colors = {0: 'red', 1: 'blue'}
fig, ax = plt.subplots(figsize=(12,8))
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.title('Circle Data')
#plt.show()

#### Build the neural network

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

print(net)



## Training?
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

fig, ax = plt.subplots(2, 1, figsize=(12,8))
ax[0].plot(train_loss)
ax[0].set_ylabel('Loss')
ax[0].set_title('Training Loss')

ax[1].plot(train_accuracy)
ax[1].set_ylabel('Classification Accuracy')
ax[1].set_title('Training Accuracy')

plt.tight_layout()
#plt.show()

## Plot our network's partitioning barrier

# Pass test data
X_test_t = torch.FloatTensor(X_test)
y_hat_test = net(X_test_t)
y_hat_test_class = np.where(y_hat_test.detach().numpy()<0.5, 0, 1)
test_accuracy = np.sum(Y_test.reshape(-1,1)==y_hat_test_class) / len(Y_test)
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
plt.scatter(X_test[:,0], X_test[:,1], c=Y_test,
            cmap=plt.cm.Accent)
plt.show()


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

    counter_example_coords = [ eval(str(lin0_in_0_interp)),  eval(str(lin0_in_1_interp))]
    counter_example_class = relu3_out_0_interp

    print("Counter-example: ({}, {}) -> {}".format(counter_example_coords[0], counter_example_coords[1], counter_example_class))

    # Confirm that the counter-example is indeed mapped to the wrong class.
    counter_example_array = np.ndarray(shape=(1,2), dtype=float, buffer=np.array(counter_example_coords))
    counter_example_input_tensor = torch.FloatTensor(counter_example_array)
    counter_example_output_tensor = net(counter_example_input_tensor)
    counter_example_output_class = np.where(counter_example_output_tensor.detach().numpy()<0.5, 0, 1)
    print(counter_example_output_class)

else:
    print("Could not find a counterexample.")