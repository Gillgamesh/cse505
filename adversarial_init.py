# Data loading for MNIST: https://www.digitalocean.com/community/tutorials/mnist-dataset-in-python
# https://medium.com/tebs-lab/how-to-classify-mnist-digits-with-different-neural-network-architectures-39c75a0f03e3

import lantern
from lantern import *
from z3 import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch import optim


import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

'''
Very simple 2-D example.
Given an inner box represented by Xs, and outer box represented by Ys. 
The inner box is contained inside the outer box.
Assume everything contained inside the outer box is classified as class 1
And everything else is classified as class 2.
Goal is to find the min norm(Delta X) such that the inner box + Delta X is classified as class 2.  
'''
def simple_find_box_distance(box_dim):

    Xs = const_vector("x", box_dim)
    DXs = const_vector("dx", box_dim)

    x1 = Xs[0]
    x2 = Xs[1]

    # Initialize the solver
    s = Solver()

    # Making the assumption that the inner box is always contained in the outer box
    # Add the constraint of the inner box [-1, 1] x [-1, 1]
    #s.add(And(x1 <= 1, x1 >= -1, x2 <= 1, x2 >= -1))
    s.add(And(x1 == 0, x2 == 0))

    # Construct the norm of the adversarial vector
    dxs_norm_c, norm = get_max_norm(DXs)
    s.add(dxs_norm_c)

    # Add the adversarial addition to the initial vector and treat the sum as the input vector
    adv_c, advs = add_adversarial(Xs, DXs)
    s.add(adv_c)

    # Make the results (which is the added vec/box in this case) of the function bigger than the outer box
    # The outer box is [-3, 3] x [-3, 3]
    s.add(Or([Or(advs[i] <= -3, advs[i] >= 3) for i in range(0, len(advs))]))

    # Constrain the norm with in some alpha.
    # Be careful, either not set the upper bound equal to the box bound
    #    or set the checking condition without an equal sign.
    sol_tup = search_alpha(norm, s, 20)
    print_searched_sol(sol_tup)
    return

'''
The example trying over the very simple NN with W = [1, 1] b = 1, and Relu
The input is [0, 0]^T, and the output is bounded by the range of [-1, 3]
Goal is to find the box which is [-1, 1] x [-1, 1] in the input to be safe 
'''
def simple_NN_ex():
    # Obtain the network and the corresponding constraints.
    lin = nn.Linear(2, 1)
    nn.init.ones_(lin.weight)
    nn.init.ones_(lin.bias)

    net = nn.Sequential(
        lin,
        nn.ReLU())

    print("A PyTorch network:")
    print(net)
    print("Network parameters:")
    print(list(net.parameters()))

    constraints, in_vars, out_vars = lantern.as_z3(net)
    print("Z3 constraints, input variables, output variables (Real-sorted):")
    print(constraints)
    print(in_vars)
    print(out_vars)
    print()

    # Initialize the z3 solvers and corresponding vars
    s = Solver()
    s.add(constraints)

    Xs = const_vector("x", 2)
    DXs = const_vector("dx", 2)
    x1 = Xs[0]
    x2 = Xs[1]
    # Set the initial point to [0, 0]^T
    #s.add(And(x1 == 0, x2 == 0))
    s.add(And(x1 <= 0.5, x1 >= -0.5, x2 <= 0.5, x2 >= -0.5))

    # Construct the norm of the adversarial vector
    dxs_norm_c, norm = get_max_norm(DXs)
    s.add(dxs_norm_c)

    # Add the adversarial addition to the initial vector and treat the sum as the input vector
    adv_c, advs = add_adversarial(Xs, DXs)
    s.add(adv_c)

    # The variables that are the real input of the network.
    s.add(And([in_vars[i] == advs[i] for i in range(0, len(advs))]))

    # Constraint the output situation
    s.add(Or(out_vars[0] >= 3, out_vars[0] < 0))

    # Search for the initial box.
    sol_tup = search_alpha(norm, s, 2)
    print_searched_sol(sol_tup)


'''
Assumptions: Given a lower and upper error such that
1. lower error is always classifying without error (usually set to 0)
2. upper error is always classifying to a different class
3. lo <= up
4. The solver needs to have the constraint indicating that the output of the vector under the adversarial attack need
    to be classified to the other side. For example, if inside the box [-3, 3] x [-3, 3] would be classify as 1 and the 
    other is 2, s.add(Or(Or(x1 <=-3, x1 >=3), Or(x2 <=-3, x2 >=3), ......)) can represent the mis-classification.
norm: the z3 variable respecting the norm of the adversarial vector
s: the z3 solver which has already loaded all constraints except the constraint for the norm 
cp_err: the floating point comparison error. If abs(f1-f2) <= 0.001, then we consider they are equal. This argument
         affects the convergence rate.
'''
def search_alpha(norm, s: Solver, up, lo = 0.0, cp_err = 1e-3):

    # First need to check is lo would not cause mis-classification and up would cause mis-classification

    s.push()
    s.add(And(norm >= 0, norm <= lo))
    check_lo = s.check()
    if check_lo == sat:
        print("lower bound is too large causing a mis-classification. Suggest lo = 0.0")
        exit(1)
    s.pop()
    print("First bound check finish")
    s.push()
    s.add(And(norm >= 0, norm <= up))
    check_up = s.check()
    if check_up != sat:
        print("upper bound is not large enough to cause a mis-classification.")
        exit(1)
    s.pop()
    print("Second bound check finish")

    # Main start of the binary search algorithm
    sol_model = None
    # We need to loop when lower and upper are not equal
    while not math.isclose(lo, up, abs_tol=cp_err):

        # For each round, we need to compute the mid point
        # Also pushing solver to create a new scope
        alpha = (lo + up) / 2
        print("Try out alpha: ", alpha)
        s.push()
        s.add(And(norm >= 0, norm <= alpha))
        check = s.check()
        if check != sat:
            # Meaning with such alpha we are still safe and not having mis-classification
            # We need to push up (increase) the safer bound (the lower one)
            lo = alpha
        else:
            # Meaning with such alpha we are unsafe since we found some mis-classification
            # We need to push down (decrease) the unsafe bound (the upper one)
            # Also we need to store the solution model
            up = alpha
            sol_model = s.model()
        s.pop()
    # After convergence that we found two bounds match,
    # lo --> a max bound of modification over input would remains the same (tolerating the cp_err)
    # up --> a min bound of modification over input would change the output (tolerating the cp_err)
    return (lo, up, sol_model)

def print_searched_sol(tup):
    lo = tup[0]
    up = tup[1]
    sol_model = tup[2]
    print("The max error rate without mis-classification is: ", lo)
    print("The min error rate with mis-classification is: ", up)
    if sol_model is not None:
        print(sol_model)
    else:
        print("There's no solution model, double check if the upper bound is big enough.")
'''
Return the constraint of X + Delta x
'''
def add_adversarial(Xs, DXs):
    constraints = []
    advs = []
    for i in range(0, len(Xs)):
        temp_adv = Real(('adv__' + str(i)))
        advs.append(temp_adv)
        temp_added = (Xs[i] + DXs[i])
        constraint = z3.simplify(temp_adv == temp_added)
        constraints.append(constraint)
    return constraints, advs


def test_norm():
    Xs = const_vector("x", 4)
    x1 = Xs[0]
    x2 = Xs[1]
    x3 = Xs[2]
    x4 = Xs[3]
    s = Solver()
    s.add(And(x1 == -5, x2 == 4, x3 == 20, x4 == -50))
    x_c, norm = get_max_norm(Xs)
    s.add(x_c)
    check = s.check()
    if check != sat:
        print("Unsat!")
        exit(1)  # check if satisfiable
    print(s.model())  # print the solution

'''
The constraints of getting L1 norm of a vector
'''
def get_max_norm(Xs, name = "max_norm_X"):
    def abs(x):
        return z3.If(x >= 0, x, -x)

    # Initialize the z3 norm and the
    max_norm = Real(name)
    constraints = []
    temp_max = abs(Xs[0])
    for i in range(1, len(Xs)):
        temp_max = z3.If(abs(Xs[i]) > temp_max, abs(Xs[i]), temp_max)
    constraint = z3.simplify(max_norm == temp_max)
    constraints.append(constraint)
    return constraints, max_norm

'''
This function only works for 2_D
'''
def get_L2_square_norm(Xs, name = "max_L2_norm_X"):
    print("Constraining a L2 norm! Only works for 2-D cases")
    max_norm = Real(name)
    constraints = [z3.simplify(max_norm == ((Xs[0] * Xs[0]) + (Xs[1] * Xs[1])))]
    return constraints, max_norm

'''
The MNIST section
'''

class MNIST:
    def __init__(self):
        self.load_data_set()
        self.build_network()
        return

    def load_data_set(self):
        # Preprocess the data and perform PCA
        # Following: https://www.kaggle.com/code/jonathankristanto/experimenting-with-pca-on-mnist-dataset
        (train_X, train_y), (test_X, test_y) = mnist.load_data()

        train_X = train_X.reshape(60000, 784)
        test_X = test_X.reshape(10000, 784)

        train_X = train_X.astype('float32')
        test_X = test_X.astype('float32')

        train_y = train_y.astype('float32')
        test_y = test_y.astype('float32')

        sc = StandardScaler()
        train_X = sc.fit_transform(train_X)
        test_X = sc.transform(test_X)

        pca = PCA(n_components=10)
        pca.fit(train_X)
        train_imgs = pca.transform(train_X)
        test_imgs = pca.transform(test_X)

        train_y_1_indices = np.where(np.array(train_y) == 1)
        train_y_7_indices = np.where(np.array(train_y) == 7)
        train_X1 = np.array(train_imgs[train_y_1_indices[0], :])
        train_X7 = np.array(train_imgs[train_y_7_indices[0], :])
        train_Y1 = np.array(train_y[train_y_1_indices[0]])
        train_Y7 = np.array([0 for i in range(0, train_X7.shape[0])])

        train_data = np.concatenate((train_X1, train_X7), axis=0)
        train_label = np.concatenate((train_Y1, train_Y7))

        test_y_1_indices = np.where(np.array(test_y) == 1)
        test_y_7_indices = np.where(np.array(test_y) == 7)
        test_X1 = np.array(test_imgs[test_y_1_indices[0], :])
        test_X7 = np.array(test_imgs[test_y_7_indices[0], :])
        test_Y1 = np.array(test_y[test_y_1_indices[0]])
        test_Y7 = np.array([0 for k in range(0, test_X7.shape[0])])

        test_data = np.concatenate((test_X1, test_X7), axis=0)
        test_label = np.concatenate((test_Y1, test_Y7))

        self.train_imgs = torch.from_numpy(train_data)
        self.train_imgs_label = torch.from_numpy(train_label).reshape(-1, 1)
        self.test_imgs = torch.from_numpy(test_data)
        self.test_imgs_label = torch.from_numpy(test_label).reshape(-1, 1)
        print('train_imgs: ' + str(self.train_imgs.size()))
        print('train_imgs_label: ' + str(self.train_imgs_label.size()))
        print('test_imgs: ' + str(self.test_imgs.size()))
        print('test_imgs_label: ' + str(self.test_imgs_label.size()))


    def build_network(self):
        num_input = self.train_imgs.shape[1]
        middle_dim = 4
        self.net = nn.Sequential(
            nn.Linear(num_input, middle_dim),
            nn.ReLU(),
            #nn.Linear(middle_dim, middle_dim),
            #nn.ReLU(),
            nn.Linear(middle_dim, 1),
            nn.ReLU()
        )
    def train_network(self):
        loss_func = nn.L1Loss()
        learning_rate = 0.003
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        self.train_loss = []
        self.train_accuracy = []
        iters = 500
        for i in range(iters):
            y_hat = self.net(self.train_imgs)
            loss = loss_func(y_hat, self.train_imgs_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            y_hat_class = np.where(y_hat.detach().numpy() < 0.5, 0, 1)
            np_train_imgs_label = self.train_imgs_label.detach().numpy()
            accuracy = np.sum(np_train_imgs_label==y_hat_class) / len(np_train_imgs_label)
            self.train_accuracy.append(accuracy)
            self.train_loss.append(loss.item())

    def plot_training(self):
        fig, ax = plt.subplots(2, 1, figsize=(12,8))
        ax[0].plot(self.train_loss)
        ax[0].set_ylabel('Loss')
        ax[0].set_title('Training Loss')

        ax[1].plot(self.train_accuracy)
        ax[1].set_ylabel('Classification Accuracy')
        ax[1].set_title('Training Accuracy')

        plt.tight_layout()
        plt.show()
        #fig.savefig("./MNIST/Train Results")

    def assess_net(self):
        # Pass test data
        y_hat_test = self.net(self.test_imgs)
        y_hat_test_class = np.where(y_hat_test.detach().numpy() < 0.5, 0, 1)
        np_test_imgs_label = self.test_imgs_label.detach().numpy()
        test_accuracy = np.sum(np_test_imgs_label == y_hat_test_class) / len(np_test_imgs_label)
        print("Test Accuracy {:.2f}".format(test_accuracy))

def network_test(model, test_imgs, test_imgs_label):
    net = model.layers
    y_hat_test = net(test_imgs)
    y_hat_test_class = np.where(y_hat_test.detach().numpy() < 0.5, 0, 1)
    np_test_imgs_label = test_imgs_label.detach().numpy()
    test_accuracy = np.sum(np_test_imgs_label == y_hat_test_class) / len(np_test_imgs_label)
    print("Test Accuracy {:.2f}".format(test_accuracy))

def plot_training_info(train_loss, train_accuracy):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].plot(train_loss)
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training Loss')

    ax[1].plot(train_accuracy)
    ax[1].set_ylabel('Classification Accuracy')
    ax[1].set_title('Training Accuracy')

    plt.tight_layout()
    plt.show()

def MNIST_example(name = "./MNIST2/"):
    # Loading network and do some plots.
    model = torch.load(name + "mnist_model.pt")
    train_imgs = torch.load(name + "train_imgs.pt")
    train_imgs_label = torch.load(name + "train_imgs_label.pt")
    test_imgs = torch.load(name + "test_imgs.pt")
    test_imgs_label = torch.load(name + "test_imgs_label.pt")
    train_accuracy = torch.load(name + "train_accuracy.pt")
    train_loss = torch.load(name + "train_loss.pt")
    network_test(model, test_imgs, test_imgs_label)

    net = model.layers
    ex_img = test_imgs[0]
    y_hat = net(ex_img)
    if (y_hat < 0.5):
        label = 0
        print("The label is: ", label, " representing letter 7")
    else:
        label = 1
        print("The label is: ", label, " representing letter 1")

    # Try to plot the image
    #ex_img = ex_img.detach().numpy().reshape(5, 6)
    #plt.imshow(ex_img, interpolation='nearest')
    #plt.show()
    ex_img = ex_img.detach().numpy()
    length = ex_img.shape[0]
    constraints, in_vars, out_vars = lantern.as_z3(net)
    z3.set_param("parallel.enable", "true");
    print("Z3 constraints, input variables, output variables (Real-sorted):")
    print(constraints)
    print(in_vars)
    print(out_vars)
    print()

    s = Solver()
    s.add(constraints)

    Xs = const_vector("x", length)
    DXs = const_vector("dx", length)

    s.add(And([Xs[i] == ex_img[i] for i in range(0, length)]))

    # Construct the norm of the adversarial vector
    dxs_norm_c, norm = get_max_norm(DXs)
    s.add(dxs_norm_c)

    # Add the adversarial addition to the initial vector and treat the sum as the input vector
    adv_c, advs = add_adversarial(Xs, DXs)
    s.add(adv_c)

    # The variables that are the real input of the network.
    s.add(And([in_vars[i] == advs[i] for i in range(0, len(advs))]))

    # Constraint the output situation
    s.add(out_vars[0] < 0.5)

    # Search for the initial box.
    sol_tup = search_alpha(norm, s, 2)
    print_searched_sol(sol_tup)

    # Plot the network training info
    plot_training_info(train_loss, train_accuracy)


class MNIST_Model(torch.nn.Module):
    def __init__(self, seq: nn.Sequential):
        super().__init__()
        self.layers = seq

    def forward(self, x):
        return self.layers(x)

def simply_train_MNIST():
    mnist = MNIST()
    mnist.train_network()
    mnist.plot_training()
    mnist.assess_net()

def save_MNIST_Model_DATA(name = "./MNIST2/"):
    mnist = MNIST()
    mnist.train_network()
    mnist_model = MNIST_Model(mnist.net)
    torch.save(mnist_model, name + "mnist_model.pt")
    torch.save(mnist.train_imgs, name + "train_imgs.pt")
    torch.save(mnist.train_imgs_label, name + "train_imgs_label.pt")
    torch.save(mnist.test_imgs, name + "test_imgs.pt")
    torch.save(mnist.test_imgs_label, name + "test_imgs_label.pt")
    torch.save(mnist.train_accuracy, name + "train_accuracy.pt")
    torch.save(mnist.train_loss, name + "train_loss.pt")
    mnist.plot_training()
    mnist.assess_net()

if __name__ == '__main__':
    #simple_find_box_distance(2)
    #test_norm()
    #simple_NN_ex()
    #save_MNIST_Model_DATA()
    #MNIST_example()
    #simply_train_MNIST()


    print()
'''
Test float comparison
abs_tol = 1e-4
r1 = 0.1234
r2 = 0.123500000000001
#print(r1 == r2)
print(math.isclose(r1, r2, abs_tol=abs_tol))
'''

'''
test z3 solver pushing and popping
x1 = Int('x1')
x2 = Int('x2')
s = Solver()
s.add(x1 == 1)
s.push()
s.add(x2 <= 2)
s.add(x2 >= -1)
print(s)
s.pop()
print(s)
'''

