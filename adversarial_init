import lantern
from lantern import *
from z3 import *
import math
import torch
import torch.nn as nn

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
    s.add(And(x1 <= 1, x1 >= -1, x2 <= 1, x2 >= -1))
    #s.add(And(x1 == 0, x2 == 0))

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
    s.push()
    s.add(And(norm >= 0, norm <= up))
    check_up = s.check()
    if check_up != sat:
        print("upper bound is not large enough to cause a mis-classification.")
        exit(1)
    s.pop()

    # Main start of the binary search algorithm
    sol_model = None
    # We need to loop when lower and upper are not equal
    while not math.isclose(lo, up, abs_tol=cp_err):
        # For each round, we need to compute the mid point
        # Also pushing solver to create a new scope
        alpha = (lo + up) / 2
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


if __name__ == '__main__':
    #simple_find_box_distance(2)
    #test_norm()
    simple_NN_ex()
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

