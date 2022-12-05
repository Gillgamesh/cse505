import lantern
import torch
from man_test import BasicSequentialModel
import z3

if __name__ == "__main__":
    model = BasicSequentialModel()
    model.load_state_dict(torch.load("network.tm"))
    constraints, in_vars, out_vars = lantern.as_z3(model.layers)
    print(in_vars)
    # add square constraints on input
    constraints.append(in_vars[0] < 1)
    constraints.append(in_vars[0] > -1)
    constraints.append(in_vars[1] < 1)
    constraints.append(in_vars[1] > -1)

    # add circle constraints on output
    constraints.append(out_vars[0]**2 + out_vars[1]**2 > 1)
    # this SHOULD give no solution in a proper thing
    
    print(len(constraints))
        
    z3.set_param("parallel.enable", "true");
    z3.solve(constraints)

