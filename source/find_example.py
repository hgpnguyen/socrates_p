import json
import numpy as np
import ast
from test import newApproach, save_tf_model, add_assertion, add_solver
from json_parser import parse, parse_solver
from run_refinepoly import refinepoly_run

INPUT_BOUND = (-3,3)
EPS = 1
def create_model(len_in, no_layers, no_neuron):
    model = {}
    model["shape"] = str((1,len_in))
    model["bounds"] =str([INPUT_BOUND])
    layers = []
    previous_layer = len_in
    for i in range(no_layers):
        linear_layer = {"type":"linear"}
        bias = str(np.round(generate_rand((no_neuron,), -5, 5), 4).tolist())
        weights = str(np.round(generate_rand((no_neuron, previous_layer), -5, 5), 4).tolist())
        previous_layer = no_neuron
        linear_layer["weights"] = weights
        linear_layer["bias"] = bias
        layers.append(linear_layer)
        layers.append({"type": "function", "func": "relu"})

    linear_layer = {"type":"linear"}
    bias = str(np.round(generate_rand((2,), -5, 5), 3).tolist())
    weights = str(np.round(generate_rand((2, previous_layer), -5, 5), 3).tolist())
    linear_layer["weights"] = weights
    linear_layer["bias"] = bias
    layers.append(linear_layer)

    model["layers"] = layers
    final = {"model" : model}
    #with open("temp/spec.json", "w") as jsonFile:
        #jsonFile.write(json.dumps(final, indent=4))
    return final

def generate_rand(shape, lower, upper):
    args = shape
    rand = np.random.rand(*args)
    rand = (upper - lower) * rand + lower
    return rand

def find_example():
    spec = create_model(2, 3, 4)
    with open("temp/spec.json", "w") as jsonFile:
        jsonFile.write(json.dumps(spec, indent=4))
    #with open("temp/spec.json", 'r') as f:
    #    spec = json.load(f)
    index = 0
    add_assertion(spec, EPS)
    add_solver(spec)
    model, assertion, solver, display = parse(spec)
    network_tf = "temp/network.tf"
    save_tf_model(network_tf, model, len(model.layers)-1)
    find = False
    for no in range(5):
        x0s = np.round(generate_rand((100, 2), INPUT_BOUND[0], INPUT_BOUND[1]), 4)
        example = []

        res = refinepoly_run(x0s, None, network_tf, EPS, 2)
        print("Refine result", res)
        for j in res:
            success_lst = newApproach(model, x0s[j], '')
            if success_lst:
                print("Find example")
                print(x0s[j])
                with open("temp/x{}.txt".format(index), "w") as exp:
                    exp.write(str(x0s[j].tolist()))
                find = True
                index += 1
    return find
        
def print_detail():
    network_tf = "temp/network.tf"
    with open("temp/examples/spec.json", 'r') as f:
        spec = json.load(f)
    add_assertion(spec, 0.01)
    add_solver(spec)
    model, assertion, solver, display = parse(spec)
    x0 =  np.array(ast.literal_eval(open("temp/examples/x2.txt", 'r').readline()))
    res = refinepoly_run(np.reshape(x0, (-1,2)), None, network_tf, 2)
    success_lst = newApproach(model, x0, '')
    if success_lst:
        print("Find example")
    assertion['x0'] = str(x0.tolist())
    res = solver.solve(model, assertion)
    if res == 0:
        print("DeepPoly can't prove this")

if __name__ == '__main__':
    res = False
    while not res:
        res = find_example()
    #print_detail()


        




