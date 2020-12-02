import json

from solver.deepcegar_impl import Poly
from json_parser import parse
from sklearn import svm
from model.lib_models import Model
from utils import *
from pathlib import Path

def add_assertion(spec):
    assertion = dict()

    assertion['robustness'] = 'local'
    assertion['distance'] = 'di'
    assertion['eps'] = '1e9' # bounds are updated so eps is not necessary

    spec['assert'] = assertion

def add_solver(spec):
    solver = dict()

    solver['algorithm'] = "optimize"

    spec['solver'] = solver

def buildDeepPoly(model):
    x0_poly = Poly()

    x0_poly.lw = [-1, -1]
    x0_poly.up = [1, 1]

    x0_poly.lt = np.eye(3)[0:-1]
    x0_poly.gt = np.eye(3)[0:-1]

    lst_poly = [x0_poly]
    for idx in range(len(model.layers)-1):
        xi_poly_curr = model.forward(lst_poly[idx], idx, lst_poly)
        lst_poly.append(xi_poly_curr)
    return lst_poly

def generate_sample(model, layer_idx, lst_poly):
    if layer_idx == -1 or layer_idx >= len(model.layers):
        new_model = model
    else:       
        no_neuron = len(lst_poly[layer_idx].lw)
        shape = np.array([1, no_neuron])
        lower = lst_poly[layer_idx].lw
        upper = lst_poly[layer_idx].up
        new_model = Model(shape, lower, upper, model.layers[layer_idx+1:], None)

    size = np.prod(new_model.shape)

    n = 10000
    generate_y = np.zeros((n, size))
    input_x = np.zeros((n, size))
    
    for i in range(n):
        x0 = generate_x(size, new_model.lower, model.upper)
        y0 = new_model.apply(x0)
        generate_y[i] = y0
        input_x[i] = x0
    
    return input_x, generate_y
        

def main():
    base_path = Path(__file__).parent
    model_path = base_path / "./deeppoly_model/spec.json"
    with open(model_path, 'r') as f:
        spec = json.load(f)

    add_assertion(spec)
    add_solver(spec)
    
    model, assertion, solver, display = parse(spec)

    lst_poly = buildDeepPoly(model)
    input_x, generate_y = generate_sample(model, 0, lst_poly)

    label = [a == b for a, b in zip(np.argmax(input_x, axis=1), np.argmax(generate_y, axis=1))]
    label = np.array(label, dtype=int)
    clf = svm.SVC(kernel="linear")
    clf.fit(generate_y, label)
    print(clf.coef_)
    print(clf.intercept_)


    

if __name__ == '__main__':
    main()
