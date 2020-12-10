import json

from solver.deepcegar_impl import Poly
from json_parser import parse
from sklearn import svm
from z3 import *
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
        lower = lst_poly[layer_idx + 1].lw
        upper = lst_poly[layer_idx + 1].up
        print("Lower Upper:", lower, upper)
        new_model = Model(shape, lower, upper, model.layers[layer_idx+1:], None)
        print("New Model bound:", new_model.lower, model.upper)

    size = np.prod(new_model.shape)
    
    n = 100000
    generate_y = np.zeros((n, size))
    input_x = np.zeros((n, size))
    
    for i in range(n):
        x0 = generate_x(size, new_model.lower, model.upper)
        y0 = new_model.apply(x0)
        generate_y[i] = y0
        input_x[i] = x0
    
    return input_x, generate_y

def prove(coef, intercept):
    s = Solver()
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = Reals('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12')
    
    P1 = And([x1 >= -1, x1 <= 1, x2 >= -1, x2 <= 1, x3 <= x1 + x2, x3 >= x1 + x2, x4 <= x1 - x2, x4 >= x1 - x2, x5 >= 0, x5 <= 0.5*x3 + 1, x6 >= 0, x6 <= 0.5*x4 + 1])
    P2 = And([x5 >= 0, x5 <= 2, x6 >= 0, x6 <= 2, x7 >= x5 + x6, x7 <= x5 + x6, x8 >= x5 - x6, x8 <= x5 - x6, x9 >= x7, x9 >= x7, x10 >= 0, x10 <= 0.5*x8 + 1,
              x11 >= x9 + x10 + 1, x11 <= x9 + x10 + 1, x12 >= x10, x12 <= x10])
    f = coef[0]*x5 + coef[1]*x6 + intercept > 0
    
    Property = ForAll([x5, x6], x12 < x11)
              
    s.push()
    s.add(Not(Implies(And(P1, P2), Property))) #if unsat then Property hold
    r = s.check()
    print(r)
    if r == unsat:
        print("Property hold")
    s.pop()
    
    s.push()
    s.add(Not(Implies(P1, f))) #if unsat then P1 => f is valid
    r = s.check()
    print(r)
    if r == unsat:
        print("P1 => f is valid")
    else:
        print("P1 => f is not valid")       
    s.pop()
    
    s.add(Not(Implies(And(P2, f), Property))) #if unsat then P2 and f => Property is valid

    r = s.check()
    print(r)
    if r == unsat:
        print("P2 and f => Property is valid")
    else:
        print("P2 and f => Property is not valid")

    s.add(Not(Implies(P2, Property))) #if unsat then P2 => Property is valid
    r = s.check()
    print(r)
    if r == unsat:
        print("P2 => Property is valid")
    else:
        print("P2 => Property is not valid")
        print(s.model())
        

def main():
    base_path = Path(__file__).parent
    model_path = base_path / "./deeppoly_model/spec.json"
    with open(model_path, 'r') as f:
        spec = json.load(f)

    add_assertion(spec)
    add_solver(spec)
    
    model, assertion, solver, display = parse(spec)

    #Sample before relu layer
    #print("Sample before relu layer")
    #lst_poly = buildDeepPoly(model)
    #input_x, generate_y = generate_sample(model, 0, lst_poly)

    #label = [a == b for a, b in zip(np.argmax(input_x, axis=1), np.argmax(generate_y, axis=1))]
            
    #label = np.array(label, dtype=int)

    #clf = svm.SVC(kernel="linear")
    #clf.fit(generate_y, label)
    #print(clf.coef_)
    #print(clf.intercept_)
    #prove(clf.coef_[0], clf.intercept_[0])

    #print()
    print("Sample after relu layer")
    #Sample after relu layer 
    lst_poly = buildDeepPoly(model)
    input_x, generate_y = generate_sample(model, 1, lst_poly)

    label = [a == b for a, b in zip(np.argmax(input_x, axis=1), np.argmax(generate_y, axis=1))]
    label = np.array(label, dtype=int)
    clf = svm.LinearSVC()
    clf.fit(input_x, label)
    print(clf.coef_)
    print(clf.intercept_)
    prove(clf.coef_[0], clf.intercept_[0])

    

if __name__ == '__main__':
    main()
