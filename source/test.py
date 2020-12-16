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

    no_neuron = len(x0_poly.lw)

    x0_poly.lt = np.zeros((no_neuron, no_neuron + 1))
    x0_poly.gt = np.zeros((no_neuron, no_neuron + 1))

    x0_poly.lt[:,-1] = x0_poly.up
    x0_poly.gt[:,-1] = x0_poly.lw

    lst_poly = [x0_poly]
    for idx in range(len(model.layers)):
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
        new_model = Model(shape, lower, upper, model.layers[layer_idx+1:], None)


    size = np.prod(new_model.shape)
    
    n = 100000
    generate_y = np.zeros((n, size))
    input_x = np.zeros((n, size))
    
    for i in range(n):
        x0 = generate_x(size, new_model.lower, new_model.upper)
        y0 = new_model.apply(x0)
        generate_y[i] = y0
        input_x[i] = x0

    
    return input_x, generate_y

def dot(a, b):
    return simplify(Sum([x*y for x, y in zip(a,b)]))

def getConstraints(lst_poly, index, start, end):
    if start == 0:
        pre_X = [Real("x%s" % i) for i in range(index, len(lst_poly[start].lw) + index)] + [1]
        P = []
    else:
        pre_X = [Real("x%s" % i) for i in range(index - len(lst_poly[start-1].lw), index)] + [1]
        P = [bound for x, l, u in zip(pre_X[:-1], lst_poly[start-1].lw, lst_poly[start-1].up) for bound in [x >= l, x <= u]]
    
    for idx in range(start, end):
        X = [Real("x%s" % i) for i in range(index, index + len(lst_poly[idx].lw))]
        index += len(lst_poly[idx].lw)
        lower = [dot(pre_X, l) for l in lst_poly[idx].gt]
        upper = [dot(pre_X, u) for u in lst_poly[idx].lt]
        P += [bound for x, l, u in zip(X, lower, upper) for bound in [x >=l, x <= u]]
        pre_X = X + [1]
    return And(P), index, X

def prove2(x0, idx_ly, clf, model, lst_poly):
    s = Solver()

    P1, index, X = getConstraints(lst_poly, 1, 0, idx_ly + 1)
    P2, index, y = getConstraints(lst_poly, index, idx_ly + 1, len(lst_poly))
    print(P1)
    print(P2)

    y0_arg = np.argmax(model.apply(np.array([x0])), axis=1)[0]
    Property = And([ForAll(X, y[y0_arg] > y[i]) for i in range(len(y)) if i != y0_arg])
    print(Property)

    f = dot(X + [1], [*clf.coef_[0], *clf.intercept_]) > 0 
    print(f)

    s.push()
    s.add(Not(Implies(And(P1, P2), Property))) #if unsat then Property hold
    r = s.check()
    print(r)
    if r == unsat:
        print("Property hold")
    else:
        print("Property not hold")
        print(s.model())
    s.pop()
    
    s.push()
    s.add(Not(Implies(P1, f))) #if unsat then P1 => f is valid
    r = s.check()
    print(r)
    if r == unsat:
        print("P1 => f is valid")
    else:
        print("P1 => f is not valid")
        print(s.model())
    s.pop()
    
    s.push()
    s.add(Not(Implies(And(P2, f), Property))) #if unsat then P2 and f => Property is valid
    r = s.check()
    print(r)
    if r == unsat:
        print("P2 and f => Property is valid")
    else:
        print("P2 and f => Property is not valid")
        print(s.model())
    s.pop()

    s.push()
    s.add(Not(Implies(P2, Property))) #if unsat then P2 => Property is valid
    r = s.check()
    print(r)
    if r == unsat:
        print("P2 => Property is valid")
    else:
        print("P2 => Property is not valid")
        print(s.model())
    s.pop()
        

def main():
    base_path = Path(__file__).parent
    model_path = base_path / "./deeppoly_model/spec_Cegar.json"
    with open(model_path, 'r') as f:
        spec = json.load(f)

    add_assertion(spec)
    add_solver(spec)
    
    model, assertion, solver, display = parse(spec)
    x0 = np.array([0,0])
    print("Label of x0: {}".format(model.apply(np.array([x0])).argmax(axis=1)[0]))
    
    print("Sample after relu layer")
    #Sample after relu layer 
    lst_poly = buildDeepPoly(model)

    
    idx_ly = 1 
    input_x, generate_y = generate_sample(model, idx_ly, lst_poly)

    label = [x0.argmax() == y.argmax() for y in generate_y]
    label = np.array(label, dtype=int)
    #print(input_x)
    #print(generate_y)
    #print(label)
    clf = svm.LinearSVC()
    clf.fit(input_x, label)
    print(clf.coef_)
    print(clf.intercept_)
    #prove(clf.coef_[0], clf.intercept_[0])
    prove2(x0, idx_ly + 1, clf, model, lst_poly)

    

if __name__ == '__main__':
    main()
