import json
import ast
import sys
from os import path
sys.path.append(path.abspath("../eran/tf_verify"))
sys.path.insert(0, path.abspath("../eran/ELINA/python_interface/"))
sys.path.insert(0, path.abspath("../eran/deepg/code/"))


import krelu
from solver.deepcegar_impl import Poly
from json_parser import parse, parse_solver
from sklearn import svm
from z3 import *
from model.lib_models import Model
from utils import *
from pathlib import Path
from assertion.lib_functions import d2
from eran_test import checkRefinePoly
import matplotlib.pyplot as plt
import numpy as np
import time

def add_assertion(spec):
    assertion = dict()

    assertion['robustness'] = 'local'
    assertion['distance'] = 'di'
    assertion['eps'] = str(eps) # bounds are updated so eps is not necessary

    spec['assert'] = assertion

def add_solver(spec):
    solver = dict()

    solver['algorithm'] = 'deepcegar'
    solver['has_ref'] = 'False'
    solver['max_ref'] = '20'
    solver['ref_typ'] = '0'
    solver['max_sus'] = '100'


    spec['solver'] = solver

def plot(X, y, const, sup_vec=True):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    xy_ = np.insert(xy, len(xy[0]), 1, axis=1)

    #Z = clf.decision_function(xy).reshape(XX.shape)
    Z = np.matmul(xy_, const).reshape(XX.shape)

    # plot decision boundary and margins
    if sup_vec:
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])
    else:
        ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5,
                   linestyles=['-']) 

    plt.savefig('plot.png', format='png')

def printPlotLine(x0, const, model, layer_idx, lst_poly):
    if layer_idx == -1 or layer_idx >= len(model.layers):
        new_model = model
    else:       
        no_neuron = len(lst_poly[layer_idx].lw)
        shape = np.array([1, no_neuron])
        lower = lst_poly[layer_idx + 1].lw
        upper = lst_poly[layer_idx + 1].up
        new_model = Model(shape, lower, upper, model.layers[layer_idx+1:], None)


    size = np.prod(new_model.shape)
    
    n = 5000
    generate_y = np.zeros((n, size))
    input_x = np.zeros((n, size))
    
    for i in range(n):
        x = generate_x(size, new_model.lower, new_model.upper)
        #x = np.around(x, 4)
        y = new_model.apply(x)
        generate_y[i] = y
        input_x[i] = x
        
    y0 = model.apply(x0)
    label = [y0.argmax() == y.argmax() for y in generate_y]
    #label = [y[0] <= 4 for y in generate_y]
    label = np.array(label, dtype=int)
    plot(input_x, label, const, False)

def norm(const):
    x_idx = -1
    for i in range(len(const)):
        if const[i] != 0 :
            x_idx = i
            break
    const /= abs(const[x_idx])
    return const

def getDeepPoly(model, x0, eps):
    x0_poly = Poly()

    
    lw = np.maximum(model.lower, x0 - eps)
    up = np.minimum(model.upper, x0 + eps)

    x0_poly.lw = lw
    x0_poly.up = up

    no_neuron = len(x0_poly.lw)

    x0_poly.le = np.zeros((no_neuron, no_neuron + 1))
    x0_poly.ge = np.zeros((no_neuron, no_neuron + 1))

    x0_poly.le[:,-1] = x0_poly.up
    x0_poly.ge[:,-1] = x0_poly.lw

    lst_poly = [x0_poly]
    for idx in range(len(model.layers)):
        xi_poly_curr = model.forward(lst_poly[idx], idx, lst_poly)
        lst_poly.append(xi_poly_curr)
    return lst_poly


def generate_sample(no_sp, model, const):
    idx = 0
    size = model.shape[1]
                  
    x_idx = -1
    for i in range(size):
        if const[i] != 0 :
            x_idx = i
            break
    mask = np.ones(size, dtype=bool)
    mask[x_idx] = False
    coef = np.array(const[:-1])
    intercept = const[-1]

    x = np.random.rand(2 * no_sp, size - 1)
    x = (model.upper[mask] - model.lower[mask]) * x + model.lower[mask]
    last = (-intercept - np.dot(x, coef[mask]))/coef[x_idx]
    x = np.insert(x, x_idx, last, axis=1)
    filter_ = [True if val >= model.lower[x_idx] and val <= model.upper[x_idx] else False for val in last] 
    x = x[filter_]
    x = np.around(x, 4)
    x = np.unique(x, axis=0)
    y_sample = apply_model(model, x)
    slice_ = no_sp if no_sp <= x.shape[0] else x.shape[0]
    #sample *= 100
    return x[:slice_], y_sample[:slice_]

def active_learning(y0, model, clf, data, time_limit):
    index = 0
    input_x, label = data
    const = np.concatenate((clf.coef_[0], clf.intercept_))
    start = time.perf_counter()
    while time.perf_counter() - start < time_limit:
        index += 1
        if index % 100 == 0:
            print("score:", clf.score(input_x, label))
            print("const:", const)

        sample, y_sample = generate_sample(100, model, const)
        new_label = np.array([y0.argmax() == y.argmax() for y in y_sample], dtype=int)
        #new_label = np.array([y[0] <= 4 for y in y_sample], dtype=int)
        input_x = np.concatenate((input_x, sample))
        label = np.concatenate((label, new_label))
        clf.fit(input_x, label)
        new_const = norm(np.concatenate((clf.coef_[0], clf.intercept_)))

        if d2(new_const, const) < 1e-4:
            break
        const = new_const
    return const

def generate_const(x0, model, layer_idx, lst_poly):
    if layer_idx == -1 or layer_idx >= len(model.layers):
        new_model = model
    else:       
        no_neuron = len(lst_poly[layer_idx].lw)
        shape = np.array([1, no_neuron])
        lower = lst_poly[layer_idx + 1].lw
        upper = lst_poly[layer_idx + 1].up
        new_model = Model(shape, lower, upper, model.layers[layer_idx+1:], None)


    size = np.prod(new_model.shape)
    
    n = 15000
    generate_y = np.zeros((n, size))
    input_x = np.random.rand(n, size)
    input_x = (new_model.upper - new_model.lower) * input_x + new_model.lower
    input_x = np.around(input_x, 4)
    generate_y = apply_model(new_model, input_x)
        
    y0 = model.apply(x0)
    label = [y0.argmax() == y.argmax() for y in generate_y]
    #label = [y[0] <= 4 for y in generate_y]
    label = np.array(label, dtype=int)
    
    clf = svm.LinearSVC(C=100, tol=1e-5)
    clf.fit(input_x, label)
    #new_input = np.array(list(map(lambda x: x[0], input_x))).reshape(-1, 1)
    #clf.fit(new_input, label)
    
    #print("Size:", clg.coef_.shape())
    const = np.concatenate((clf.coef_[0], clf.intercept_))
    #const = np.insert(const, 1, 0)

    #print("score:", clf.score(input_x, label))
    #plot(input_x, label, const)
    norm(const)
    #print("const:", const)
    #prove(x0, layer_idx + 1, const, model, lst_poly)
    
    const = active_learning(y0, new_model, clf, (input_x, label), 10)
    return const

def dot(a, b):
    return simplify(Sum([x*y for x, y in zip(a,b)]))

def printModelVal(model):
    m = model
    counter = sorted ([(d, float(m[d].numerator_as_long())/float(m[d].denominator_as_long())) for d in m], key = lambda x: str(x[0]))
    coun_str = ', '.join([str(x[0]) + " = " + str(x[1]) for x in counter])
    return coun_str

def valid_prove(a, b, msg = "a => b", out = True):
    s = Solver()
    s.add(Not(Implies(a, b)))
    r = s.check()
    if out:
        if r == unsat:
            print(msg + " is valid")
        elif r == sat:
            m = 0
            #print(msg + " not valid")
            #print("Counterexample:", printModelVal(s.model()))
        else:
            print("Unknown if it valid or not")
    if r == unsat:
        return True
    return False

def getConstraints(lst_poly, index, start, end):
    if start == 0:
        pre_X = [Real("x%s" % i) for i in range(index, len(lst_poly[start].lw) + index)] + [1]
        P = []
    else:
        #pre_X = [Real("x%s" % i) for i in range(index - len(lst_poly[start-1].lw), index)] + [1]
        pre_X = [Real("x%s" % i) for i in range(index, len(lst_poly[start].lw) + index)] + [1]
        P = [bound for x, l, u in zip(pre_X[:-1], lst_poly[start-1].lw, lst_poly[start-1].up) for bound in [x >= l, x <= u]]
        index += len(lst_poly[start].lw)
    first_X = pre_X[0:-1]
    for idx in range(start, end):
        X = [Real("x%s" % i) for i in range(index, index + len(lst_poly[idx].lw))]
        index += len(lst_poly[idx].lw)
        lower = [dot(pre_X, l) for l in lst_poly[idx].ge]
        upper = [dot(pre_X, u) for u in lst_poly[idx].le]
        P += [bound for x, l, u in zip(X, lower, upper) for bound in [x >=l, x <= u]]
        pre_X = X + [1]

    #if start == end:
    #    X = pre_X[0:-1]
    
    return And(P), first_X, X 

def prove(x0, idx_ly, const, model, lst_poly):
    s = Solver()

    P2, X, y = getConstraints(lst_poly, 0, idx_ly + 1, len(lst_poly))

    #print(P2)
    
    y0_arg = np.argmax(model.apply(np.array([x0])), axis=1)[0]
    Property = ForAll(X, And([y[y0_arg] > y[i] for i in range(len(y)) if i != y0_arg]))
    #Property = ForAll(X, y[0] <= 4)
    #print(Property)

    f = dot(X + [1], const) > 0 
    #print(f)
    re = valid_prove(And(P2, f), Property, "P2 and f = > Property")
    #valid_prove(P2, Property, "P2 => Property")
    return re

def checkSum(model, x0):
    size = np.prod(model.shape)
    n = 10000
    generate_y = np.zeros((n, size))
    input_x = np.zeros((n, size))
    
    for i in range(n):
        x = generate_x(size, model.lower, model.upper)
        x = np.around(x, 4)
        y = model.apply(x)
        generate_y[i] = y
        input_x[i] = x
        
    y0 = model.apply(x0)
    label = [y0.argmax() == y.argmax() for y in generate_y]
    label = np.array(label, dtype=int)
    print("Sum of label:", sum(label))

def sort_layer(lst_poly):
    lst_idx = range(len(lst_poly)-2)
    num_neurons_lst = [len(i.lw) for i in lst_poly[1:len(lst_poly)-1]]
    lst_idx = list(zip(lst_idx, num_neurons_lst))
    lst_idx.reverse()
    lst_idx.sort(reverse=True, key= lambda x:x[1])
    return [x[0] for x in lst_idx]

def save_tf_model(path, model, idx):
    ops, wei_biass = [], []
    for i_lay in range(idx+1):
        layer = model.layers[i_lay]
        op, wei_bias = layer.to_tf() 
        if type(layer).__name__ == "Function":
            if i_lay != 0 and ops[-1] == 'Affine':
                ops[-1] = op
            elif i_lay != 0 and ops[-1] != 'Affine':
                raise NameError('Function layer next to a function layer')
            else:
                raise NameError('Function layer as first layer')
        else:
            ops.append(op)
            wei_biass.append(wei_bias)

    with open(path / 'temp.tf', 'w') as f:
        for op, wei_bias in zip(ops, wei_biass):
            f.write(op+'\n')
            f.write(wei_bias)


def testCegar(model, x0):
    spec = dict()
    add_solver(spec)
    add_assertion(spec)
    spec['assert']['x0'] = x0
    spec['solver']['has_ref'] = 'True'
    solver = parse_solver(spec['solver'])
    res = solver.solve(model, spec['assert'])
    return res

def refineF(x0, idx_ly, const, model, lst_poly, objVal):
    ori = const[-1]
    const[-1] -= objVal - 0.01
    re = prove(x0, idx_ly + 1, const, model, lst_poly)
    while not re and const[-1] < ori + objVal:
        const[-1] += objVal/10
        re = prove(x0, idx_ly + 1, const, model, lst_poly)
    return re

eps = 0.01

def main():
    base_path = Path(__file__).parent.parent
    models_path = base_path / "source/deeppoly_model/"
    #model_path = models_path / "spec_Cegar.json"
    model_path = base_path / "benchmark/cegar/nnet/mnist_relu_3_10/spec.json"
    with open(model_path, 'r') as f:
        spec = json.load(f)

    add_assertion(spec)
    add_solver(spec)
    
    model, assertion, solver, display = parse(spec)
    #assertion['x0'] = '[0,0]'
    #x0 = np.array([0,0])
    list_potential, unknow, find_exp = [], [], []
    for idx_data in [6]:
        find = True
        data_file = "data" + str(idx_data) + ".txt"
        PathX = base_path / "benchmark/cegar/data/mnist_fc" / data_file
        assertion['x0'] = PathX
        x0 = np.array(ast.literal_eval(read(PathX)))
        #print("Label of x0: {}".format(model.apply(np.array([x0])).argmax(axis=1)[0]))

        res = solver.solve(model, assertion)
        if res == 1 or res == 2:
            print("DeepPoly can prove this model")
            continue
        else:
            print("DeepPoly can't prove this model:", idx_data)
            res = testCegar(model, PathX)
        
        if res == 1:
            list_potential.append(idx_data)
        elif res == 2:
            continue
        else:
            print("Cegar also can't prove this")
            unknow.append(idx_data)

        lst_poly = getDeepPoly(model, x0, eps) #May need to get it directly from solver
        #lst_poly.pop()
        print(sort_layer(lst_poly))
        for idx_ly in sort_layer(lst_poly):
            #idx_ly = 2
            try:
                const = generate_const(x0, model, idx_ly, lst_poly)
                const = np.around(const, 3)
            except ValueError as err:
                continue

            #const = np.ones(11)
            print(const)

            save_tf_model(models_path / "nets/", model, idx_ly)
            res, objVal = checkRefinePoly(models_path / "nets/temp.tf", lst_poly[0].lw, lst_poly[0].up, const, True)
            if res:
                print("Prove P1 => f")
            else:
                print("Not Prove P1 => f")
        
            #printPlotLine(x0, const, model, idx_ly, lst_poly)
            re = prove(x0, idx_ly + 1, const, model, lst_poly)
            refine = False
            #checkSum(model, x0)
            #test_apply(model)
            if res and not re:
                re = refineF(x0, idx_ly, const, model, lst_poly, objVal)
                refine = True
            if not re:
                print("Not prove P2 and f => Property")
            if res and re:
                print("Find the example:", PathX)
                print("Layer use:", idx_ly)
                find_exp.append((PathX, idx_ly, refine))
                
    print("potential:", list_potential)
    print("Unknow:", unknow)
    print("Example Find:", find_exp)

    

    

if __name__ == '__main__':
    main()
