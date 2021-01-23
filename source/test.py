import json

from solver.deepcegar_impl import Poly
from json_parser import parse
from sklearn import svm
from z3 import *
from model.lib_models import Model
from utils import *
from pathlib import Path
from assertion.lib_functions import d2
import matplotlib.pyplot as plt

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
    
    plt.show()

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

def buildDeepPoly(model):
    x0_poly = Poly()

    x0_poly.lw = [-1, -1]
    x0_poly.up = [1, 1]

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
    size = np.prod(model.shape)
    sample = np.zeros((no_sp, size))
    y_sample = []
                  
    x_idx = -1
    for i in range(size):
        if const[i] != 0 :
            x_idx = i
            break
    mask = np.ones(size, dtype=bool)
    mask[x_idx] = False
    coef = np.array(const[:-1])
    intercept = const[-1]
    
    while idx < no_sp:
        x = generate_x(size - 1, model.lower[mask], model.upper[mask])
        last = (-intercept - np.dot(coef[mask], x))/coef[x_idx]
        if last > model.lower[x_idx] and last < model.upper[x_idx]:
            x = np.insert(x, x_idx, last)
            #x = np.around(x, 4)
            sample[idx] = x
            y_sample.append(model.apply(x))
            idx += 1
    #sample *= 100
    return sample, np.array(y_sample)

        

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
    label = np.array(label, dtype=int)
    #input_x *= 100
    
    clf = svm.SVC(kernel="linear", C=1000)
    clf.fit(input_x, label)
    const = np.concatenate((clf.coef_[0], clf.intercept_))
    print("score:", clf.score(input_x, label))
    plot(input_x, label, const)
    norm(const)
    print("const:", const)
    #prove(x0, layer_idx + 1, const, model, lst_poly)
    if True:
        return const
    index = 0
    
    while True:
        index += 1
        if index % 100 == 0:
            print("score:", clf.score(input_x, label))
            print("const:", const)
        #clf = svm.SVC(kernel="linear", C=1e10)
        sample, y_sample = generate_sample(10, new_model, const)
        new_label = np.array([y0.argmax() == y.argmax() for y in y_sample], dtype=int)
        input_x = np.concatenate((input_x, sample))
        label = np.concatenate((label, new_label))
        clf.fit(input_x, label)
        new_const = norm(np.concatenate((clf.coef_[0], clf.intercept_)))
        #print(np.concatenate((clf.coef_[0], clf.intercept_)))
        if d2(new_const, const) < 1e-6:
            break
        const = new_const


        
    #plot(input_x, label, clf)
    #print(index)
    print("score:", clf.score(input_x, label))
    print("const:", const)
    #const[-1] /= 100
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
            print(msg + " not valid")
            print("Counterexample:", printModelVal(s.model()))
        else:
            print("Unknown")
    if r == unsat:
        return True
    return False

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
        lower = [dot(pre_X, l) for l in lst_poly[idx].ge]
        upper = [dot(pre_X, u) for u in lst_poly[idx].le]
        P += [bound for x, l, u in zip(X, lower, upper) for bound in [x >=l, x <= u]]
        pre_X = X + [1]
    return And(P), index, X

def prove(x0, idx_ly, const, model, lst_poly):
    s = Solver()

    P1, index, X = getConstraints(lst_poly, 1, 0, idx_ly + 1)
    P2, index, y = getConstraints(lst_poly, index, idx_ly + 1, len(lst_poly))
    #print(P1)
    #print(P2)

    y0_arg = np.argmax(model.apply(np.array([x0])), axis=1)[0]
    Property = And([ForAll(X, y[y0_arg] > y[i]) for i in range(len(y)) if i != y0_arg])
    #print(Property)

    f = dot(X + [1], const) > 0 
    #print(f)

    valid_prove(And(P1, P2), Property, "P1 and P2 => Property")
    
    valid_prove(P1, f, "P1 => f")

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
        x = np.around(x, 5)
        y = model.apply(x)
        generate_y[i] = y
        input_x[i] = x
        
    y0 = model.apply(x0)
    label = [y0.argmax() == y.argmax() for y in generate_y]
    label = np.array(label, dtype=int)
    print("Sum of label:", sum(label))

def main():
    base_path = Path(__file__).parent
    model_path = base_path / "./deeppoly_model/spec_ReLu.json"
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

    idx_ly = 3 

    const = generate_const(x0, model, idx_ly, lst_poly)
    #const = np.array([-1, 2.97250515, 6])
    #printPlotLine(x0, const, model, idx_ly, lst_poly)
    re = prove(x0, idx_ly + 1, const, model, lst_poly)
    #print("f: x >", -const[-1]/const[0])
    #checkSum(model, x0)
    




    

    

if __name__ == '__main__':
    main()
