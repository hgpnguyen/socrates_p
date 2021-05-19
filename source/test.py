import json
import ast
import sys
from os import path
sys.path.append(path.abspath("../eran/tf_verify"))
sys.path.insert(0, path.abspath("../eran/ELINA/python_interface/"))
sys.path.insert(0, path.abspath("../eran/deepg/code/"))

import csv
import krelu
from solver.deepcegar_impl import Poly
from json_parser import parse, parse_solver
from sklearn import svm
from z3 import *
from model.lib_models import Model
from utils import *
from pathlib import Path
from assertion.lib_functions import d2
from eran_test import checkRefinePoly, getModel
from run_refinepoly import refinepoly_run
import matplotlib.pyplot as plt
import numpy as np
import time

EPS = 0.01

def add_assertion(spec, eps):
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


def generate_sample(no_sp, middle_x, model, neuron_lst, const):
    idx = 0
    size = model.shape[1]
                  
    x_idx = -1
    for i in range(size):
        if const[i] != 0 :
            x_idx = i
            break
    mask = np.ones(size, dtype=bool)
    mask[x_idx] = False
    #print(const, len(const))
    coef = np.array(const[:-1])
    intercept = const[-1]
    #print(coef, len(coef))

    x = np.random.rand(2 * no_sp, size - 1)
    x = (model.upper[mask] - model.lower[mask]) * x + model.lower[mask]
    last = (-intercept - np.dot(x, coef[mask]))/coef[x_idx]
    x = np.insert(x, x_idx, last, axis=1)
    filter_ = [True if val >= model.lower[x_idx] and val <= model.upper[x_idx] else False for val in last] 
    x = x[filter_]
    x = np.around(x, 4)
    x = np.unique(x, axis=0)
    template = np.repeat(middle_x, x.shape[0], axis=0)
    template[:, neuron_lst] = x[:, neuron_lst]
    x = template
    y_sample = apply_model(model.layers, x)
    slice_ = no_sp if no_sp <= x.shape[0] else x.shape[0]
    return x[:slice_, neuron_lst], y_sample[:slice_]

def getClf():
    return svm.LinearSVC(loss='hinge', C=1000, tol=1e-6)

def active_learning(middle_x, model, const, neuron_lst, data, time_limit):
    index = 0
    y0 = model.apply(middle_x)
    input_x, label = data
    clf = getClf()
    start = time.perf_counter()
    while time.perf_counter() - start < time_limit:
        index += 1
        if index % 100 == 0:
            print("score:", clf.score(input_x, label))
            print("const:", const)

        sample, y_sample = generate_sample(100, middle_x, model, neuron_lst, const)
        new_label = np.array([y0.argmax() == y.argmax() for y in y_sample], dtype=int)
        #new_label = np.array([y[0] <= 4 for y in y_sample], dtype=int)
        input_x = np.concatenate((input_x, sample))
        label = np.concatenate((label, new_label))
        clf.fit(input_x, label)
        new_const = np.zeros(len(const)-1)
        new_const[neuron_lst] = clf.coef_[0]
        new_const = norm(np.concatenate((new_const, clf.intercept_)))

        if d2(new_const, const) < 1e-4:
            break
        const = new_const
    return const

def generate_const(x0, model, layer_idx, lst_poly, neuron_lst=None):
    if layer_idx == -1 or layer_idx >= len(model.layers):
        new_model = model
    else:       
        no_neuron = len(lst_poly[layer_idx].lw)
        shape = np.array([1, no_neuron])
        lower = lst_poly[layer_idx + 1].lw
        upper = lst_poly[layer_idx + 1].up
        new_model = Model(shape, lower, upper, model.layers[layer_idx+1:], None)


    size = np.prod(new_model.shape)
    if neuron_lst is None:
        neuron_lst = list(range(size))

    n = 15000

    middle_x = apply_model(model.layers[:layer_idx+1], x0)
    input_x = np.repeat(middle_x, n, axis=0)
    input_x = np.around(input_x, 4)
    rand_int = np.random.rand(n, len(neuron_lst))
    rand_int = (new_model.upper[neuron_lst] - new_model.lower[neuron_lst]) * rand_int + new_model.lower[neuron_lst]
    rand_int = np.around(rand_int, 4)
    input_x[:, neuron_lst] = rand_int
    generate_y = apply_model(new_model.layers, input_x)
        
    y0 = model.apply(x0)
    label = [y0.argmax() == y.argmax() for y in generate_y]
    #label = [y[0] <= 4 for y in generate_y]
    label = np.array(label, dtype=int)
    
    clf = getClf()
    clf.fit(rand_int, label)
    #new_input = np.array(list(map(lambda x: x[0], input_x))).reshape(-1, 1)
    #clf.fit(new_input, label)
    
    #print("Size:", clg.coef_.shape())
    const = np.zeros(size)
    const[neuron_lst] = clf.coef_[0]
    const = np.concatenate((const, clf.intercept_))
    #plot(input_x, label, const)
    norm(const)

    
    const = active_learning(middle_x, new_model, const, neuron_lst, (rand_int, label), 3)
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
    s.set("timeout", 1000)
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

def sort_layer(layers, lst_poly):
    lst_idx, num_neurons_lst = [], []
    for idx in range(len(layers)-1):
        if not layers[idx].is_poly_exact():
            lst_idx.append(idx)
            num_neurons_lst.append(len(lst_poly[idx+1].lw))
    lst_idx = list(zip(lst_idx, num_neurons_lst))
    lst_idx.reverse()
    lst_idx.sort(reverse=False, key= lambda x:x[1])
    return [x[0] for x in lst_idx]

def sort_neuron(model, lst_poly, x0, idx_layer):
    assert len(lst_poly) == len(model.layers) + 1
    if model.layers[idx_layer].is_poly_exact():
        return None
    y0 = np.argmax(model.apply(x0), axis=1)[0]
    poly_out = lst_poly[-1]
    no_neurons = len(poly_out.lw)
    check_false = False
    #get lst_ge
    for y in range(no_neurons):
        if y != y0 and poly_out.lw[y0] <= poly_out.up[y]:
            poly_res = Poly()

            poly_res.lw = np.zeros(1)
            poly_res.up = np.zeros(1)

            poly_res.le = np.zeros([1, no_neurons + 1])
            poly_res.ge = np.zeros([1, no_neurons + 1])

            poly_res.ge[0,y0] = 1
            poly_res.ge[0,y] = -1

            lst_le, lst_ge = poly_res.back_substitute(lst_poly, True)

            assert len(lst_ge) == len(lst_poly)

            if poly_res.lw[0] <= 0:
                check_false = True
                break
    #assert checK_false If violate mean that the model is robust
    if not check_false:
        return None

    poly_i = lst_poly[idx_layer]
    ge_i = lst_ge[idx_layer]
    layer = model.layers[idx_layer]
    func = layer.func

    sum_impact = 0
    impact_lst = []

    for ref_idx in range(len(poly_i.lw)):
        lw = poly_i.lw[ref_idx]
        up = poly_i.up[ref_idx]
        cf = ge_i[ref_idx]

        impact = 0
        if ((func == sigmoid or func == tanh) and lw < up) \
            or (func == relu and lw < 0 and up > 0):
            impact = max(abs(cf * lw), abs(cf * up))
            sum_impact = sum_impact + impact
        impact_lst.append(impact)

    if sum_impact > 0:
        impact_lst = list(map(lambda x: x/sum_impact, impact_lst))
        sort_lst = list(zip(list(range(len(impact_lst))), impact_lst))
        sort_lst.sort(reverse=True, key=lambda x: x[1])
        return [x[0] for x in sort_lst]
    else:
        print("All exact")
        return None


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

    with open(path, 'w') as f:
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
    const[-1] += -(objVal - 0.01)
    re = prove(x0, idx_ly + 1, const, model, lst_poly)
    if objVal <= 0:
        return re
    #while not re and const[-1] < ori:
    #    const[-1] += objVal/10
    #    re = prove(x0, idx_ly + 1, const, model, lst_poly)
    return re

def newApproach(model, x0, models_path):
    lst_poly = getDeepPoly(model, x0, EPS) #May need to get it directly from solver
    #print(sort_layer(model.layers, lst_poly))
    success_lst = []

    for idx_ly in sort_layer(model.layers, lst_poly):
        start_lst = 0
        neuron_lst = sort_neuron(model, lst_poly, x0, idx_ly)
        if neuron_lst is None:
            neuron_lst = list(range(len(lst_poly[idx_ly+1].lw)))
            start_lst = len(neuron_lst) - 1

        save_tf_model(models_path + "temp/temp.tf", model, idx_ly)
        counter, var_list, refinepoly_model = getModel(models_path + "temp/temp.tf", lst_poly[0].lw, lst_poly[0].up) 
        for i in range(start_lst, len(neuron_lst)):
            try:
                const = generate_const(x0, model, idx_ly, lst_poly, neuron_lst[:i+1])
                const = np.around(const, 3)
            except ValueError as err:
                continue
            #const = np.ones(11)
            print("List", neuron_lst)
            print("Const", const)


            res, objVal = checkRefinePoly(counter, var_list, refinepoly_model, const, True)
            if res:
                print("Prove P1 => f")
            else:
                print("Not Prove P1 => f")
            
                #printPlotLine(x0, const, model, idx_ly, lst_poly)
            re = prove(x0, idx_ly + 1, const, model, lst_poly)
            refine = False
            #checkSum(model, x0)
            #test_apply(model)
            if (res and not re) or (not res and re):
                re = refineF(x0, idx_ly, const, model, lst_poly, objVal)
                refine = True
            if not re:
                print("Not prove P2 and f => Property")
            if res and re:
                success_lst.append((idx_ly, i+1, refine))
                return success_lst
    return success_lst

def mnist_challenge(model_path, _range, x_path, y_path=None):
    model_name = str(model_path).split('/')[-2]
    #list_file = []   
    #with open("result_newapproach/{}_refinepoly_failed.csv".format(model_name), 'r') as csvfile: 
    #    list_file_raw = csv.reader(csvfile, delimiter=',')
    #    for test in list_file_raw:
    #        list_file.append((test[0], ast.literal_eval(test[1])))
    
    

    with open(model_path, 'r') as f:
        spec = json.load(f)

    add_assertion(spec, EPS)
    add_solver(spec)
    model, assertion, solver, display = parse(spec)
    list_potential2, unknow2, find_exp2 = [], [], []
    index = 0
    for i in _range:
    #for i in list_file:
        testfileName = i
        pathX = x_path + str(testfileName) + ".txt" #i -> i[0]
        pathY = None if y_path is None else y_path + str(testfileName) + ".txt" #i -> i[0]
        x0s = np.array(ast.literal_eval(read(pathX)))
        y0s = None if y_path is None else np.array(ast.literal_eval(read(pathY)))
        x0s = x0s if len(x0s.shape) == 2 else np.array([x0s])
        list_potential, unknow, find_exp, csv_result = [], [], [], []
        for j in range(x0s.shape[0]):
        #for j in i[1]:
            #print('Index:', index)
            #index += 1
            x0 = x0s[j]
            assertion['x0'] = str(x0.tolist())
            output_x0 = model.apply(x0)
            lbl_x0 = np.argmax(output_x0, axis=1)[0] 
            y0 = y0s[j] if y0s is not None else lbl_x0
            if lbl_x0 != y0:
                print("Skip")
                continue
            start = time.time()
            #res = solver.solve(model, assertion)
            res = refinepoly_run(np.array([x0]), np.array([y0]))
            print("DeepPoly time:", time.time()-start)
            if res == 1 or res == 2:
                print("DeepPoly can prove this model")
                continue
            else:
                print("DeepPoly can't prove this model:", j)
                list_potential.append(j)
                #res = testCegar(model, str(x0.tolist()))
            #if res == 1:
            #    list_potential.append(j)
            #elif res == 2:
            #    continue
            #else:
            #    print("Cegar also can't prove this")
            #    unknow.append(j)
            start = time.time()
            success_lst = newApproach(model, x0, '')
            if success_lst:
                print("Find example:", j)
                find_exp.append((j, success_lst))
                csv_result.append([str(testfileName) + '_' + str(j), time.time()-start, success_lst])
    
        if list_potential:
            list_potential2.append([testfileName,list_potential])
            with open('result_newapproach/{}_refinepoly_failed.csv'.format(model_name), 'a') as f:
                writer = csv.writer(f)
                writer.writerow([testfileName,list_potential])
            
        #unknow2.append((i,unknow))
        find_exp2.append((testfileName, find_exp))
        with open('result_newapproach/{}_approach_cp_refine_result.csv'.format(model_name), 'a') as f:
            writer = csv.writer(f)
            writer.writerows(csv_result)

    print("potential:", list_potential2)
    #print("Unknow:", unknow2)
    print("Example Find:", find_exp2)



def main():
    base_path = Path(__file__).parent.parent
    models_path = base_path / "source/deeppoly_model/"
    #model_path = models_path / "spec_Cegar.json"
    model_path = base_path / "benchmark/cegar/nnet/mnist_relu_4_20/spec.json"
    #[6, 24, 65, 96]
    #PathX = base_path / "benchmark/cegar/data/mnist_fc/data"
    PathX = base_path / "benchmark/mnist_challenge/x_y/x"
    PathY = base_path / "benchmark/mnist_challenge/x_y/y"

    mnist_challenge(model_path, list(range(8, 10)), str(PathX), str(PathY))

    #list_file = []
    #with open("deeppoly_failed.csv", 'r') as csvfile: 
    #    list_file_raw = csv.reader(csvfile, delimiter='\t')
    #    for test in list_file_raw:
    #        list_file.append((test[0], ast.literal_eval(test[1])))
   

    

    

if __name__ == '__main__':
    main()
