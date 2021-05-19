import sys
import ast
from os import path
sys.path.append(path.abspath("../eran/tf_verify"))
sys.path.insert(0, path.abspath("../eran/ELINA/python_interface/"))
sys.path.insert(0, path.abspath("../eran/deepg/code/"))

from eran import ERAN
from pathlib import Path
from read_net_file import *
from config import config
from analyzer import layers, Analyzer
from ai_milp import create_model
import tensorflow as tf
from gurobipy import *
import matplotlib.pyplot as plt
import utils as utl

is_tf_version_2=tf.__version__[0]=='2'

if is_tf_version_2:
    tf= tf.compat.v1


def getERAN(netname, num_pixels):
    is_trained_with_pytorch = True
    domain = 'refinepoly'
    model, is_conv, means, stds = read_tensorflow_net(netname, num_pixels, is_trained_with_pytorch, (domain == 'gpupoly' or domain == 'refinegpupoly'))
    eran = ERAN(model, is_onnx=False)
    return eran

def getAnalyzer(netname, spec_lb, spec_ub):
    eran = getERAN(netname, spec_lb.size)

    specLB = np.reshape(spec_lb, (-1,))
    specUB = np.reshape(spec_ub, (-1,))

    nn = layers()
    nn.specLB = specLB
    nn.specUB = specUB
    domain = "refinepoly"

    execute_list, output_info = eran.optimizer.get_deeppoly(nn, specLB, specUB, None, None, None, None, None, None, 0)

    analyzer = Analyzer(execute_list, nn, domain, config.timeout_lp, config.timeout_milp, None, config.use_default_heuristic, -1, -1)
    return analyzer

def getModel(netname, specLB, specUB):
    config.complete = False
    config.refine_neurons = True
    config.sparse_n = 2
    config.timeout_milp = 10
    config.timeout_lp = 10
    analyzer = getAnalyzer(netname, specLB, specUB)
    element, nlb, nub = analyzer.get_abstract0()
    output_size = 0
    output_size = analyzer.ir_list[-1].output_length
    analyzer.nn.ffn_counter = 0
    analyzer.nn.conv_counter = 0
    analyzer.nn.pool_counter = 0
    analyzer.nn.concat_counter = 0
    analyzer.nn.tile_counter = 0
    analyzer.nn.residual_counter = 0
    analyzer.nn.activation_counter = 0
    counter, var_list, model = create_model(analyzer.nn, analyzer.nn.specLB, analyzer.nn.specUB, nlb, nub, analyzer.relu_groups, analyzer.nn.numlayer, config.complete==True)
    if config.complete==True:
        model.setParam(GRB.Param.TimeLimit,config.timeout_milp)
    else:
        model.setParam(GRB.Param.TimeLimit,config.timeout_lp)
    
    return counter, var_list, model


def checkRefinePoly(counter, var_list, model, invariant, printModel=False):
    num_var = len(var_list)
    output_size = num_var - counter
    assert output_size + 1 == len(invariant), "The number of coeffecient don't match number of neuron"
    
    obj = LinExpr()
    for idx in range(output_size):
        obj += invariant[idx] * var_list[counter + idx]
    obj += invariant[-1] #Constant
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    if printModel:
        #model.write("model.lp")
        print("objVal:", model.objVal)
    return model.objVal >= 0, model.objVal



def main():
    base_path = Path(__file__).parent.parent
    #netname = base_path / "benchmark/cegar/nnet/mnist_relu_3_10/original/mnist_relu_3_10.tf"
    netname = base_path / "source/deeppoly_model/nets/relu_2_2.tf"
    eran = getERAN(netname)
    #PathX = base_path / "benchmark/cegar/data/mnist_fc/data5.txt"
    #x0 = np.array(ast.literal_eval(utl.read(PathX)))

    spec_lb = np.full(2, -1)
    spec_ub = np.ones(2)
    #if checkRefinePoly(netname, spec_lb, spec_ub, np.array([1, 0, -4]), True):
    #    print("ok")
    return

if __name__ == '__main__':
    main()