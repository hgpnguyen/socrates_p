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
import utils as utl

is_tf_version_2=tf.__version__[0]=='2'

if is_tf_version_2:
    tf= tf.compat.v1

num_pixels = 2

def getERAN(netname):
    is_trained_with_pytorch = True
    domain = 'refinepoly'
    model, is_conv, means, stds = read_tensorflow_net(netname, num_pixels, is_trained_with_pytorch, (domain == 'gpupoly' or domain == 'refinegpupoly'))
    print("means: ", means)
    print("stds: ", stds)
    eran = ERAN(model, is_onnx=False)
    return eran

def getAnalyzer(netname):
    eran = getERAN(netname)
    spec_lb = np.full(2, -1)
    spec_ub = np.ones(2)
    specLB = np.reshape(spec_lb, (-1,))
    specUB = np.reshape(spec_ub, (-1,))

    nn = layers()
    nn.specLB = specLB
    nn.specUB = specUB
    domain = "refinepoly"

    execute_list, output_info = eran.optimizer.get_deeppoly(nn, specLB, specUB, None, None, None, None, None, None, 0)
    #Remove after automatic generate model 
    execute_list.pop()
    nn.numlayer -= 1
    #
    print(nn.specLB)
    print("Num Layer:", len(execute_list))
    analyzer = Analyzer(execute_list, nn, domain, config.timeout_lp, config.timeout_milp, None, config.use_default_heuristic, -1, -1)
    return analyzer

def getModel(analyer):
    element, nlb, nub = analyer.get_abstract0()
    output_size = 0
    output_size = analyer.ir_list[-1].output_length
    analyer.nn.ffn_counter = 0
    analyer.nn.conv_counter = 0
    analyer.nn.pool_counter = 0
    analyer.nn.concat_counter = 0
    analyer.nn.tile_counter = 0
    analyer.nn.residual_counter = 0
    analyer.nn.activation_counter = 1
    counter, var_list, model = create_model(analyer.nn, analyer.nn.specLB, analyer.nn.specUB, nlb, nub, analyer.relu_groups, analyer.nn.numlayer, config.complete==True)
    if config.complete==True:
        model.setParam(GRB.Param.TimeLimit,config.timeout_milp)
    else:
        model.setParam(GRB.Param.TimeLimit,config.timeout_lp)
    
    return counter, var_list, model


def checkRefinePoly(netname, specLB, specUB, invariant, printModel=False):
    config.complete = True
    analyzer = getAnalyzer(netname)
    counter, var_list, model = getModel(analyzer)
    num_var = len(var_list)
    output_size = num_var - counter
    assert output_size == len(invariant) + 1, "The number of coeffecient don't match number of neuron"
    
    obj = LinExpr()
    for idx in range(output_size):
        obj += invariant[idx] * var_list[counter + idx]
    obj += invariant[-1] #Constant
    model.setObjective(obj, GRB.MAXIMIZE)
    if printModel:
        model.write("model.lp")
    return model.objVal <= 0





def main():
    base_path = Path(__file__).parent.parent
    #netname = base_path / "benchmark/cegar/nnet/mnist_relu_3_10/original/mnist_relu_3_10.tf"
    netname = base_path / "source/deeppoly_model/nets/relu_2_2.tf"
    eran = getERAN(netname)
    PathX = base_path / "benchmark/cegar/data/mnist_fc/data5.txt"
    x0 = np.array(ast.literal_eval(utl.read(PathX)))

    #spec_lb = np.copy(x0)
    #spec_ub = np.copy(x0)
    spec_lb = np.full(2, -1)
    spec_ub = np.ones(2)

    #label, nn, nlb, nub, _, _ = eran.analyze_box(spec_lb, spec_ub, 'refinepoly', config.timeout_lp, config.timeout_milp,
    #                                               config.use_default_heuristic)
    #config.complete = True


    analyzer = getAnalyzer(netname)
    #dominant_class, nlb, nub, failed_labels, x = analyzer.analyze()
    #print("upper:", nub)
    #print("lower:", nlb)
    getModel(analyzer)

    return

if __name__ == '__main__':
    main()