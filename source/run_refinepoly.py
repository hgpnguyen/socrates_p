import sys
import os
sys.path.append(os.path.abspath("../eran/tf_verify"))
sys.path.insert(0, os.path.abspath("../eran/ELINA/python_interface/"))
sys.path.insert(0, os.path.abspath("../eran/deepg/code/"))

from utils import *
import numpy as np
from eran import ERAN
from read_net_file import *
from read_zonotope_file import read_zonotope
import tensorflow as tf
import csv
import time
from tqdm import tqdm
from ai_milp import *
import argparse
from config import config
from constraint_utils import *
import re
import itertools
from multiprocessing import Pool, Value
import onnxruntime.backend as rt
import logging
import torch
import spatial
from copy import deepcopy
from tensorflow_translator import *
from onnx_translator import *
from optimizer import *
from analyzer import *
if config.domain=='gpupoly' or config.domain=='refinegpupoly':
    from refine_gpupoly import *

#ZONOTOPE_EXTENSION = '.zt'
EPS = 10**(-9)

is_tf_version_2=tf.__version__[0]=='2'

if is_tf_version_2:
    tf= tf.compat.v1


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def isnetworkfile(fname):
    _, ext = os.path.splitext(fname)
    if ext not in ['.pyt', '.meta', '.tf','.onnx', '.pb']:
        raise argparse.ArgumentTypeError('only .pyt, .tf, .onnx, .pb, and .meta formats supported')
    return fname



def parse_input_box(text):
    intervals_list = []
    for line in text.split('\n'):
        if line!="":
            interval_strings = re.findall("\[-?\d*\.?\d+, *-?\d*\.?\d+\]", line)
            intervals = []
            for interval in interval_strings:
                interval = interval.replace('[', '')
                interval = interval.replace(']', '')
                [lb,ub] = interval.split(",")
                intervals.append((np.double(lb), np.double(ub)))
            intervals_list.append(intervals)

    # return every combination
    boxes = itertools.product(*intervals_list)
    return list(boxes)


def show_ascii_spec(lb, ub, n_rows, n_cols, n_channels):
    print('==================================================================')
    for i in range(n_rows):
        print('  ', end='')
        for j in range(n_cols):
            print('#' if lb[n_cols*n_channels*i+j*n_channels] >= 0.5 else ' ', end='')
        print('  |  ', end='')
        for j in range(n_cols):
            print('#' if ub[n_cols*n_channels*i+j*n_channels] >= 0.5 else ' ', end='')
        print('  |  ')
    print('==================================================================')


def normalize(image, means, stds, dataset):
    # normalization taken out of the network
    if len(means) == len(image):
        for i in range(len(image)):
            image[i] -= means[i]
            if stds!=None:
                image[i] /= stds[i]
    elif dataset == 'mnist'  or dataset == 'fashion':
        for i in range(len(image)):
            image[i] = (image[i] - means[0])/stds[0]
    elif(dataset=='cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = (image[count] - means[0])/stds[0]
            count = count + 1
            tmp[count] = (image[count] - means[1])/stds[1]
            count = count + 1
            tmp[count] = (image[count] - means[2])/stds[2]
            count = count + 1

        
        is_gpupoly = (domain=='gpupoly' or domain=='refinegpupoly')
        if is_conv and not is_gpupoly:
            for i in range(3072):
                image[i] = tmp[i]
            #for i in range(1024):
            #    image[i*3] = tmp[i]
            #    image[i*3+1] = tmp[i+1024]
            #    image[i*3+2] = tmp[i+2048]
        else:
            count = 0
            for i in range(1024):
                image[i] = tmp[count]
                count = count+1
                image[i+1024] = tmp[count]
                count = count+1
                image[i+2048] = tmp[count]
                count = count+1


def normalize_plane(plane, mean, std, channel, is_constant):
    plane_ = plane.clone()

    if is_constant:
        plane_ -= mean[channel]

    plane_ /= std[channel]

    return plane_


def normalize_poly(num_params, lexpr_cst, lexpr_weights, lexpr_dim, uexpr_cst, uexpr_weights, uexpr_dim, means, stds, dataset):
    # normalization taken out of the network
    if dataset == 'mnist' or dataset == 'fashion':
        for i in range(len(lexpr_cst)):
            lexpr_cst[i] = (lexpr_cst[i] - means[0]) / stds[0]
            uexpr_cst[i] = (uexpr_cst[i] - means[0]) / stds[0]
        for i in range(len(lexpr_weights)):
            lexpr_weights[i] /= stds[0]
            uexpr_weights[i] /= stds[0]
    else:
        for i in range(len(lexpr_cst)):
            lexpr_cst[i] = (lexpr_cst[i] - means[i % 3]) / stds[i % 3]
            uexpr_cst[i] = (uexpr_cst[i] - means[i % 3]) / stds[i % 3]
        for i in range(len(lexpr_weights)):
            lexpr_weights[i] /= stds[(i // num_params) % 3]
            uexpr_weights[i] /= stds[(i // num_params) % 3]


def denormalize(image, means, stds, dataset):
    if dataset == 'mnist'  or dataset == 'fashion':
        for i in range(len(image)):
            image[i] = image[i]*stds[0] + means[0]
    elif(dataset=='cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = image[count]*stds[0] + means[0]
            count = count + 1
            tmp[count] = image[count]*stds[1] + means[1]
            count = count + 1
            tmp[count] = image[count]*stds[2] + means[2]
            count = count + 1

        for i in range(3072):
            image[i] = tmp[i]


def model_predict(base, input):
    if is_onnx:
        pred = base.run(input)
    else:
        pred = base.run(base.graph.get_operation_by_name(model.op.name), {base.graph.get_operations()[0].name + ':0': input})
    return pred


def estimate_grads(specLB, specUB, dim_samples=3):
    specLB = np.array(specLB, dtype=np.float32)
    specUB = np.array(specUB, dtype=np.float32)
    inputs = [((dim_samples - i) * specLB + i * specUB) / dim_samples for i in range(dim_samples + 1)]
    diffs = np.zeros(len(specLB))

    # refactor this out of this method
    if is_onnx:
        runnable = rt.prepare(model, 'CPU')
    elif sess is None:
        runnable = tf.Session()
    else:
        runnable = sess

    for sample in range(dim_samples + 1):
        pred = model_predict(runnable, inputs[sample])

        for index in range(len(specLB)):
            if sample < dim_samples:
                l_input = [m if i != index else u for i, m, u in zip(range(len(specLB)), inputs[sample], inputs[sample+1])]
                l_input = np.array(l_input, dtype=np.float32)
                l_i_pred = model_predict(runnable, l_input)
            else:
                l_i_pred = pred
            if sample > 0:
                u_input = [m if i != index else l for i, m, l in zip(range(len(specLB)), inputs[sample], inputs[sample-1])]
                u_input = np.array(u_input, dtype=np.float32)
                u_i_pred = model_predict(runnable, u_input)
            else:
                u_i_pred = pred
            diff = np.sum([abs(li - m) + abs(ui - m) for li, m, ui in zip(l_i_pred, pred, u_i_pred)])
            diffs[index] += diff
    return diffs / dim_samples



progress = 0.0




def get_tests(dataset, geometric):
    if geometric:
        csvfile = open('../deepg/code/datasets/{}_test.csv'.format(dataset), 'r')
    else:
        if config.subset == None:
            csvfile = open('../data/{}_test.csv'.format(dataset), 'r')
        else:
            filename = '../data/'+ dataset+ '_test_' + config.subset + '.csv'
            csvfile = open(filename, 'r')
    tests = csv.reader(csvfile, delimiter=',')

    return tests


def init_domain(d):
    if d == 'refinezono':
        return 'deepzono'
    elif d == 'refinepoly':
        return 'deeppoly'
    else:
        return d

config.netname = 'benchmark/cegar/nnet/mnist_relu_3_20/original/mnist_relu_3_20.tf'
config.dataset = 'mnist'
config.domain = 'refinepoly'
config.epsilon = 0.01
config.use_milp = True
config.complete = True
config.refine_neurons = True
config.sparse_n = 12
config.timeout_lp = 10
config.timeout_milp = 10

if config.specnumber and not config.input_box and not config.output_constraints:
    config.input_box = '../data/acasxu/specs/acasxu_prop_' + str(config.specnumber) + '_input_prenormalized.txt'
    config.output_constraints = '../data/acasxu/specs/acasxu_prop_' + str(config.specnumber) + '_constraints.txt'

assert config.netname, 'a network has to be provided for analysis.'

#if len(sys.argv) < 4 or len(sys.argv) > 5:
#    print('usage: python3.6 netname epsilon domain dataset')
#    exit(1)

netname = config.netname
filename, file_extension = os.path.splitext(netname)

is_trained_with_pytorch = file_extension==".pyt"
is_saved_tf_model = file_extension==".meta"
is_pb_file = file_extension==".pb"
is_tensorflow = file_extension== ".tf"
is_onnx = file_extension == ".onnx"
assert is_trained_with_pytorch or is_saved_tf_model or is_pb_file or is_tensorflow or is_onnx, "file extension not supported"

epsilon = config.epsilon
#assert (epsilon >= 0) and (epsilon <= 1), "epsilon can only be between 0 and 1"

zonotope_file = config.zonotope
zonotope = None
zonotope_bool = (zonotope_file!=None)
if zonotope_bool:
    zonotope = read_zonotope(zonotope_file)

domain = config.domain

if zonotope_bool:
    assert domain in ['deepzono', 'refinezono'], "domain name can be either deepzono or refinezono"
elif not config.geometric:
    assert domain in ['deepzono', 'refinezono', 'deeppoly', 'refinepoly', 'gpupoly', 'refinegpupoly'], "domain name can be either deepzono, refinezono, deeppoly, refinepoly, gpupoly, refinegpupoly"

dataset = config.dataset

if zonotope_bool==False:
   assert dataset in ['mnist', 'cifar10', 'acasxu', 'fashion'], "only mnist, cifar10, acasxu, and fashion datasets are supported"

constraints = None
if config.output_constraints:
    constraints = get_constraints_from_file(config.output_constraints)

mean = 0
std = 0

complete = (config.complete==True)

if(dataset=='acasxu'):
    print("netname ", netname, " specnumber ", config.specnumber, " domain ", domain, " dataset ", dataset, "args complete ", config.complete, " complete ",complete, " timeout_lp ",config.timeout_lp)
else:
    print("netname ", netname, " epsilon ", epsilon, " domain ", domain, " dataset ", dataset, "args complete ", config.complete, " complete ",complete, " timeout_lp ",config.timeout_lp)

non_layer_operation_types = ['NoOp', 'Assign', 'Const', 'RestoreV2', 'SaveV2', 'PlaceholderWithDefault', 'IsVariableInitialized', 'Placeholder', 'Identity']

sess = None
if is_saved_tf_model or is_pb_file:
    netfolder = os.path.dirname(netname)

    tf.logging.set_verbosity(tf.logging.ERROR)

    sess = tf.Session()
    if is_saved_tf_model:
        saver = tf.train.import_meta_graph(netname)
        saver.restore(sess, tf.train.latest_checkpoint(netfolder+'/'))
    else:
        with tf.gfile.GFile(netname, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.graph_util.import_graph_def(graph_def, name='')
    ops = sess.graph.get_operations()
    last_layer_index = -1
    while ops[last_layer_index].type in non_layer_operation_types:
        last_layer_index -= 1
    model = sess.graph.get_tensor_by_name(ops[last_layer_index].name + ':0')

    eran = ERAN(model, sess)

else:
    if(zonotope_bool==True):
        num_pixels = len(zonotope)
    elif(dataset=='mnist'):
        num_pixels = 784
    elif (dataset=='cifar10'):
        num_pixels = 3072
    elif(dataset=='acasxu'):
        num_pixels = 5
    if is_onnx:
        model, is_conv = read_onnx_net(netname)
    else:
        model, is_conv, means, stds = read_tensorflow_net(netname, num_pixels, is_trained_with_pytorch, (domain == 'gpupoly' or domain == 'refinegpupoly'))
    if domain == 'gpupoly' or domain == 'refinegpupoly':
        if is_onnx:
            translator = ONNXTranslator(model, True)
        else:
            translator = TFTranslator(model)
        operations, resources = translator.translate()
        optimizer  = Optimizer(operations, resources)
        nn = layers()
        network, relu_layers, num_gpu_layers = optimizer.get_gpupoly(nn) 
    else:    
        eran = ERAN(model, is_onnx=is_onnx)

if not is_trained_with_pytorch:
    if dataset == 'mnist' and not config.geometric:
        means = [0]
        stds = [1]
    elif dataset == 'acasxu':
        means = [1.9791091e+04,0.0,0.0,650.0,600.0]
        stds = [60261.0,6.28318530718,6.28318530718,1100.0,1200.0]
    else:
        means = [0.5, 0.5, 0.5]
        stds = [1, 1, 1]

is_trained_with_pytorch = is_trained_with_pytorch or is_onnx

if config.mean is not None:
    means = config.mean
    stds = config.std


correctly_classified_images = 0
verified_images = 0


#if dataset:
#    if config.input_box is None:
#        tests = get_tests(dataset, config.geometric)
#    else:
#        tests = open(config.input_box, 'r').read()




def run_refinepoly():
    global correctly_classified_images, verified_images, domain, epsilon
    model_name = config.netname.split('/')[-3] 
    
    PathX = "benchmark/mnist_challenge/x_y/x"
    PathY = "benchmark/mnist_challenge/x_y/y"
    names, mnist_tests, true_labels = getFailedDeepPoly("result_newapproach/{}_deeppoly_failed.csv".format(model_name), PathX, PathY)
    target = []
    if config.target != None:
        targetfile = open(config.target, 'r')
        targets = csv.reader(targetfile, delimiter=',')
        for i, val in enumerate(targets):
            target = val   
   
   
    if config.epsfile != None:
        epsfile = open(config.epsfile, 'r')
        epsilons = csv.reader(epsfile, delimiter=',')
        for i, val in enumerate(epsilons):
            eps_array = val  
    
    verified_safe, verified_unsafe = [], []

    for i in range(mnist_tests.shape[0]):
        image = mnist_tests[i]
        specLB = np.copy(image)
        print(specLB.shape)
        specUB = np.copy(image)
        if config.quant_step:
            specLB = np.round(specLB/config.quant_step)
            specUB = np.round(specUB/config.quant_step)
        #cifarfile = open('/home/gagandeepsi/eevbnn/input.txt', 'r')
        
        #cifarimages = csv.reader(cifarfile, delimiter=',')
        #for _, image in enumerate(cifarimages):
        #    specLB = np.float64(image)
        #specUB = np.copy(specLB)
        normalize(specLB, means, stds, dataset)
        normalize(specUB, means, stds, dataset)


        #print("specLB ", len(specLB), "specUB ", specUB)
        is_correctly_classified = False
        if domain == 'gpupoly' or domain == 'refinegpupoly':
            #specLB = np.reshape(specLB, (32,32,3))#np.ascontiguousarray(specLB, dtype=np.double)
            #specUB = np.reshape(specUB, (32,32,3))
            #print("specLB ", specLB)
            is_correctly_classified = network.test(specLB, specUB, true_labels[i], True)
        else:
            label,nn,nlb,nub,_,_ = eran.analyze_box(specLB, specUB, init_domain(domain), config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
            print("concrete ", nlb[-1])
            if label == true_labels[i]:
                is_correctly_classified = True
        #for number in range(len(nub)):
        #    for element in range(len(nub[number])):
        #        if(nub[number][element]<=0):
        #            print('False')
        #        else:
        #            print('True')
        if config.epsfile!= None:
            epsilon = np.float64(eps_array[i])
        
        #if(label == int(test[0])):
        if is_correctly_classified == True:
            perturbed_label = None
            correctly_classified_images +=1
            if config.normalized_region==True:
                specLB = np.clip(image - epsilon,0,1)
                specUB = np.clip(image + epsilon,0,1)
                normalize(specLB, means, stds, dataset)
                normalize(specUB, means, stds, dataset)
            else:
                specLB = specLB - epsilon
                specUB = specUB + epsilon
            if config.quant_step:
                specLB = np.round(specLB/config.quant_step)
                specUB = np.round(specUB/config.quant_step)
            start = time.time()
            if config.target == None:
                prop = -1
            else:
                prop = int(target[i])
            if domain == 'gpupoly' or domain =='refinegpupoly':
                is_verified = network.test(specLB, specUB, true_labels[i])
                #print("res ", res)
                if is_verified:
                    print("img", i, "Verified", true_labels[i])
                    verified_images+=1
                elif domain == 'refinegpupoly':
                    # Matrix that computes the difference with the expected layer.
                    diffMatrix = np.delete(-np.eye(num_outputs), true_labels[i], 0)
                    diffMatrix[:, label] = 1
                    diffMatrix = diffMatrix.astype(np.float64)
                    
                    # gets the values from GPUPoly.
                    res=network.evalAffineExpr(diffMatrix, back_substitute=network.BACKSUBSTITUTION_WHILE_CONTAINS_ZERO)
                    
                    
                    labels_to_be_verified = []
                    num_outputs = len(nn.weights[-1])
                    var = 0
                    nn.specLB = specLB
                    nn.specUB = specUB
                    nn.predecessors = []
                    
                    for pred in range(0,nn.numlayer+1):
                        predecessor = np.zeros(1, dtype=np.int)
                        predecessor[0] = int(pred-1)
                        nn.predecessors.append(predecessor)
                    #print("predecessors ", nn.predecessors[0][0])
                    for labels in range(num_outputs):
                        #print("num_outputs ", num_outputs, nn.numlayer, len(nn.weights[-1]))
                        if labels != true_labels[i]:
                            if res[var][0] < 0:
                                labels_to_be_verified.append(labels)
                            var = var+1
                    #print("relu layers", relu_layers)
                    is_verified, x = refine_gpupoly_results(nn, network, num_gpu_layers, relu_layers, true_labels[i], labels_to_be_verified)
                    if is_verified:
                        print("img", i, "Verified", true_labels[i])
                        verified_images+=1
                        verified_safe.append([names[i], time.time() - start])
                    else:
                        if x != None:
                            adv_image = np.array(x)
                            res=np.argmax((network.eval(adv_image))[:,0])
                            if res!=true_labels[i]:
                                denormalize(x,means, stds, dataset)
                                print("img", i, "Verified unsafe with adversarial image ", adv_image, "cex label", res, "correct label ", true_labels[i])
                                verified_unsafe.append([names[i], time.time() - start])
                            else:
                                print("img", i, "Failed") 
                else:
                    print("img", i, "Failed")
            else:    
                perturbed_label, _, nlb, nub,failed_labels, x = eran.analyze_box(specLB, specUB, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic,label=label, prop=prop)
                print("nlb ", nlb[-1], " nub ", nub[-1],"adv labels ", failed_labels)
                if(perturbed_label==label):
                    print("img", i, "Verified", label)
                    verified_safe.append([names[i], time.time() - start])
                    verified_images += 1
                else:
                    #if complete==True:
                    if False:
                        constraints = get_constraints_for_dominant_label(label, failed_labels)
                        verified_flag,adv_image = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
                        if(verified_flag==True):
                            print("img", i, "Verified as Safe", label)
                            verified_images += 1
                            verified_safe.append([names[i], time.time() - start])
                        else:
                        
                            if adv_image != None:
                                cex_label,_,_,_,_,_ = eran.analyze_box(adv_image[0], adv_image[0], 'deepzono', config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
                                if(cex_label!=label):
                                    denormalize(adv_image[0], means, stds, dataset)
                                    print("img", i, "Verified unsafe with adversarial image ", adv_image, "cex label", cex_label, "correct label ", label)
                                    #verified_images+=1
                                    verified_unsafe.append([names[i], time.time() - start]) 
                            print("img", i, "Failed")
                    else:
                    
                        if x != None:
                            cex_label,_,_,_,_,_ = eran.analyze_box(x,x,'deepzono',config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
                            print("cex label ", cex_label, "label ", label)
                            if(cex_label!=label):
                                denormalize(x,means, stds, dataset)
                                print("img", i, "Verified unsafe with adversarial image ", x, "cex label ", cex_label, "correct label ", label)
                                #verified_images+=1
                                verified_unsafe.append([names[i], time.time() - start]) 
                            else:
                                print("img", i, "Failed")
                        else:
                            print("img", i, "Failed")

                
                end = time.time()
                print(end - start, "seconds")
        else:
            if domain != "gpupoly" and domain!= "refinegpupoly":
                print("img",i,"not considered, correct_label", true_labels[i], "classified label ", label)

    print('analysis precision ',verified_images,'/ ', correctly_classified_images)
    print('Num safe testcase:', len(verified_safe))
    print('Num unsafe testcase:', len(verified_unsafe))
    with open('result_newapproach/{}_refinepoly_result.csv'.format(model_name), 'w') as f:
        headers = ['Safe testfile', 'time1', 'Unsafe testfile', 'time2']
        writer = csv.writer(f)
        writer.writerow(headers)
        for k in range(max(len(verified_safe), len(verified_unsafe))):
            if k >= len(verified_unsafe):
                temp = verified_safe[k] + ['','']
            elif k >= len(verified_safe):
                temp = ['',''] + verified_unsafe[k]
            else:
                 temp = verified_safe[k] + verified_unsafe[k]
            writer.writerow(temp)
    
def refinepoly_run(mnist_tests, true_labels):
    global correctly_classified_images, verified_images, domain, epsilon
    model_name = config.netname.split('/')[-3] 
    
    target = []
    if config.target != None:
        targetfile = open(config.target, 'r')
        targets = csv.reader(targetfile, delimiter=',')
        for i, val in enumerate(targets):
            target = val   
   
   
    if config.epsfile != None:
        epsfile = open(config.epsfile, 'r')
        epsilons = csv.reader(epsfile, delimiter=',')
        for i, val in enumerate(epsilons):
            eps_array = val  
    
    verified_safe, verified_unsafe = [], []

    for i in range(mnist_tests.shape[0]):
        image = mnist_tests[i]
        specLB = np.copy(image)
        print(specLB.shape)
        specUB = np.copy(image)
        if config.quant_step:
            specLB = np.round(specLB/config.quant_step)
            specUB = np.round(specUB/config.quant_step)
        #cifarfile = open('/home/gagandeepsi/eevbnn/input.txt', 'r')
        
        #cifarimages = csv.reader(cifarfile, delimiter=',')
        #for _, image in enumerate(cifarimages):
        #    specLB = np.float64(image)
        #specUB = np.copy(specLB)
        normalize(specLB, means, stds, dataset)
        normalize(specUB, means, stds, dataset)


        #print("specLB ", len(specLB), "specUB ", specUB)
        is_correctly_classified = False
        if domain == 'gpupoly' or domain == 'refinegpupoly':
            #specLB = np.reshape(specLB, (32,32,3))#np.ascontiguousarray(specLB, dtype=np.double)
            #specUB = np.reshape(specUB, (32,32,3))
            #print("specLB ", specLB)
            is_correctly_classified = network.test(specLB, specUB, true_labels[i], True)
        else:
            label,nn,nlb,nub,_,_ = eran.analyze_box(specLB, specUB, init_domain(domain), config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
            print("concrete ", nlb[-1])
            if label == true_labels[i]:
                is_correctly_classified = True
        #for number in range(len(nub)):
        #    for element in range(len(nub[number])):
        #        if(nub[number][element]<=0):
        #            print('False')
        #        else:
        #            print('True')
        if config.epsfile!= None:
            epsilon = np.float64(eps_array[i])
        
        #if(label == int(test[0])):
        if is_correctly_classified == True:
            perturbed_label = None
            correctly_classified_images +=1
            if config.normalized_region==True:
                specLB = np.clip(image - epsilon,0,1)
                specUB = np.clip(image + epsilon,0,1)
                normalize(specLB, means, stds, dataset)
                normalize(specUB, means, stds, dataset)
            else:
                specLB = specLB - epsilon
                specUB = specUB + epsilon
            if config.quant_step:
                specLB = np.round(specLB/config.quant_step)
                specUB = np.round(specUB/config.quant_step)
            start = time.time()
            if config.target == None:
                prop = -1
            else:
                prop = int(target[i])
            if domain == 'gpupoly' or domain =='refinegpupoly':
                is_verified = network.test(specLB, specUB, true_labels[i])
                #print("res ", res)
                if is_verified:
                    print("img", i, "Verified", true_labels[i])
                    verified_images+=1
                elif domain == 'refinegpupoly':
                    # Matrix that computes the difference with the expected layer.
                    diffMatrix = np.delete(-np.eye(num_outputs), true_labels[i], 0)
                    diffMatrix[:, label] = 1
                    diffMatrix = diffMatrix.astype(np.float64)
                    
                    # gets the values from GPUPoly.
                    res=network.evalAffineExpr(diffMatrix, back_substitute=network.BACKSUBSTITUTION_WHILE_CONTAINS_ZERO)
                    
                    
                    labels_to_be_verified = []
                    num_outputs = len(nn.weights[-1])
                    var = 0
                    nn.specLB = specLB
                    nn.specUB = specUB
                    nn.predecessors = []
                    
                    for pred in range(0,nn.numlayer+1):
                        predecessor = np.zeros(1, dtype=np.int)
                        predecessor[0] = int(pred-1)
                        nn.predecessors.append(predecessor)
                    #print("predecessors ", nn.predecessors[0][0])
                    for labels in range(num_outputs):
                        #print("num_outputs ", num_outputs, nn.numlayer, len(nn.weights[-1]))
                        if labels != true_labels[i]:
                            if res[var][0] < 0:
                                labels_to_be_verified.append(labels)
                            var = var+1
                    #print("relu layers", relu_layers)
                    is_verified, x = refine_gpupoly_results(nn, network, num_gpu_layers, relu_layers, true_labels[i], labels_to_be_verified)
                    if is_verified:
                        print("img", i, "Verified", true_labels[i])
                        verified_images+=1
                        return 1
                    else:
                        if x != None:
                            adv_image = np.array(x)
                            res=np.argmax((network.eval(adv_image))[:,0])
                            if res!=true_labels[i]:
                                denormalize(x,means, stds, dataset)
                                print("img", i, "Verified unsafe with adversarial image ", adv_image, "cex label", res, "correct label ", true_labels[i])
                                return 2
                            else:
                                print("img", i, "Failed") 
                                return 0
                else:
                    print("img", i, "Failed")
                    return 0
            else:    
                perturbed_label, _, nlb, nub,failed_labels, x = eran.analyze_box(specLB, specUB, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic,label=label, prop=prop)
                print("nlb ", nlb[-1], " nub ", nub[-1],"adv labels ", failed_labels)
                if(perturbed_label==label):
                    print("img", i, "Verified", label)
                    verified_images += 1
                    return 1
                else:
                    #if complete==True:
                    if False:
                        constraints = get_constraints_for_dominant_label(label, failed_labels)
                        verified_flag,adv_image = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
                        if(verified_flag==True):
                            print("img", i, "Verified as Safe", label)
                            verified_images += 1
                            return 1
                        else:
                        
                            if adv_image != None:
                                cex_label,_,_,_,_,_ = eran.analyze_box(adv_image[0], adv_image[0], 'deepzono', config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
                                if(cex_label!=label):
                                    denormalize(adv_image[0], means, stds, dataset)
                                    print("img", i, "Verified unsafe with adversarial image ", adv_image, "cex label", cex_label, "correct label ", label)
                                    #verified_images+=1
                                    return 2
                            print("img", i, "Failed")
                            return 0
                    else:
                    
                        if x != None:
                            cex_label,_,_,_,_,_ = eran.analyze_box(x,x,'deepzono',config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
                            print("cex label ", cex_label, "label ", label)
                            if(cex_label!=label):
                                denormalize(x,means, stds, dataset)
                                print("img", i, "Verified unsafe with adversarial image ", x, "cex label ", cex_label, "correct label ", label)
                                #verified_images+=1
                                return 2
                            else:
                                print("img", i, "Failed")
                                return 0
                        else:
                            print("img", i, "Failed")
                            return 0

                
                end = time.time()
                print(end - start, "seconds")
        else:
            if domain != "gpupoly" and domain!= "refinegpupoly":
                print("img",i,"not considered, correct_label", true_labels[i], "classified label ", label)

    print('analysis precision ',verified_images,'/ ', correctly_classified_images)
    print('Num safe testcase:', len(verified_safe))
    print('Num unsafe testcase:', len(verified_unsafe))



if __name__ == '__main__':
    run_refinepoly()
