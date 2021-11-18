#!/usr/bin/env python
# coding: utf-8

# _Last Updated: 11/05/2021_

# ## Greedy Layer-by-layer MIP Training with a Negative Log-Likelihood Loss and Binary Activations

from math import exp
import time
import random
import copy
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import json
import os
import re

def LayerwiseTrainer(x,y):
    
    D = x[0].shape[0]                                                            # dimension of data
    
    # Create a new model
    m = gp.Model("Greedy_Binary")

    # Create variables
    
    alpha = {}
    beta = {}
    h = {}
    g = {}
    z = {}
    omega = {}
    log_softmax = {}
    r = {}

    for k in range(K):
        for j in range(J):
            alpha[(k,j,1)] = m.addVar(lb=w_lb, ub=w_ub, vtype=GRB.CONTINUOUS, name="alpha "+str((k,j,1)))
            
        for d in range(D):
            
            alpha[(d,k,0)] = m.addVar(lb=w_lb, ub=w_ub, vtype=GRB.CONTINUOUS, name="alpha "+str((d,k,0)))
                

        beta[(k,0)] = m.addVar(lb=b_lb, ub=b_ub, vtype=GRB.CONTINUOUS, name="beta "+str((k,0)))
    
    for j in range(J):
        beta[(j,1)] = m.addVar(lb=b_lb, ub=b_ub, vtype=GRB.CONTINUOUS, name="beta "+str((j,1)))
    
    for n in range(N):
        for j in range(J):
            h[(n,j,1)] = m.addVar(lb=-np.inf, vtype=GRB.CONTINUOUS, name="h "+str((n,j,1)))
            
            log_softmax[(n,j)] = m.addVar(lb=-np.inf, vtype=GRB.CONTINUOUS, name="log_softmax "+str((n,j)))
            
            for i in range(J):
                if i == j:
                    continue
                r[(n,i,j)] = m.addVar(vtype=GRB.BINARY, name="r "+str((n,i,j)))
            
        for k in range(K):
            h[(n,k,0)] = m.addVar(vtype=GRB.BINARY, name="h "+str((n,k,0)))
                
            g[(n,k,1)] = m.addVar(vtype=GRB.BINARY, name="g "+str((n,k,1)))
                        
            for j in range(J):
                z[(n,k,j,1)] = m.addVar(lb=w_lb, ub=w_ub, vtype=GRB.CONTINUOUS, name= "z "+str((n,k,j,1)))
        
        omega[n] = m.addVar(lb=-np.inf, vtype=GRB.CONTINUOUS, name='omega '+str(n))
                                
    # Set objective
    m.setObjective(
        sum( sum( y[n][j]*(omega[n] - h[(n,j,1)]) for j in range(J)) for n in range(N)), GRB.MINIMIZE)

     # Add constraints
    
    for n in range(N):
        
        for k in range(K):
            
            M = (D*w_ub*1)+b_ub # initiating M for the 0th layer
                        
            m.addConstr(sum(alpha[(d,k,0)]*x[n][d] for d in range(D)) + beta[(k,0)] 
                        <= (M + epsilon)*h[(n,k,0)], name="C1 Binary Neuron "+str((n,k,0)))
            
            m.addConstr(sum(alpha[(d,k,0)]*x[n][d] for d in range(D)) + beta[(k,0)] 
                        >= epsilon + (-M - epsilon)*(1-h[(n,k,0)]), name="C2 Binary Neuron "+str((n,k,0)))  

            m.addConstr(g[(n,k,1)] == h[(n,k,0)], name="g-def "+str((n,k,1)))

        for k_prime in range(K):
            for j in range(J):
                m.addConstr(z[(n,k_prime,j,1)] <= alpha[(k_prime,j,1)] + (w_ub-w_lb)*(1.0-g[(n,k_prime,1)]), name="z-alpha UB "+str((n,k_prime,j,1))) 
                m.addConstr(z[(n,k_prime,j,1)] >= alpha[(k_prime,j,1)] + (w_lb-w_ub)*(1.0-g[(n,k_prime,1)]), name="z-alpha LB "+str((n,k_prime,j,1))) 
                m.addConstr(z[(n,k_prime,j,1)] <= (w_ub)*g[(n,k_prime,1)], name="z-g UB "+str((n,k_prime,j,1)))
                m.addConstr(z[(n,k_prime,j,1)] >= (w_lb)*g[(n,k_prime,1)], name="z-g LB "+str((n,k_prime,j,1)))
        
        M = (K*w_ub*M)+b_ub # updating M for the last layer
        
        for j in range(J):
            
            m.addConstr(h[(n,j,1)] <= sum(z[(n,k_prime,j,1)] for k_prime in range(K)) + beta[(j,1)], name='C1 Output '+str((n,j,1)))
            m.addConstr(h[(n,j,1)] >= sum(z[(n,k_prime,j,1)] for k_prime in range(K)) + beta[(j,1)], name='C2 Output '+str((n,j,1)))
                                        
            m.addConstr(log_softmax[(n,j)] == h[(n,j,1)] - omega[n], name='log_softmax def '+str((n,j)))

            m.addConstr(omega[n] >= h[(n,j,1)], name='C1 min-max '+str((n,j)))                
      
            for i in range(J):
                if i == j:
                    continue
                m.addConstr(h[(n,i,1)] + h[(n,j,1)] - 2*h[(n,i,1)]
                          <= -0.01 + 2*M*r[(n,i,j)], name="C1 Diverse Output "+str((n,i,j)))
                
                m.addConstr(h[(n,i,1)] + h[(n,j,1)] - 2*h[(n,i,1)]
                          >= 0.01 - 2*M*(1-r[(n,i,j)]), name="C2 Diverse Output "+str((n,i,j)))
                            
    # Optimize model
    m.setParam('OutputFlag', 0)
    m.optimize()
#     m.printQuality()

    output_dict = {}
    
    for v in m.getVars():
#         print('%s %s %g' % (v.varName, "=", np.round(v.x, 3)))
        output_dict[v.varName] = np.round(v.x, 3)
    
    print('Optimal Objective: %g' % m.objVal)
    
    model_params = []
    weights = []
    biases = []
    for k in range(K):
        weights.append([output_dict['alpha '+str((d,k,0))] for d in range(D)])
        biases.append(output_dict['beta '+str((k,0))])
    model_params.append(np.array(weights))
    model_params.append(np.array(biases))    
        
    weights = []
    biases = []
    for j in range(J):
        weights.append([output_dict['alpha '+str((k_prime,j,1))] for k_prime in range(K)])
        biases.append(output_dict['beta '+str((j,1))])
    model_params.append(np.array(weights))
    model_params.append(np.array(biases))
    
    return(m.objVal, output_dict, model_params)

# Run Experiments

if not os.path.exists(os.path.join(".","Greedy_Binary_MIP")):
    os.mkdir(os.path.join(".","Greedy_Binary_MIP"))

for seed in [1,2,3,4,7]:
    
    np.random.seed(seed)

    # Produce Synthetic Data
    
    inputs = [[0,0,0,0,0],
              [1,0,0,0,0],
              [0,0,1,0,0],
              [0,0,0,0,1],
              [1,0,1,0,0],
              [1,0,0,0,1],
              [0,0,1,0,1],
              [1,0,1,0,1], 
              [0,1,0,0,0],
              [1,1,0,0,0],
              [0,1,1,0,0],
              [0,1,0,0,1],
              [1,1,1,0,0],
              [1,1,0,0,1],
              [0,1,1,0,1],
              [1,1,1,0,1],
              [0,0,0,1,0],
              [1,0,0,1,0],
              [0,0,1,1,0],
              [0,0,0,1,1],
              [1,0,1,1,0],
              [1,0,0,1,1],
              [0,0,1,1,1],
              [1,0,1,1,1],
              [0,1,0,1,0],
              [1,1,0,1,0],
              [0,1,1,1,0],
              [0,1,0,1,1],
              [1,1,1,1,0],
              [1,1,0,1,1],
              [0,1,1,1,1],
              [1,1,1,1,1]]

    inputs = np.array([np.array(inputs_i) for inputs_i in inputs])

    xor_label = np.array([0,
                          1,
                          1,
                          1,
                          0,
                          0,
                          0,
                          1,
                          0,
                          1,
                          1,
                          1,
                          0,
                          0,
                          0,
                          1,
                          0,
                          1,
                          1,
                          1,
                          0,
                          0,
                          0,
                          1,
                          0,
                          1,
                          1,
                          1,
                          0,
                          0,
                          0,
                          1])

    train_size = 1000
    test_size = 250

    clean_data = [inputs, xor_label]
    idx = np.random.randint(32,size=train_size+test_size)
    sampled_data = [clean_data[0][idx,:], clean_data[1][idx]]
    negative_labels = (2*sampled_data[1]-1) # convert 0 to -1

    noisy_labels = np.array([labels_i*(2*(torch.bernoulli(torch.tensor(0.9)))-1) for labels_i in negative_labels])
    noisy_labels = (noisy_labels+1)/2    # convert -1 to 0
    noisy_data = [sampled_data[0], noisy_labels]

    # EXPERIMENT 1: VARIABLE LAYERS

    architectures = ['5-5-2','5-5-5-2','5-5-5-5-2','5-5-5-5-5-2','5-5-5-5-5-5-2']

    param_collection = {}
    accuracies = {}

    for architecture in architectures:
        architecture_splitter = []
        for substring in re.finditer('-', architecture):
            architecture_splitter.append(substring.end())

        K = int(architecture[architecture_splitter[-2]:architecture_splitter[-1]-1]) # Number of units in each hidden layer  
        J = int(architecture[architecture_splitter[-1]:])                            # Number of units in the output layer
        L = len(architecture_splitter) - 1                                           # Number of hidden layers

        one_hot_labels = []
        target = np.array(noisy_labels)
        for i in range(len(target)):
            lr = np.arange(J)
            one_hot = (lr==target[i]).astype(np.int)
            one_hot_labels.append(one_hot)

        w_ub, w_lb, b_ub, b_lb = [1,-1,1,-1]
        epsilon = 0.01

        one_hot_noisy_data = []
        for i in range(len(one_hot_labels)):
            one_hot_noisy_data.append([noisy_data[0][i], one_hot_labels[i]])

        train_data = one_hot_noisy_data[:train_size]
        test_data = one_hot_noisy_data[train_size:]

        train_inputs = np.array([train_data[i][0] for i in range(len(train_data))])
        train_labels = np.array([train_data[i][1] for i in range(len(train_data))])

        N = train_inputs.shape[0]                                        # Length of data

        # Optimize by layer
            
        print("\nTraining {0} ({1} layers with {2} units each) at seed {3}".format(architecture,L,K,seed))

        t0 = time.time()

        if L == 1:
        
            MIP_model_params = []

            print("Training layer {0} of {1}".format(0+1, L))

            loss, output, params = LayerwiseTrainer(train_inputs, train_labels)
            MIP_model_params.append(params[0])
            MIP_model_params.append(params[1])

            t1 = time.time()
            print("Took",t1-t0,"seconds")
                
        elif L > 1:
            
            print("Training layer {0} of {1}".format(L,L))
            
            MIP_model_params = list(MIP_model_params[:-2])
            
            new_input = np.array([[output['h '+str((n,k,0))] for k in range(K)] for n in range(N)])
            loss, output, params = LayerwiseTrainer(new_input, train_labels)
            MIP_model_params.append(params[0])
            MIP_model_params.append(params[1])
            
            t_end = time.time()
            print("Took",t_end-t0,"seconds")
            
        MIP_model_params.append(params[2])
        MIP_model_params.append(params[3])    

        print("\nTook {0} seconds to train {1} layers with data of size {2}".format(time.time()-t0, L, train_size))

        # Save params

        MIP_model_params_save = copy.deepcopy(MIP_model_params)
        MIP_model_params = np.array(MIP_model_params, dtype='object')

        for i in range(len(MIP_model_params_save)):
            MIP_model_params_save[i] = MIP_model_params_save[i].tolist()

        param_collection[architecture] = MIP_model_params_save

        # Training Accuray 

        train_inputs = np.array([train_data[i][0] for i in range(len(train_data))])
        train_labels = np.array([train_data[i][1] for i in range(len(train_data))])

        N = train_inputs.shape[0]
        D = train_inputs[0].shape[0]

        correct_pred = 0
        for n in range(N):

            h = np.dot(MIP_model_params[0], train_inputs[n]) + MIP_model_params[1]

            h[h <= 0] = 0
            h[h > 0] = 1

            i = 2
            for l in range(1,L):

                h = np.dot(MIP_model_params[i], h) + MIP_model_params[i+1]

                h[h <= 0] = 0
                h[h > 0] = 1

                i += 2

            h = np.array(np.dot(MIP_model_params[i], h) + MIP_model_params[i+1])

            h_max = np.zeros(J)
            h_max.fill(h.max())
            log_softmax = torch.tensor(h - h_max)
            pred = log_softmax.max(0, keepdim=True)[1].item()
            actual =  torch.tensor(train_labels[n]).max(0, keepdim=True)[1].item()

            if pred == actual:
                correct_pred += 1

        training_acc = correct_pred/(N)*100
        print("\nTraining Accuracy for ", L ,"layers = ", training_acc, "%")

        # Testing Accuracy 

        test_inputs = np.array([test_data[i][0] for i in range(len(test_data))])
        test_labels = np.array([test_data[i][1] for i in range(len(test_data))])

        N = test_inputs.shape[0]
        D = test_inputs[0].shape[0]

        correct_pred = 0
        for n in range(N):

            h = np.dot(MIP_model_params[0], test_inputs[n]) + MIP_model_params[1]

            h[h <= 0] = 0
            h[h > 0] = 1

            i = 2
            for l in range(1,L):

                h = np.dot(MIP_model_params[i], h) + MIP_model_params[i+1]

                h[h <= 0] = 0
                h[h > 0] = 1

                i += 2

            h = np.array(np.dot(MIP_model_params[i], h) + MIP_model_params[i+1])

            h_max = np.zeros(J)
            h_max.fill(h.max())
            log_softmax = torch.tensor(h - h_max)
            pred = log_softmax.max(0, keepdim=True)[1].item()
            actual =  torch.tensor(test_labels[n]).max(0, keepdim=True)[1].item()

            if pred == actual:
                correct_pred += 1

        testing_acc = correct_pred/(N)*100    
        print("\nTesting Accuracy for ", L, "layers = ", testing_acc, "%")

        # Save accuracies

        accuracy_dict = {"Training":training_acc, "Testing":testing_acc}
        accuracies[architecture] = accuracy_dict

        print("Took ", time.time()-t0, "seconds to train", architecture, "at seed ", str(seed))

    with open(os.path.join('Greedy_Binary_MIP','Experiment 1 seed '+ str(seed) + ' accuracies.json'),'w') as f:
        json.dump(accuracies, f)
    with open(os.path.join('Greedy_Binary_MIP','Experiment 1 seed ' + str(seed) + ' params.json'),'w') as f:
        json.dump(param_collection, f)

    # EXPERIMENT 2: VARIABLE UNITS

    architectures = ['5-5-5-5-2', '5-10-10-10-2','5-20-20-20-2','5-30-30-30-2', '5-40-40-40-2', '5-50-50-50-2']

    param_collection = {}
    accuracies = {}

    for architecture in architectures:
        architecture_splitter = []
        for substring in re.finditer('-', architecture):
            architecture_splitter.append(substring.end())

        K = int(architecture[architecture_splitter[-2]:architecture_splitter[-1]-1]) # Number of units in each hidden layer  
        J = int(architecture[architecture_splitter[-1]:])                            # Number of units in the output layer
        L = len(architecture_splitter) - 1                                           # Number of hidden layers

        one_hot_labels = []
        target = np.array(noisy_labels)
        for i in range(len(target)):
            lr = np.arange(J)
            one_hot = (lr==target[i]).astype(np.int)
            one_hot_labels.append(one_hot)

        w_ub, w_lb, b_ub, b_lb = [1,-1,1,-1]
        epsilon = 0.01

        one_hot_noisy_data = []
        for i in range(len(one_hot_labels)):
            one_hot_noisy_data.append([noisy_data[0][i], one_hot_labels[i]])

        train_data = one_hot_noisy_data[:train_size]
        test_data = one_hot_noisy_data[train_size:]

        train_inputs = np.array([train_data[i][0] for i in range(len(train_data))])
        train_labels = np.array([train_data[i][1] for i in range(len(train_data))])

        N = train_inputs.shape[0]                                        # Length of data
        
        # Optimize by layer

        print("Training {0} ({1} units in {2} layers) at seed {3}".format(architecture,K,L,seed))

        t0 = time.time()

        MIP_model_params = []

        print("Training layer {0} of {1}".format(0+1, L))

        loss, output, params = LayerwiseTrainer(train_inputs, train_labels)
        MIP_model_params.append(params[0])
        MIP_model_params.append(params[1])

        t1 = time.time()
        print("Took",t1-t0,"seconds")

        for l in range(1,L):

            print("\nTraining layer {0} of {1}".format(l+1,L))
            t_start = time.time()

            new_input = np.array([[output['h '+str((n,k,0))] for k in range(K)] for n in range(N)])
            loss, output, params = LayerwiseTrainer(new_input, train_labels)
            MIP_model_params.append(params[0])
            MIP_model_params.append(params[1])

            t_end = time.time()
            print("Took",t_end-t_start,"seconds")
        MIP_model_params.append(params[2])
        MIP_model_params.append(params[3])    

        print("\nTook {0} seconds to train {1} layers with data of size {2}".format(time.time()-t0, L, train_size))

        # Save params

        MIP_model_params_save = copy.deepcopy(MIP_model_params)
        MIP_model_params = np.array(MIP_model_params, dtype='object')

        for i in range(len(MIP_model_params_save)):
            MIP_model_params_save[i] = MIP_model_params_save[i].tolist()

        param_collection[architecture] = MIP_model_params_save

        # Training Accuray 

        train_inputs = np.array([train_data[i][0] for i in range(len(train_data))])
        train_labels = np.array([train_data[i][1] for i in range(len(train_data))])

        N = train_inputs.shape[0]
        D = train_inputs[0].shape[0]

        correct_pred = 0
        for n in range(N):

            h = np.dot(MIP_model_params[0], train_inputs[n]) + MIP_model_params[1]

            h[h <= 0] = 0
            h[h > 0] = 1

            i = 2
            for l in range(1,L):

                h = np.dot(MIP_model_params[i], h) + MIP_model_params[i+1]

                h[h <= 0] = 0
                h[h > 0] = 1

                i += 2

            h = np.array(np.dot(MIP_model_params[i], h) + MIP_model_params[i+1])

            h_max = np.zeros(J)
            h_max.fill(h.max())
            log_softmax = torch.tensor(h - h_max)
            pred = log_softmax.max(0, keepdim=True)[1].item()
            actual =  torch.tensor(train_labels[n]).max(0, keepdim=True)[1].item()

            if pred == actual:
                correct_pred += 1

        training_acc = correct_pred/(N)*100
        print("\nTraining Accuracy for ", K ,"units = ", training_acc, "%")

        # Testing Accuracy 

        test_inputs = np.array([test_data[i][0] for i in range(len(test_data))])
        test_labels = np.array([test_data[i][1] for i in range(len(test_data))])

        N = test_inputs.shape[0]
        D = test_inputs[0].shape[0]

        correct_pred = 0
        for n in range(N):

            h = np.dot(MIP_model_params[0], test_inputs[n]) + MIP_model_params[1]

            h[h <= 0] = 0
            h[h > 0] = 1

            i = 2
            for l in range(1,L):

                h = np.dot(MIP_model_params[i], h) + MIP_model_params[i+1]

                h[h <= 0] = 0
                h[h > 0] = 1

                i += 2

            h = np.array(np.dot(MIP_model_params[i], h) + MIP_model_params[i+1])

            h_max = np.zeros(J)
            h_max.fill(h.max())
            log_softmax = torch.tensor(h - h_max)
            pred = log_softmax.max(0, keepdim=True)[1].item()
            actual =  torch.tensor(test_labels[n]).max(0, keepdim=True)[1].item()

            if pred == actual:
                correct_pred += 1

        testing_acc = correct_pred/(N)*100    
        print("Testing Accuracy for ", K, "units = ", testing_acc, "%")

        # Save accuracies

        accuracy_dict = {"Training":training_acc, "Testing":testing_acc}
        accuracies[architecture] = accuracy_dict

        print("Took ", time.time()-t0, "seconds to train", architecture, "at seed ", str(seed))

    with open(os.path.join('Greedy_Binary_MIP','Experiment 2 seed ' + str(seed) + ' accuracies.json'),'w') as f:
        json.dump(accuracies, f)
    with open(os.path.join('Greedy_Binary_MIP','Experiment 2 seed ' + str(seed) + ' params.json'),'w') as f:
        json.dump(param_collection, f)
