#!/usr/bin/env python
# coding: utf-8

# _Last Updated: 11/10/2021_

# ## Greedy Layer-by-layer MIP Training with a Negative Log-Likelihood Loss and ReLU Activations

from math import exp
import time
import copy
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import re

ReLU = nn.ReLU()
LogSoftmax = nn.LogSoftmax(dim=0)

def LayerwiseTrainer(x,y):
    
    McCormick_miss_count = 0
        
    D = x[0].shape[0]                                                            # Dimension of data
#     print("D =",D)
    
    # Create a new model
    m = gp.Model("Greedy_ReLU_MIP")

    # Create variables
    
    alpha = {}
    beta = {}
    h = {}
    h_relu = {}
    z = {}
    omega = {}
    log_softmax = {}
    r = {}
    λ = {}
        
    for k in range(K):
        for j in range(J):
            alpha[(k,j,1)] = m.addVar(lb=w_lb, ub=w_ub, vtype=GRB.CONTINUOUS, name="alpha "+str((k,j,1)))
            
            for p in range(P):
                λ[(k,j,1,p)] = m.addVar(vtype=GRB.BINARY, name="λ "+str((k,j,1,p)))
            
        for d in range(D):
            
            alpha[(d,k,0)] = m.addVar(lb=w_lb, ub=w_ub, vtype=GRB.CONTINUOUS, name="alpha "+str((d,k,0)))                

        beta[(k,0)] = m.addVar(lb=b_lb, ub=b_ub, vtype=GRB.CONTINUOUS, name="beta "+str((k,0)))
    
    for j in range(J):
        beta[(j,1)] = m.addVar(lb=b_lb, ub=b_ub, vtype=GRB.CONTINUOUS, name="beta "+str((j,1)))
    
    for n in range(N):
        for j in range(J):
            h[(n,j,1)] = m.addVar(lb=-np.inf, vtype=GRB.CONTINUOUS, name="h "+str((n,j,1)))
            
            log_softmax[(n,j)] = m.addVar(lb=-np.inf, ub=np.inf, vtype=GRB.CONTINUOUS, name="log_softmax "+str((n,j)))
            
            for i in range(J):
                if i == j:
                    continue
                r[(n,i,j)] = m.addVar(vtype=GRB.BINARY, name="r "+str((n,i,j)))
            
        for k in range(K):
            h[(n,k,0)] = m.addVar(vtype=GRB.BINARY, name="h "+str((n,k,0)))
            
            h_relu[(n,k,0)] = m.addVar(lb=0, ub=np.inf, vtype=GRB.CONTINUOUS, name="h_relu "+str((n,k,0)))
                        
            for j in range(J):
                z[(n,k,j,1)] = m.addVar(lb=w_lb, ub=w_ub, vtype=GRB.CONTINUOUS, name= "z "+str((n,k,j,1)))
        
        omega[n] = m.addVar(lb=-np.inf, vtype=GRB.CONTINUOUS, name='omega '+str(n))
                                
    # Set objective
    m.setObjective(
        sum( sum( y[n][j]*(omega[n] - h[(n,j,1)]) for j in range(J)) for n in range(N)), GRB.MINIMIZE)

     # Add constraints
    
    for n in range(N):
        
        for k in range(K):
            
            M1 = (D*w_ub*1)+b_ub # initiating M for the 0th layer
                        
            m.addConstr(sum(alpha[(d,k,0)]*x[n][d] for d in range(D)) + beta[(k,0)] 
                        <= (M1 + epsilon)*h[(n,k,0)], name="C1 Binary Neuron "+str((n,k,0)))
            
            m.addConstr(sum(alpha[(d,k,0)]*x[n][d] for d in range(D)) + beta[(k,0)] 
                        >= epsilon + (-M1 - epsilon)*(1-h[(n,k,0)]), name="C2 Binary Neuron "+str((n,k,0)))
            
            m.addConstr(h_relu[(n,k,0)] - (sum(alpha[(d,k,0)]*x[n][d] for d in range(D)) + beta[(k,0)]) <= 
                        M1*(1-h[(n,k,0)]), name='C1 ReLU '+str((n,k,0)))
            
            m.addConstr(h_relu[(n,k,0)] - (sum(alpha[(d,k,0)]*x[n][d] for d in range(D)) + beta[(k,0)]) >= 
                       0*(1-h[(n,k,0)]), name='C2 ReLU '+str((n,k,0))) 
            # Here, -M is 0 as the smallest value that h_relu - (w^T x + b) can be is 0
            
            m.addConstr(h_relu[(n,k,0)] <= M1*h[(n,k,0)], name='C3 ReLU '+str((n,k,0)))
            
            m.addConstr(h_relu[(n,k,0)] >= 0*h[(n,k,0)], name='C4 ReLU '+str((n,k,0))) 
            # again, -M is 0 because the lower bound on h_relu is 0
        
        M2 = (K*w_ub*M1)+b_ub # updating M for the last layer
        
        for j in range(J):
            
            m.addConstr(h[(n,j,1)] <= sum(z[(n,k_prime,j,1)] for k_prime in range(K)) + beta[(j,1)], name='C1 Output '+str((n,j,1)))
            m.addConstr(h[(n,j,1)] >= sum(z[(n,k_prime,j,1)] for k_prime in range(K)) + beta[(j,1)], name='C2 Output '+str((n,j,1)))            
            
            m.addConstr(log_softmax[(n,j)] == h[(n,j,1)] - omega[n], name='log_softmax def '+str((n,j)))

            m.addConstr(omega[n] >= h[(n,j,1)], name='C1 min-max '+str((n,j)))
      
            for i in range(J):
                if i == j:
                    continue
                m.addConstr(h[(n,i,1)] + h[(n,j,1)] - 2*h[(n,i,1)]
                          <= -0.01 + 2*M2*r[(n,i,j)], name="C1 Diverse Output "+str((n,i,j)))
                
                m.addConstr(h[(n,i,1)] + h[(n,j,1)] - 2*h[(n,i,1)]
                          >= 0.01 - 2*M2*(1-r[(n,i,j)]), name="C2 Diverse Output "+str((n,i,j)))
                                
            for k_prime in range(K):
                
                for p in range(P):
                    
                    # Big M values
                    
                    if alpha_partitions[p][0] <= 0:
                        McCormick_m1 = -1*M1 - alpha_partitions[p][0]*0
                        McCormick_M2 = 1*M1 - alpha_partitions[p][0]*M1 + alpha_partitions[p][0]*M1
                    else:
                        McCormick_m1 = -1*M1 - alpha_partitions[p][0]*M1
                        McCormick_M2 = 1*M1 - alpha_partitions[p][0]*0 + alpha_partitions[p][0]*M1
                        
                    if alpha_partitions[p][1] <= 0:
                        McCormick_m2 = -1*M1 - alpha_partitions[p][1]*0 + alpha_partitions[p][1]*M1
                        McCormick_M1 = 1*M1 - alpha_partitions[p][1]*M1
                    else:
                        McCormick_m2 = -1*M1 - alpha_partitions[p][1]*M1 + alpha_partitions[p][1]*M1
                        McCormick_M1 = 1*M1 - alpha_partitions[p][1]*0
                                        
                    m.addConstr(z[(n,k_prime,j,1)] >= 
                                    alpha_partitions[p][0]*h_relu[(n,k_prime,0)] + McCormick_m1*(1-λ[(k,j,1,p)]), 
                                name = 'C1 McCormick '+str((n,k_prime,j,1,p)))

                    m.addConstr(z[(n,k_prime,j,1)] >= 
                                    alpha_partitions[p][1]*h_relu[(n,k_prime,0)] + alpha[(k_prime,j,1)]*M1 - alpha_partitions[p][1]*M1 + McCormick_m2*(1-λ[(k,j,1,p)]), 
                                name = 'C2 McCormick '+str((n,k_prime,j,1,p)))

                    m.addConstr(z[(n,k_prime,j,1)] <= 
                                    alpha_partitions[p][1]*h_relu[(n,k_prime,0)] + McCormick_M1*(1-λ[(k,j,1,p)]), 
                                name = 'C3 McCormick '+str((n,k_prime,j,1,p)))

                    m.addConstr(z[(n,k_prime,j,1)] <= 
                                    alpha_partitions[p][0]*h_relu[(n,k_prime,0)] + alpha[(k_prime,j,1)]*M1 - alpha_partitions[p][0]*M1 + McCormick_M2*(1-λ[(k,j,1,p)]), 
                                name = 'C4 McCormick '+str((n,k_prime,j,1,p)))
                    
    for k in range(K):
        
        for j in range(J):
            
            m.addConstr(sum(λ[(k,j,1,p)] for p in range(P)) == 1, name='C5 McCormick '+str((k,j,1)))
            
            m.addConstr(alpha[(k,j,1)] >= sum(alpha_partitions[p][0]*λ[(k,j,1,p)] for p in range(P)), name='C6 McCormick '+str((k,j,1)))
            m.addConstr(alpha[(k,j,1)] <= sum(alpha_partitions[p][1]*λ[(k,j,1,p)] for p in range(P)), name='C7 McCormick '+str((k,j,1)))            
                            
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

def LastLayerTrainer(x,y):
        
    D = x[0].shape[0]                                                            # dimension of data
#     print("D =",D)
    
    # Create a new model
    m = gp.Model("Output_Layer_Trainer")

    # Create variables
    
    alpha = {}
    beta = {}
    h = {}
    omega = {}
    log_softmax = {}
    r = {}

    for d in range(D):
        for j in range(J):
            alpha[(d,j,0)] = m.addVar(lb=w_lb, ub=w_ub, vtype=GRB.CONTINUOUS, name="alpha "+str((d,j,0)))

    for j in range(J):      
        beta[(j,0)] = m.addVar(lb=b_lb, ub=b_ub, vtype=GRB.CONTINUOUS, name="beta "+str((j,0)))
        
    for n in range(N):
        for j in range(J):
            h[(n,j,0)] = m.addVar(lb=-np.inf, vtype=GRB.CONTINUOUS, name="h "+str((n,j,0)))
            
            log_softmax[(n,j)] = m.addVar(lb=-np.inf, ub=np.inf, vtype=GRB.CONTINUOUS, name="log_softmax "+str((n,j)))
            
            for i in range(J):
                if i == j:
                    continue
                r[(n,i,j)] = m.addVar(vtype=GRB.BINARY, name="r "+str((n,i,j)))
                    
        omega[n] = m.addVar(lb=-np.inf, vtype=GRB.CONTINUOUS, name='omega '+str(n))
                                
    # Set objective
    m.setObjective(
        sum( sum( y[n][j]*(omega[n] - h[(n,j,0)]) for j in range(J)) for n in range(N)), GRB.MINIMIZE)

     # Add constraints
    
    for n in range(N):
        
        for j in range(J):
            
            M = (D*w_ub*1)+b_ub # initiating M for the 0th layer
                        
            m.addConstr(sum(alpha[(d,j,0)]*x[n][d] for d in range(D)) + beta[(j,0)] 
                        <= h[(n,j,0)], name="C1 Output Neuron "+str((n,j,0)))
            
            m.addConstr(sum(alpha[(d,j,0)]*x[n][d] for d in range(D)) + beta[(j,0)] 
                        >= h[(n,j,0)], name="C2 Output Neuron "+str((n,j,0)))
                                
            m.addConstr(log_softmax[(n,j)] == h[(n,j,0)] - omega[n], name='log_softmax def '+str((n,j)))

            m.addConstr(omega[n] >= h[(n,j,0)], name='C1 min-max '+str((n,j)))
      
            for i in range(J):
                if i == j:
                    continue
                m.addConstr(h[(n,i,0)] + h[(n,j,0)] - 2*h[(n,i,0)]
                          <= -0.01 + 2*M*r[(n,i,j)], name="C1 Diverse Output "+str((n,i,j)))
                
                m.addConstr(h[(n,i,0)] + h[(n,j,0)] - 2*h[(n,i,0)]
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
    for j in range(J):
        weights.append([output_dict['alpha '+str((d,j,0))] for d in range(D)])
        biases.append(output_dict['beta '+str((j,0))])
    model_params.append(np.array(weights))
    model_params.append(np.array(biases))
    
    return(m.objVal, output_dict, model_params)


# Run Experiments

if not os.path.exists(os.path.join(".","Greedy_ReLU_MIP")):
    os.mkdir(os.path.join(".","Greedy_ReLU_MIP"))

for seed in [1,2,3,4,7]:
    
    P = 10                              # Number of McCormick Partitions
    
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

    architectures = ['5-5-2','5-5-5-2']
    # '5-5-2','5-5-5-2','5-5-5-5-2','5-5-5-5-5-2','5-5-5-5-5-5-2'

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
        
        # Make partitions for Piecewise McCormick

        alpha_partitions = []
        for p in range(1,P+1):
            alpha_partitions.append([w_lb+(w_ub-w_lb)*((p-1)/P), w_lb+(w_ub-w_lb)*(p/P)])

        
        train_data = one_hot_noisy_data[:train_size]
        test_data = one_hot_noisy_data[train_size:]

        train_inputs = torch.Tensor([train_data[i][0] for i in range(len(train_data))])
        train_labels = torch.Tensor([train_data[i][1] for i in range(len(train_data))])

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
            
            new_input = np.array([[output['h_relu '+str((n,k,0))] for k in range(K)] for n in range(N)])
            loss, output, params = LayerwiseTrainer(new_input, train_labels)
            MIP_model_params.append(params[0])
            MIP_model_params.append(params[1])
            
            t_end = time.time()
            print("Took",t_end-t0,"seconds")
            
        t_last = time.time()
    
        penultimate_relu = [np.array([output['h_relu '+str((n,k,0))] for k in range(K)]) for n in range(N)]
        np.array(penultimate_relu)

        print("\nTraining Output layer")

        loss, output_last, params = LastLayerTrainer(penultimate_relu, train_labels)
        MIP_model_params.append(params[0])
        MIP_model_params.append(params[1])

        t_L = time.time()
        print("Took",t_L-t_last,"seconds")    

        print("\nTook {0} seconds to train {1} layers with data of size {2}".format(time.time()-t0, L, train_size))

        # Save params

        MIP_model_params_save = copy.deepcopy(MIP_model_params)

        for i in range(len(MIP_model_params_save)):
            MIP_model_params_save[i] = MIP_model_params_save[i].tolist()

        param_collection[architecture] = MIP_model_params_save
        
        for i in range(len(MIP_model_params)):
            MIP_model_params[i] = torch.Tensor(MIP_model_params[i])
        
        # Training Accuray 

        train_inputs = torch.Tensor([train_data[i][0] for i in range(len(train_data))])
        train_labels = torch.Tensor([train_data[i][1] for i in range(len(train_data))])

        N = train_inputs.shape[0]
        D = train_inputs[0].shape[0]   
        
        correct_pred = 0
        for n in range(N):

            h = torch.matmul(MIP_model_params[0], train_inputs[n]) + MIP_model_params[1].data

            h = ReLU(h)

            i = 2
            for l in range(1,L):

                h = torch.matmul(MIP_model_params[i], h) + MIP_model_params[i+1].data

                h = ReLU(h)

                i += 2

            h = np.array(torch.matmul(MIP_model_params[i], h) + MIP_model_params[i+1].data)

            h_max = np.zeros(J)
            h_max.fill(h.max())
            log_softmax = torch.tensor(h - h_max)
            pred = log_softmax.max(0, keepdim=True)[1].item()
            actual = torch.tensor(train_labels[n]).max(0, keepdim=True)[1].item()

            if pred == actual:
                correct_pred += 1

        training_acc = correct_pred/(N)*100
        print("\nTraining Accuracy for ", L ,"layers = ", training_acc, "%")

        # Testing Accuracy 

        test_inputs = torch.Tensor([test_data[i][0] for i in range(len(test_data))])
        test_labels = torch.Tensor([test_data[i][1] for i in range(len(test_data))])

        N = test_inputs.shape[0]
        D = test_inputs[0].shape[0]

        correct_pred = 0
        for n in range(N):

            h = torch.matmul(MIP_model_params[0], test_inputs[n]) + MIP_model_params[1].data

            h = ReLU(h)

            i = 2
            for l in range(1,L):

                h = torch.matmul(MIP_model_params[i], h) + MIP_model_params[i+1].data

                h = ReLU(h)

                i += 2

            h = np.array(torch.matmul(MIP_model_params[i], h) + MIP_model_params[i+1].data)

            h_max = np.zeros(J)
            h_max.fill(h.max())
            log_softmax = torch.tensor(h - h_max)
            pred = log_softmax.max(0, keepdim=True)[1].item()
            actual =  torch.tensor(test_labels[n]).max(0, keepdim=True)[1].item()

            if pred == actual:
                correct_pred += 1

        testing_acc = correct_pred/(N)*100    
        print("Testing Accuracy for ", L, "layers = ", testing_acc, "%")

        # Save accuracies

        accuracy_dict = {"Training":training_acc, "Testing":testing_acc}
        accuracies[architecture] = accuracy_dict

        print("Took ", time.time()-t0, "seconds to train", architecture, "at seed ", str(seed))

    with open(os.path.join('Greedy_ReLU_MIP','Experiment 1 seed ' + str(seed) + ' params.json'),'w') as f:
            json.dump(accuracies, f)
    with open(os.path.join('Greedy_ReLU_MIP','Experiment 1 seed ' + str(seed) + ' params.json'),'w') as f:
        json.dump(param_collection, f)

    # EXPERIMENT 2: VARIABLE UNITS

    architectures = ['5-5-5-5-2', '5-10-10-10-2','5-20-20-20-2','5-30-30-30-2', '5-40-40-40-2', '5-50-50-50-2']
    # '5-5-5-5-2', '5-10-10-10-2','5-20-20-20-2','5-30-30-30-2', '5-40-40-40-2', '5-50-50-50-2'

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
                
        # Make partitions for Piecewise McCormick

        alpha_partitions = []
        for p in range(1,P+1):
            alpha_partitions.append([w_lb+(w_ub-w_lb)*((p-1)/P), w_lb+(w_ub-w_lb)*(p/P)])

        
        train_data = one_hot_noisy_data[:train_size]
        test_data = one_hot_noisy_data[train_size:]

        train_inputs = torch.Tensor([train_data[i][0] for i in range(len(train_data))])
        train_labels = torch.Tensor([train_data[i][1] for i in range(len(train_data))])

        N = train_inputs.shape[0]                                        # Length of data
        
        # Skip some large units due to time and resource constraints
        
        if architecture in ['5-10-10-10-2','5-20-20-20-2','5-30-30-30-2', '5-40-40-40-2', '5-50-50-50-2']:
            
            accuracy_dict = {"Training":None, "Testing":None}
            accuracies[architecture] = accuracy_dict
            
            continue

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

            new_input = np.array([[output['h_relu '+str((n,k,0))] for k in range(K)] for n in range(N)])
            loss, output, params = LayerwiseTrainer(new_input, train_labels)
            MIP_model_params.append(params[0])
            MIP_model_params.append(params[1])

            t_end = time.time()
            print("Took",t_end-t_start,"seconds")
            
        penultimate_relu = [np.array([output['h_relu '+str((n,k,0))] for k in range(K)]) for n in range(N)]
        np.array(penultimate_relu)

        print("\nTraining Output Layer")

        loss, output_last, params = LastLayerTrainer(penultimate_relu, train_labels)
        MIP_model_params.append(params[0])
        MIP_model_params.append(params[1])

        t_L = time.time()
        print("Took",t_L-t_end,"seconds")

        print("\nTook {0} seconds to train {1} layers with data of size {2}".format(time.time()-t0, L, train_size))

        # Save params

        MIP_model_params_save = copy.deepcopy(MIP_model_params)
        
        for i in range(len(MIP_model_params_save)):
            MIP_model_params_save[i] = MIP_model_params_save[i].tolist()

        param_collection[architecture] = MIP_model_params_save
        
        for i in range(len(MIP_model_params)):
        
            MIP_model_params[i] = torch.Tensor(MIP_model_params[i])

        # Training Accuray 

        train_inputs = torch.Tensor([train_data[i][0] for i in range(len(train_data))])
        train_labels = torch.Tensor([train_data[i][1] for i in range(len(train_data))])

        N = train_inputs.shape[0]
        D = train_inputs[0].shape[0]

        correct_pred = 0
        for n in range(N):

            h = torch.matmul(MIP_model_params[0], train_inputs[n]) + MIP_model_params[1].data

            h = ReLU(h)

            i = 2
            for l in range(1,L):

                h = torch.matmul(MIP_model_params[i], h) + MIP_model_params[i+1].data

                h = ReLU(h)

                i += 2

            h = np.array(torch.matmul(MIP_model_params[i], h) + MIP_model_params[i+1].data)

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

        test_inputs = torch.Tensor([test_data[i][0] for i in range(len(test_data))])
        test_labels = torch.Tensor([test_data[i][1] for i in range(len(test_data))])

        N = test_inputs.shape[0]
        D = test_inputs[0].shape[0]

        correct_pred = 0
        for n in range(N):

            h = torch.matmul(MIP_model_params[0], test_inputs[n]) + MIP_model_params[1].data

            h = ReLU(h)

            i = 2
            for l in range(1,L):

                h = torch.matmul(MIP_model_params[i], h) + MIP_model_params[i+1].data

                h = ReLU(h)

                i += 2

            h = np.array(torch.matmul(MIP_model_params[i], h) + MIP_model_params[i+1].data)

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

    with open(os.path.join('Greedy_ReLU_MIP','Experiment 2 seed ' + str(seed) + ' params.json'),'w') as f:
            json.dump(accuracies, f)
    with open(os.path.join('Greedy_ReLU_MIP','Experiment 2 seed ' + str(seed) + ' params.json'),'w') as f:
        json.dump(param_collection, f)
        