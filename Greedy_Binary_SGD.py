#!/usr/bin/env python
# coding: utf-8

# _Last Updated: 11/03/2021_

# ## Greedy Layer-by-layer Stochastic Gradient Descent Training with a Negative Log-Likelihood Loss and Binary Activations

from math import exp
import time
import copy
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, ConcatDataset, Subset
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import json
import os
import re

LogSoftmax = nn.LogSoftmax(dim=0)

def SGD_Trainer(X,Y):
    class XOR(nn.Module):
        def __init__(self, input_dim = D, output_dim=J):
            super(XOR, self).__init__()
            self.layer0 = nn.Linear(input_dim, K)
            self.output = nn.Linear(K, output_dim)
            self.LogSoftmax = nn.LogSoftmax(dim = 0)

        def forward(self, x):
            x = self.layer0(x)
            x = torch.sign(x)
            x[x <= 0] = 0
            x = self.output(x)
            return self.LogSoftmax(x)
    
    model = XOR()
    loss_fn = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    epochs = 10000
    for i in range(epochs):
        
        output = model(X)
        loss = loss_fn(output, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 2500 == 0:
            print("\tEpoch: {}, Loss: {}, ".format(i, loss.item()))

    model_params = list(model.parameters())    
    return model_params

# Run Experiments

if not os.path.exists(os.path.join(".","Greedy_Binary_SGD")):
    os.mkdir(os.path.join(".","Greedy_Binary_SGD"))

for seed in [1,2,3,4,7]:

    torch.manual_seed(seed) 
    np.random.seed(seed)

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

    noisy_data = []
    for i in range(len(noisy_labels)):
        noisy_data.append([sampled_data[0][i], noisy_labels[i]])

    # EXPERIMENT 1: VARIABLE LAYERS

    architectures = ['5-5-2','5-5-5-2','5-5-5-5-2','5-5-5-5-5-2','5-5-5-5-5-5-2']

    param_collection = {}
    accuracies = {}

    for architecture in architectures:

        t0 = time.time()

        architecture_splitter = []
        for substring in re.finditer('-', architecture):
            architecture_splitter.append(substring.end())

        K = int(architecture[architecture_splitter[-2]:architecture_splitter[-1]-1]) # Number of units in each hidden layer  
        J = int(architecture[architecture_splitter[-1]:])                            # Number of units in the output layer
        L = len(architecture_splitter) - 1                                           # Number of hidden layers

        train_data = noisy_data[:train_size]
        test_data = noisy_data[train_size:]

        train_inputs = torch.Tensor([train_data[i][0] for i in range(len(train_data))])
        train_labels = torch.Tensor([int(train_data[i][1]) for i in range(len(train_data))])
        train_labels = train_labels.type(torch.LongTensor) # convert from float to long


        N = train_inputs.shape[0]                                        # Length of data
        D = train_inputs[0].shape[0]                                     # Dimension of data

        # Optimize

        print("Training {0} ({1} layer(s) with {2} units each) at seed {3}".format(architecture,L,K,seed))

        print("Training layer", 0)
        SGD_model_params = []
        iteration_model_params = SGD_Trainer(train_inputs, train_labels)
        SGD_model_params += (iteration_model_params[:2])

        new_train_inputs = []
        for n in range(N):
            new_train_inputs_n = torch.matmul(iteration_model_params[0].data, train_inputs[n]) + iteration_model_params[1].T.data
            new_train_inputs_n = torch.sign(new_train_inputs_n)
            new_train_inputs_n[new_train_inputs_n <= 0] = 0

            new_train_inputs.append(new_train_inputs_n.numpy())
        new_train_inputs =  torch.Tensor(new_train_inputs)

        for l in range(1,L):
            print("Training layer", l)
            D = new_train_inputs[0].shape[0]                                                            # Dimension of data
            iteration_model_params = SGD_Trainer(new_train_inputs, train_labels)
            SGD_model_params += (iteration_model_params[:2])

            old_train_inputs = new_train_inputs
            new_train_inputs = []
            for n in range(N):
                new_train_inputs_n = torch.matmul(iteration_model_params[0].data, old_train_inputs[n]) + iteration_model_params[1].T.data
                new_train_inputs_n = torch.sign(new_train_inputs_n)
                new_train_inputs_n[new_train_inputs_n <= 0] = 0
                new_train_inputs.append(new_train_inputs_n.numpy())
            new_train_inputs =  torch.Tensor(new_train_inputs)
        SGD_model_params += iteration_model_params[2:]

        # Save params

        SGD_model_params_save = copy.deepcopy(SGD_model_params)
        SGD_model_params = np.array(SGD_model_params, dtype='object')

        for i in range(len(SGD_model_params_save)):
            SGD_model_params_save[i] = SGD_model_params_save[i].tolist()

        param_collection[architecture] = SGD_model_params_save

        # Training Accuracy

        train_inputs = torch.Tensor([train_data[i][0] for i in range(len(train_data))])
        train_labels = torch.Tensor([train_data[i][1] for i in range(len(train_data))])

        N = train_inputs.shape[0]
        D = train_inputs[0].shape[0]

        correct_pred = 0
        for n in range(N):

            h = torch.matmul(SGD_model_params[0], train_inputs[n]) + SGD_model_params[1].data

            h = torch.sign(h)
            h[h <= 0] = 0    

            i = 2
            for l in range(1,L):
                h = torch.matmul(SGD_model_params[i], h) + SGD_model_params[i+1].data

                h = torch.sign(h)
                h[h <= 0] = 0

                i += 2

            h = torch.matmul(SGD_model_params[i], h) + SGD_model_params[i+1].data

            log_softmax = LogSoftmax(h)
            pred = log_softmax.max(0, keepdim=True)[1].item()
            actual =  int(train_labels[n].item())

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

            h = torch.matmul(SGD_model_params[0], test_inputs[n]) + SGD_model_params[1].data

            h = torch.sign(h)
            h[h <= 0] = 0

            i = 2
            for l in range(1,L):

                h = torch.matmul(SGD_model_params[i], h) + SGD_model_params[i+1].data

                h = torch.sign(h)
                h[h <= 0] = 0

                i += 2

            h = torch.matmul(SGD_model_params[i], h) + SGD_model_params[i+1].data

            log_softmax = LogSoftmax(h)
            pred = log_softmax.max(0, keepdim=True)[1].item()
            actual =  int(test_labels[n].item())

            if pred == actual:
                correct_pred += 1

        testing_acc = correct_pred/(N)*100    
        print("Testing Accuracy for ", L, "layers = ", testing_acc, "%")

        # Save accuracies

        accuracy_dict = {"Training":training_acc, "Testing":testing_acc}
        accuracies[architecture] = accuracy_dict

        print("Took {0} seconds to train {1} \n".format(time.time()-t0, architecture))

    with open(os.path.join('Greedy_Binary_SGD','Experiment 1 seed '+ str(seed) + ' accuracies.json'),'w') as f:
        json.dump(accuracies, f)
    with open(os.path.join('Greedy_Binary_SGD','Experiment 1 seed ' + str(seed) + ' params.json'),'w') as f:
        json.dump(param_collection, f)


    # EXPERIMENT 2: VARIABLE UNITS

    architectures = ['5-5-5-5-2', '5-10-10-10-2','5-20-20-20-2','5-30-30-30-2', '5-40-40-40-2', '5-50-50-50-2']
    # '5-5-5-5-2', '5-10-10-10-2','5-20-20-20-2','5-30-30-30-2', '5-40-40-40-2', '5-50-50-50-2'

    param_collection = {}
    accuracies = {}

    for architecture in architectures:

        t0 = time.time()

        architecture_splitter = []
        for substring in re.finditer('-', architecture):
            architecture_splitter.append(substring.end())

        K = int(architecture[architecture_splitter[-2]:architecture_splitter[-1]-1]) # Number of units in each hidden layer  
        J = int(architecture[architecture_splitter[-1]:])                            # Number of units in the output layer
        L = len(architecture_splitter) - 1                                           # Number of hidden layers

        train_data = noisy_data[:train_size]
        test_data = noisy_data[train_size:]

        train_inputs = torch.Tensor([train_data[i][0] for i in range(len(train_data))])
        train_labels = torch.Tensor([int(train_data[i][1]) for i in range(len(train_data))])
        train_labels = train_labels.type(torch.LongTensor) # convert from float to long


        N = train_inputs.shape[0]                                        # Length of data
        D = train_inputs[0].shape[0]                                     # Dimension of data

        # Optimize

        print("Training {0} ({1} units in {2} layers each) at seed {3}".format(architecture,K,L,seed))

        print("Training layer",0)
        SGD_model_params = []
        iteration_model_params = SGD_Trainer(train_inputs, train_labels)
        SGD_model_params += (iteration_model_params[:2])

        new_train_inputs = []
        for n in range(N):
            new_train_inputs_n = torch.matmul(iteration_model_params[0].data, train_inputs[n]) + iteration_model_params[1].T.data
            new_train_inputs_n = torch.sign(new_train_inputs_n)
            new_train_inputs_n[new_train_inputs_n <= 0] = 0

            new_train_inputs.append(new_train_inputs_n.numpy())
        new_train_inputs =  torch.Tensor(new_train_inputs)

        for l in range(1,L):
            print("Training layer",l)
            D = new_train_inputs[0].shape[0]                                                            # Dimension of data
            iteration_model_params = SGD_Trainer(new_train_inputs, train_labels)
            SGD_model_params += (iteration_model_params[:2])

            old_train_inputs = new_train_inputs
            new_train_inputs = []
            for n in range(N):
                new_train_inputs_n = torch.matmul(iteration_model_params[0].data, old_train_inputs[n]) + iteration_model_params[1].T.data
                new_train_inputs_n = torch.sign(new_train_inputs_n)
                new_train_inputs_n[new_train_inputs_n <= 0] = 0
                new_train_inputs.append(new_train_inputs_n.numpy())
            new_train_inputs =  torch.Tensor(new_train_inputs)
        SGD_model_params += iteration_model_params[2:]

        # Save params

        SGD_model_params_save = copy.deepcopy(SGD_model_params)
        SGD_model_params = np.array(SGD_model_params, dtype='object')

        for i in range(len(SGD_model_params_save)):
            SGD_model_params_save[i] = SGD_model_params_save[i].tolist()

        param_collection[architecture] = SGD_model_params_save

        # Training Accuracy

        train_inputs = torch.Tensor([train_data[i][0] for i in range(len(train_data))])
        train_labels = torch.Tensor([train_data[i][1] for i in range(len(train_data))])

        N = train_inputs.shape[0]
        D = train_inputs[0].shape[0]

        correct_pred = 0
        for n in range(N):

            h = torch.matmul(SGD_model_params[0], train_inputs[n]) + SGD_model_params[1].data

            h = torch.sign(h)
            h[h <= 0] = 0    

            i = 2
            for l in range(1,L):
                h = torch.matmul(SGD_model_params[i], h) + SGD_model_params[i+1].data

                h = torch.sign(h)
                h[h <= 0] = 0

                i += 2

            h = torch.matmul(SGD_model_params[i], h) + SGD_model_params[i+1].data

            log_softmax = LogSoftmax(h)
            pred = log_softmax.max(0, keepdim=True)[1].item()
            actual =  int(train_labels[n].item())

            if pred == actual:
                correct_pred += 1

        training_acc = correct_pred/(N)*100
        print("\nTraining Accuracy for {0} units = {1}%".format(K, training_acc))

        # Testing Accuracy

        test_inputs = torch.Tensor([test_data[i][0] for i in range(len(test_data))])
        test_labels = torch.Tensor([test_data[i][1] for i in range(len(test_data))])

        N = test_inputs.shape[0]
        D = test_inputs[0].shape[0]

        correct_pred = 0
        for n in range(N):

            h = torch.matmul(SGD_model_params[0], test_inputs[n]) + SGD_model_params[1].data

            h = torch.sign(h)
            h[h <= 0] = 0

            i = 2
            for l in range(1,L):

                h = torch.matmul(SGD_model_params[i], h) + SGD_model_params[i+1].data

                h = torch.sign(h)
                h[h <= 0] = 0

                i += 2

            h = torch.matmul(SGD_model_params[i], h) + SGD_model_params[i+1].data

            log_softmax = LogSoftmax(h)
            pred = log_softmax.max(0, keepdim=True)[1].item()
            actual =  int(test_labels[n].item())

            if pred == actual:
                correct_pred += 1

        testing_acc = correct_pred/(N)*100    
        print("Testing Accuracy for {0} units = {1}%".format(K, testing_acc))

        # Save accuracies

        accuracy_dict = {"Training":training_acc, "Testing":testing_acc}
        accuracies[architecture] = accuracy_dict

        print("Took {0} seconds to train {1} \n".format(time.time()-t0, architecture))

    with open(os.path.join('Greedy_Binary_SGD','Experiment 2 seed '+ str(seed) + ' accuracies.json'),'w') as f:
        json.dump(accuracies, f)
    with open(os.path.join('Greedy_Binary_SGD','Experiment 2 seed ' + str(seed) + ' params.json'),'w') as f:
        json.dump(param_collection, f)
