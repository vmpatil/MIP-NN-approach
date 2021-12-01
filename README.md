# MIP-NN-approach
Mixed-Integer Programming approach to training Neural Networks

Authors: Vrishabh Patil (vmpatil@wisc.edu), Yonatan Mintz (ymintz@wisc.edu)

**STUDY IN PROGRESS**

### Abstract ###

Artificial Neural Networks (ANNs) are prevalent machine learning models that have been applied across various real world classification tasks. ANNs require a large amount of data to have strong out of sample performance, and many algorithms for training ANN parameters are based on stochastic gradient descent (SGD). However, the SGD ANNs that tend to perform best on prediction tasks are trained in an end to end manner that requires a large number of model parameters and random initialization. This means training ANNs is very time consuming and the resulting models take a lot of memory to deploy. In order to train more parsimonious ANN models, we propose the use of alternative methods from the constrained optimization literature for ANN training and pretraining. In particular, we propose novel mixed integer programming (MIP) formulations for training fully-connected ANNs. Our formulations can account for both binary activation and rectified linear unit (ReLU) activation ANNs, and for the use of a log likelihood loss. We also develop a layer-wise greedy approach, a technique adapted for reducing the number of layers in the ANN, for model pretraining using our MIP formulations. We then present numerical experiments comparing our MIP based methods against existing SGD based approaches and show that we are able to achieve models with competitive out of sample performance that are significantly more parsimonious.

### Code and Experiments

This repo contains python scripts used to formulate and implement the MIP and SGD models used to train the Feed Forward Neural Networks.

These are the scripts in the working directory:

```
Binary_MIP.py
  In this file, we have the final code for the Binary Mixed Integer Program model and its corresponding experiments used in the paper.
  
Binary_SGD.py
  In this file, we have the final code for the Binary Stochastic Gradient Descent model and its corresponding experiments used in the paper.
  
Greedy_Binary_MIP.py
  In this file, we have the final code for the Greedy Binary Mixed Integer Program model and its corresponding experiments used in the paper.

Greedy_Binary_MIP+SGD.py
  In this file, we have the final code for the Binary Stochastic Gradient Descent model using the Greedy Binary MIP results as pre-parameters and its corresponding experiments used in the paper.

Greedy_Binary_SGD.py
  In this file, we have the final code for the Greedy Binary Stochastic Gradient Descent model and its corresponding experiments used in the paper.
  
Greedy_Binary_SGD+SGD.py
  In this file, we have the final code for the Binary Stochastic Gradient Descent model using the Greedy Binary SGD results as pre-parameters and its corresponding experiments used in the paper.
  
Greedy_ReLU_MIP.py
  In this file, we have the final code for the RelU Mixed Integer Program model and its corresponding experiments used in the paper.

Greedy_ReLU_SGD.py
  In this file, we have the final code for the Greedy ReLU Stochastic Gradient Descent model and its corresponding experiments used in the paper.
  
Greedy_ReLU_SGD+SGD.py
  In this file, we have the final code for the ReLU Stochastic Gradient Descent model using the Greedy ReLU SGD results as pre-parameters and its corresponding experiments used in the paper.

ReLU_SGD.py
  In this file, we have the final code for the ReLU Stochastic Gradient Descent model and its corresponding experiments used in the paper.
  
```
