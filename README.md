# MIP-NN-approach
Mixed-Integer Programming approach to training Neural Networks

Authors: Yonatan Mintz (ymintz@wisc.edu), Vrishabh Patil (vmpatil@wisc.edu)

**STUDY IN PROGRESS**

### Problem Statement ###

  One of the most popular methods of training feed-forward neural networks is by using backpropagation (backprop), a machine learning algorithm that computes the gradient of the loss function with respect to the weights of the network. Backprop has many known limitations, a major one being the vanishing gradient problem, where, in large neural networks, the gradient calculated is diminishingly small and prevents the weights from being updated. Other limitations include finding a local minimum as opposed to a global minimum due to the limitations of stochastic gradient descent. Finally, end-to-end backpropagation requires substantial memory in a naïve implementation, as all the parameters, including activations and gradients at each step, need to fit in a processing unit’s working memory. This project explores an alternative to backprop as a formulation that trains neural using Mixed-Integer Programming to minimize the loss function as constrained by a convex feasible set. This approach should resolve the problems associated with vanishing gradients, as well as assure a global minimum thanks to the convex nature of the problem. Additionally, by constructing the problem as a linear convex optimization problem, we hope to avoid issues associated with stochastic initialization.

### Implementation

This repo contains scripts and notebooks used to formulate and implement the MIPs used to train the 0-1 Feed Forward Neural Networks.

These are the notebooks/scripts in the working directory:

```
Bender's Decomposition.ipynb
  In this file, there is an attempt at decomposing the general MIP using Bender's Decomposition.

General and Multilayer MIP.ipynb
  In this file, the general, non-decomposed MIP is formulated and implemented.

LD Method-Formulation 1.pdf
  In this file, there is the **first** attempt at reformulating the MIP using Lagrangian Relaxation.
  This reformulation dualizes the 4 constraints that dictate the value of the auxiliary variable, z.
  
LD Method-Formulation 2.pdf
  In this file, there is the **second** attempt at reformulating the MIP using Lagrangian Relaxation.
  This reformulation uses the artificial variable g, with constraints that maps $ g_{l} $ to $ h_{l-1} $ that are dualized.
  
LD Method-Implementation 1.ipynb
  In this file, there is an implementation of the **first** reformulation the MIP using Lagrangian Relaxation.
  
LD Method-Implementation 2.ipynb
  In this file, there is an implementation of the **first** reformulation the MIP using Lagrangian Relaxation.  

Layerwise MIP.ipynb
  In this file, the MIP is explicitly decomposed layer-by-layer to be solved by brute force.
  There are function that that solve the layers simulataneously and iteratively.

```
