# LDNet

This repository trains and attacks (using C&W L2 attack) a sample model on MNIST dataset with and without the proposed defense layer. 

## Prerequisites

Following are the libraries that need to be installed before you use the toolbox:

```
numpy
keras
tensorflow
```

Code for C&W L2 attack has been directly picked up from the paper "Towards Evaluating the Robustness of Neural Networks" by Nicholas Carlini and David Wagner.


Repositiory Link : https://github.com/carlini/nn_robust_attacks

## Usage

Running code without LDNet

```
python main.py 
```

Running code with LDNet

```
python main.py --ldnet
```
