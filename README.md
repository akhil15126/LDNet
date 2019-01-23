# DNDNet

This repository trains and attacks (using C&W L2 attack[1]) a sample model on MNIST[2] dataset with and without the proposed defense layer. 

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

Running code without DNDNet

```
python main.py 
```

Running code with DNDNet

```
python main.py --ldnet
```

## References

[1] Carlini, N. and Wagner, D. Towards evaluating the robustness of neural networks. In IEEE Symposium on Security
and Privacy, pp. 39–57, 2017b.

[2] http://yann.lecun.com/exdb/mnist/


