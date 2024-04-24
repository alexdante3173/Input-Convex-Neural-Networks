# Input-Convex-Neural-Networks
This repository is by [Lucia de Vega](https://github.com/luciadvo), [Alexandra Dinu](https://github.com/anditp) and [Ilinca Tiriblecea](https://github.com/ilincatiri) and contains our implementations of a FICNN, PICNN and a convolutional PICNN.

## Introduction
We will focus our attention on a novel architecture of a neural network, namely Input Convex Neural Networks (ICNNs). These were first introduced in the paper with the same name by Brendon Amos,  Lei Xu and J. Zico Kolter and remained relatively unexplored. To put it briefly, ICNNs are scalar-valued neural networks $` f(x, y;\theta) `$ for which we impose certain constraints on the parameters $` \theta `$ so that the function is convex in the input $`y `$.


## Experiments
Given the potential complexity and challenges associated with implementing ICNNs, we've endeavored to simplify the process by developing our own versions using PyTorch instead of TensorFlow. This choice was motivated by our aim to enhance user-friendliness and accessibility for a wider audience. Within this repository, you'll find our implementations of FICNNs, PICNNs, and convolutional PICNNs, along with our experimentation in approximating several convex functions defined on subsets of $` \mathbb{R}^2 `$ using FICNNs.
