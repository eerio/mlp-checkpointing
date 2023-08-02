# Multilayer Perceptron with Backpropagation in Numpy
This repository contains an implementation of a Multilayer Perceptron (MLP) with backpropagation using only the NumPy library. The multilayer perceptron is a type of feedforward artificial neural network widely used for various machine learning tasks, such as classification and regression.

## Overview
The multilayer perceptron is a neural network architecture composed of multiple layers of interconnected nodes, each layer consisting of a set of neurons (or nodes). The network consists of an input layer, one or more hidden layers, and an output layer. The forward pass is used to make predictions, while the backward pass (backpropagation) is used to update the weights of the network during training. This code is not really meant to be used - it will certainly be slower and have less features than the implementations available in the major Deep Learning frameworks.

I implemented it for the Deep Neural Networks course at University of Warsaw (MIMUW) in 2022, scoring 9.5/10. The notebook contains:
- the whole implementation, all the comments, links to papers and mathematical notes about calculating matrix derivatives (for the backpropagation part)
- at the end of it, benchmarks are presented, which prove that the checkpointing really does reduce the asymptotic space complexity (from `O(n)` to `O(sqrt(n))`)
- the MLP is also proved to be useful, by training a standard linear regression task - the network reaches the accuracy of ~0.9 in a minute of wall time

What's cool in all of this? Please just take a look at the elegance of these calculations:
```python
# Now go backward from the final cost applying backpropagation
deriv = self.cost_derivative(afters[-1], y)
first_iteration = True
for activation, prev_activation, weights in zip(
    reversed(afters), reversed(afters[: -1]), reversed(self.weights)
):
    # don't calculate dL/dg^N and dg^N/df^N separately for the softmax layer
    # no need to calculate the Jacobian matrix - calculate dL/df^N directly
    # in softmax regression with log-loss, this is dL/df^N, *not* dL/dg^N!
    if not first_iteration:
        # now deriv = d(L) / d(g^k)
        # calculate deriv = d(L) / d(f^k)
        # the below is the Hadamard product of the matrices
        assert deriv.shape == activation.shape
        deriv = deriv * (activation * (1 - activation))
    first_iteration = False

    # d(L)/d(W^k) = d(L)/d(f^k) @ d(f^k) / d(W^k) = d(L) / d(f^k) @ (g^{k-1})^T
    nabla_w += [deriv @ prev_activation.T]
    # d(L)/d(b^k) = d(L)/d(f^k) @ d(f^k) / d(b^k) = d(L) / d(f^k) @ [1, 1 .., 1]
    nabla_b += [deriv @ np.ones([deriv.shape[1], 1])]

    # calculate deriv = d(L) / d(g^{k - 1})
    deriv = weights.T @ deriv
nabla_b, nabla_w = list(reversed(nabla_b)), list(reversed(nabla_w))
```

I personally had much fun implementing these, because I actually calculated all the matrix derivatives myself and saw how memoization can be used for efficiency, which enabled me to finally truly understand this algorithm :)
