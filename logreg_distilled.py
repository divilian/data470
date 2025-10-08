#!/usr/bin/env python
'''
A non-functional version of logreg.py that only shows the conceptually
important lines.

DATA 470D3
Stephen Davies, University of Mary Washington, fall 2025
'''


def sigmoid(z):
    """
    Convert a logit (from -∞ to ∞) to a probability (from 0 to 1).
    """
    return 1 / (1 + torch.exp(-z))

def ce_loss(X, y, w):
    """
    Compute the mean cross-entropy loss produced by a logistic model using the
    weights w, on a data set with features X and labels y.
    """
    yhat = sigmoid(X @ w)
    return torch.sum( -(y * torch.log(yhat) + (1-y) * torch.log(1-yhat)))

# Specify dimensions.
n = 10
n_test = 5
p = 8

# Create a random dataset.
#   X_train is (n x p), y_train is (n x 1).
#   X_test is (n_test x p), y_test is (n_test x 1).
X_train, y_train, X_test, y_test = create_dataset(...)

# Initialize weights randomly.
w = (torch.rand(p) - .5).requires_grad_()

# Set GD parameters.
eta = .0000000001      # Greek letter η, a.k.a. "learning rate"

# Perform gradient descent for a fixed number of iterations. (In reality,
# will terminate when learning tails off.)
for i in range(1000):
    loss = ce_loss(X_train, y_train, w)
    loss.backward()
    with torch.no_grad():
        w -= eta * w.grad
        w.grad = None
