#!/usr/bin/env python
'''
Demonstrate manual logistic regression, using gradient descent.

DATA 470D3
Stephen Davies, University of Mary Washington, fall 2025
'''
import torch
import matplotlib.pyplot as plt

torch.manual_seed(123)
torch.set_printoptions(precision=2,sci_mode=False)


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
    
def results(X, y, w):
    """
    For inspection, produce an Nx2 matrix whose columns are (0) the correct
    labels y, and (1) the predicted labels produced by a log reg model that
    uses the weight vector w.
    """
    yhat = sigmoid(X @ w)
    return torch.cat([y.unsqueeze(1), yhat.unsqueeze(1)], dim=1)

def create_dataset(
    n_train:int = 100,
    n_test:int = 20,
    p:int = 3,
    frac0:float = .5
):
    """
    Create a dataset with binary labels (0 and 1) and with numerical features.

    n_train - number of labeled examples in training set
    n_test - number of labeled examples in test set
    p - number of features
    frac0 - fraction of data with label 0 (others will be label 1)

    Returns a 4-tuple with:
        - X_train (n_train, p)
        - y_train (n_train)
        - X_test (n_test, p)
        - y_test (n_test)
    """
    assert 0 <= frac0 <= 1

    n = n_train + n_test
    label = (torch.rand(n) < frac0).float()
    means0 = torch.empty((p)).uniform_(0,10)
    std_devs0 = torch.empty((p)).uniform_(0,2)
    means1 = torch.empty((p)).uniform_(0,10)
    std_devs1 = torch.empty((p)).uniform_(0,2)

    features0 = torch.distributions.Normal(means0,std_devs0).sample((n,))
    features1 = torch.distributions.Normal(means1,std_devs1).sample((n,))
    features = torch.where(label[:,None]==0, features0, features1)
    return (features[:n_train], label[:n_train],
            features[n_train:], label[n_train:])


if __name__ == "__main__":

    verbose = True
    plot_loss = True

    # Generate synthetic data set.
    p = 8
    X_train, y_train, X_test, y_test = create_dataset(
        n_train=100,
        n_test=50,
        p=p,
        frac0=0.4,
    )

    # Initialize weights randomly.
    w = (torch.rand(p) - .5).requires_grad_()

    # Just for gigs, print the results with initial, random, weights. 
    print("With random weights...")
    print(results(X_test, y_test, w))

    # Set GD parameters.
    eta = .0001      # Greek letter η, a.k.a. "learning rate"
    loss_delta_thresh = 0.001
    max_iters = 1000
    n_iter = 0

    # Prepare to plot.
    plot_vals = torch.empty((max_iters,))

    # Keep track of the mean cross-entropy loss for (1) our current weight
    # vector, and (2) last iteration's weight vector (so we can measure how
    # much better our model is getting).
    loss = ce_loss(X_train, y_train, w)
    last_loss = torch.tensor(torch.inf)

    # Loop until we hit max_iter, or until our Δ < loss_delta_thresh.
    while True:
        loss = ce_loss(X_train, y_train, w)
        plot_vals[n_iter] = loss
        loss.backward()
        with torch.no_grad():
            w -= eta * w.grad
            w.grad = None
            if verbose:
                print(f"{n_iter} iters, w: {w.data} (loss: {loss.item():4f}, "
                    f"Δ: {last_loss.item() - loss.item():.4f})")
            if (last_loss - loss).item() < loss_delta_thresh:
                break
        last_loss = loss
        n_iter += 1
        if n_iter >= max_iters:
            break

    if plot_loss:
        fig, ax = plt.subplots()
        ax.plot(range(n_iter), plot_vals[:n_iter].detach().numpy())
        ax.set_title("Cross-Entropy Loss during training")
        ax.set_xlabel("iteration")
        ax.set_ylabel("CE loss")
        plt.show()

    # Show performance on the held-out test set
    print("After training...")
    print(results(X_test, y_test, w))
