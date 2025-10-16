import torch
from datasets import load_dataset, Dataset

from wordcount_encoder import compute_vocab, encode_all

torch.set_printoptions(precision=2,sci_mode=False)
torch.set_printoptions(profile="default")

def results(X, y, w):
    """
    For inspection, produce an Nx2 matrix whose columns are (0) the correct
    labels y, and (1) the predicted labels produced by a log reg model that
    uses the weight vector w.
    """
    yhat = sigmoid(X @ w)
    return torch.stack([y, yhat], dim=1)

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

if __name__ == "__main__":

    plot_loss = False
    verbose = True
    p = 100

    imdb = load_dataset("imdb")
    small_tr = imdb['train'].shuffle(seed=123)[:1000]
    texts_tr = small_tr['text']
    labels_tr = small_tr['label']
    small_ts = imdb['test'].shuffle(seed=123)[:1000]
    texts_ts = small_ts['text']
    labels_ts = small_ts['label']
    vocab, dfs = compute_vocab(texts_tr, p, True)
    label_mapper = imdb['train'].features['label'].int2str

    X_train = encode_all(texts_tr,vocab,dfs)
    y_train = torch.tensor(labels_tr)

    X_test = encode_all(texts_ts,vocab,dfs)
    y_test = torch.tensor(labels_ts)

    # Initialize weights randomly.
    w = (torch.rand(p) - .5).requires_grad_()

    # Just for gigs, print the results with initial, random, weights. 
    print("With random weights...")
    print(results(X_test, y_test, w))

    # Set GD parameters.
    eta = .1      # Greek letter η, a.k.a. "learning rate"
    loss_delta_thresh = 0.0000000001
    max_iters = 100000
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
                print(f"{n_iter} iters: loss: {loss.item():4f}, "
                    f"Δ: {last_loss.item() - loss.item():.4f}")
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
    print(results(X_train, y_train, w))
    print(results(X_test, y_test, w))
