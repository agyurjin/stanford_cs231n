from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        scores_shifted_exp = np.exp(scores - np.max(scores))
        sum_of_exps = np.sum(scores_shifted_exp)
        sigmoid = lambda x: scores_shifted_exp[x] / sum_of_exps

        loss -= np.log(sigmoid(y[i]))

        for j in range(num_classes):
            dW[:, j] = (sigmoid(j) - (j == y[i]))*X[i]

    loss /= num_train
    loss += reg * np.sum(W*W)
    dW /= num_train
    dW += 2*reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    scores = X.dot(W)
    max_of_each = np.max(scores, axis=1)
    scores_shifted_exp = np.exp(scores - np.matrix(max_of_each).T)
    sum_of_exps = np.sum(scores_shifted_exp, axis=1)
    probs = scores_shifted_exp/np.matrix(sum_of_exps)

    loss = -np.mean(np.log(probs[np.arange(probs.shape[0]), y]))
    loss += reg*np.sum(W*W)

    delta = scores_shifted_exp
    delta[np.arange(delta.shape[0]), y] -= 1
    dW = np.dot(X.T, delta) / num_train
    dW += 2*reg*W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
