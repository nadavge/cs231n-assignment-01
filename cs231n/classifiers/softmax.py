from builtins import range
import numpy as np
from random import shuffle


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
    num_training = X.shape[0]
    num_classes = W.shape[1]
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(num_training):
        scores = X[i].dot(W)
        adjusted_scores_sum = 0.0
        for j in range(num_classes):
            adjusted_scores_sum += np.exp(scores[j])
        
        for j in range(num_classes):
            dW[:, j] += (np.exp(scores[j])*X[i])/adjusted_scores_sum
        dW[:, y[i]] -= X[i]

        loss += np.log(adjusted_scores_sum) - scores[y[i]]

    loss = loss/num_training
    loss += reg*(np.sum(W*W))

    dW = dW/num_training
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    X = X.T # D x N
    W = W.T # C x D
    num_training = X.shape[1]
    num_classes = W.shape[0]

    scores = W.dot(X) # C x N
    score_exps = np.exp(scores)
    dW = score_exps.dot((X / np.sum(score_exps, axis=0)).T)

    correct_mask = np.zeros_like(scores) # C x N
    correct_mask[y, range(scores.shape[1])] = 1
    dW -= correct_mask.dot(X.T)
    
    dW /= num_training

    dW += 2*reg*W

    loss = np.mean(np.log(np.sum(score_exps, axis=0)) - scores[y, range(scores.shape[1])])
    loss += reg*np.sum(W*W)

    dW = dW.T

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
