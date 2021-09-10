from builtins import range
import numpy as np
from random import shuffle

MAGIC_TEST_POINTS=[]#[(0,0), (13,3), (102, 1)]

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if (i, j) in MAGIC_TEST_POINTS:
                print(f"{i},{j} = {margin} ({correct_class_score}, {scores[j]}")
            if margin > 0:
                dW[:, j] = dW[:, j] + X[i]
                dW[:, y[i]] = dW[:, y[i]] - X[i]
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW += 2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    # D - sample length, C - classes, N - number of training samples
    X = X.T # D x N
    W = W.T # C x D
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_training = X.shape[1]
    num_classes = W.shape[0]
    loss = 0.0

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = W.dot(X) # C x N
    correct_scores = scores[y, range(num_training)]
    loss_mat = (scores - correct_scores) + 1
    # Remove self-counting of the score of the correct label with itself
    loss_mat[y, range(num_training)] = 0
    #for (i, j) in MAGIC_TEST_POINTS:
    #    print(f"{i},{j} = {loss_mat[i, j]} ({correct_scores[i]} {scores[i, j]})")
    margin_passed = loss_mat < 0
    loss_mat[margin_passed] = 0
    # REDUCTED:
    # We reduce 1 from the loss at the end because we summed a loss of 1
    # for each of the training samples since we didn't zero the y[i]-th column
    # and that adds `num_training` which on average is an extra 1 to the total loss
    loss = (reg*(np.sum(W)**2)) + (np.sum(loss_mat)/num_training)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Marks for each i, c whether score[c] - score[y[i]] + 1 > 0 for sample i
    margin_failed = (~margin_passed).astype(int) # C x N

    # This is the hardest part in code to explain, but basically what we
    # can see is that: for each sample i, we add up to the j-th class the value of Xi,
    # and subtract from the correct class the value of Xi. Therefore, we want to have
    # A matrix to describe for each training sample and class whether to add/reduce Xi.
    # 
    # Since matrix multiplication ends up as a sum over the middle dimesion
    # (in the case of A x B times B x C we sum over the B part), we can start with a
    # matrix of 1's for every time the training sample crossed the margin for a specific
    # class. We already have the opposite of that for the loss calculation to zero the negatives,
    # So we can use that negated. We then have the exact matrix for the summation of Xi's for the classes,
    # and the amount we need to subtract Xi from the correct class will simply be the number of margins
    # that we crossed for that training sample (their sum if each is marked with a 1) :) 
    dW_summer = np.zeros([num_classes, num_training])
    dW_summer[y, range(num_training)] -= np.sum(margin_failed, axis=0)
    dW_summer += margin_failed

    dW = dW_summer.dot(X.T) / num_training
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW.T
