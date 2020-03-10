from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W) # Calculate scores for each class
        # X[i].dot(W) is equivalent to np.matmul(W.T, X[i])
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                # We are calculating the gradient, not gradient descent (negative gradient)
                # (W[:, j] * Xi)' = X[i]; (- W[:, y[i]] * Xi)' = -X[i]
                dW[:, j] = dW[:, j] + X[i] # Added line
                dW[:, y[i]] = dW[:, y[i]] - X[i] # Added line

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train # Added line

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    # Calculate the gradient of reg * np.sum(W * W): (reg * np.sum(W^2))' = 2 * reg *  W
    dW += 2 * reg * W # Added line


    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_train = X.shape[0] # Added line

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Use matrix multiplication. (500,3073) * (3073,10) = (500, 10). Now scores has shape (500, 10)
    scores = np.matmul(X, W)
    # a vectorized technique: scores[[1,2],[3,4]] will return [ scores[1][3] and scores[2,4] ]
    # correct_class_scores has shape (500,), holding each training sample's correct class score
    correct_class_scores = scores[ np.arange(num_train), y ].reshape(num_train, 1)
    delta = 1
    # Loss of all samples with respect to each class: loss_of_samples_and_classes[i, j] is the loss of sample i with respect to class j
    loss_of_samples_and_classes = np.maximum(0, scores - correct_class_scores + delta)
    # Skip scores of correct class. 
    loss_of_samples_and_classes[ np.arange(num_train), y] = 0
    # Sum up all the losses and average the sum to find the final loss
    loss = np.sum(loss_of_samples_and_classes) / num_train
    pass

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

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
