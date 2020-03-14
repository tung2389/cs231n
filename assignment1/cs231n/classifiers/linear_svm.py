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
    num_classes = W.shape[1] # Added line
    num_dims = X.shape[1] # Added line

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Use matrix multiplication. (500,3073) * (3073,10) = (500, 10). Now scores has shape (500, 10)
    scores = np.matmul(X, W)
    # a vectorized technique: scores[[1,2],[3,4]] will return [ scores[1][3] and scores[2,4] ]
    # correct_class_scores has shape (500,), holding each training sample's correct class score. scores[:, y] won't work
    correct_class_scores = scores[ np.arange(num_train), y ].reshape(num_train, 1)
    delta = 1
    # Loss of all samples with respect to each class: loss_of_samples_and_classes[i, j] is the loss of sample i with respect to class j
    loss_of_samples_and_classes = np.maximum(0, scores - correct_class_scores + delta)
    # Skip scores of correct class. 
    loss_of_samples_and_classes[ np.arange(num_train), y] = 0
    # Sum up all the losses and average the sum to find the final loss
    loss = np.sum(loss_of_samples_and_classes) / num_train
    loss += reg * np.sum(W * W)
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

    # Below is not an optimized solution. It is my first trial to calculate gradient:
    # My idea for this solution: notice that in svm_loss_naive, each dW[:, j] plus X[i] up to num_train times
    # Besides, for each i, dW[:, y[i]] minus X[i] up to num_classes time.
    # Therefore we can accumulate the sum of X[i] to update dW[:, j] and dW[:, y[i]] minus X[i]
    # We create an array arr1 based on loss_of_samples_and_classes, containing values 0 or 1. Then we multiply
    # the training data matrix with the arr1 matrix. The accumulated sum (to update dW[:, j]) can be calculated according to column (across num_train)
    # ,and the accumulated sum of - X[i] can be calculated according to rows (across num_classes)
    #
    #
    # arr1 = np.where(loss_of_samples_and_classes > 0, 1, 0).reshape(num_train, num_classes, 1)
    # train_data = X.reshape(num_train, 1, num_dims)
    # gradient_arr = np.matmul(arr1, train_data)
    # add = np.sum(gradient_arr, axis=0).T # Calculate accoring to column
    # minus = np.sum(gradient_arr, axis=1) # Shape (500, 3073), Calculate according to row
    # dW = dW + add
    # for i in range(num_train):
    #     dW[:, y[i]] = dW[:, y[i]] - minus[i]
    #
    #
    # Why we don't use dW[:, y] = (dW[:, y].T - minus[np.arange(num_train)]).T ? This will not work. 
    # Numpy vectorization cannot calculate from previous value. So if we use this way, we are forced 
    # to use a for loop

    # Here is the true solution:

    count_matrix = loss_of_samples_and_classes
    # If have any loss greater than 0, the value of that element in count_matrix is one
    count_matrix[count_matrix > 0] = 1
    # Count the number of time 'margin > 0' in every training example. It is also the number of the operation
    # dW[:, y[i]] = dW[:, y[i]] - X[i] in one training example.
    count_real_loss = np.sum(count_matrix, axis=1)
    # Count the final number n in the accumulated operation: dW[:, y[i]] - n*X[i]. If count_matrix[2,3] = -2,
    # then dW[:, 3] will minus X[2] 2 times (including the + X[i] in dW[:, J])
    count_matrix[np.arange(num_train), y] = count_matrix[np.arange(num_train), y] - count_real_loss
    # Then multiply X.T with count_matrix to calculate new dW. Basically, in this solution, instead of
    # using dW[:, j] += X[i], we add each element of X[i] to or subtract it from the corresponding element
    # of dW. For example, dw[0, 2] += X[1, 0], dW[1, 2] += X[1, 1], dW[2,2] += X[1, 2], dW[3, 2] += X[1, 3], dW[n, 2] += X[1, n]
    # This accumulation gives us dW[:, 2] += X[1]
    dW = np.matmul(X.T, count_matrix)

    dW /= num_train
    dW += 2 * reg * W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
