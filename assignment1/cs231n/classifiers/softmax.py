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
    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = np.matmul(X, W)

    for i in range(num_train):
      sample_score = scores[i]
      sample_score -= np.max(sample_score) # Prevent numerical unstability due to large numbers
      # Exponential of correct class
      correct_exp = np.exp(sample_score[y[i]])
      # Sum of exponentials of all classes
      sum_exp = np.sum(np.exp(sample_score))
      loss += -np.log(correct_exp / sum_exp)
      # Gradient with respect to correct class
      # dL / dW[:, y[i]] = -X[i] + (X[i] * e^(f(yi)) / sum_exp )
      dW[:, y[i]] += -X[i] + ( (X[i] * correct_exp) / sum_exp )
      for j in range(num_classes):
        if(j == y[i]):
          continue
        # Grdient with respect to other classes
        # dL / dW[:, yj] = ( X[i] * e^(f(yj)) / sum_exp )
        dW[:, j] += ( X[i] * np.exp(sample_score[j]) ) / sum_exp 
    
    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    pass

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
    num_classes = W.shape[1]

    scores = np.matmul(X, W)

    # Use broacasting rule: [a,b] / [c,d] = [a/c, b/d] => [correct_exp[i]  / sum_exp[i]]
    correct_exp_arr = np.exp(scores[np.arange(num_train), y])
    sum_exp_arr = np.sum(np.exp(scores[np.arange(num_train)]), axis=1)
    loss = np.sum( -np.log(correct_exp_arr / sum_exp_arr) )

    # Calculate the gradient in the same way as in linear_svm with a little modification
    count_matrix = np.zeros((scores.shape))
    # Calculate exponentials of all scores
    exp_scores = np.exp(scores)
    # Reshape the sum_exp_arr to correct format so we can do the next calculation
    reshaped_sum_exp = sum_exp_arr[:].reshape(num_train, 1)
    # Assign the values of each element to corresponding np.exp(sample_score[j]) / sum_exp 
    # We must reshape so the broadcasting rule can be applied properly.
    # E.g: [[1,2,3],[4,5,6],[7,8,9]] / [[6],[15],[24]] = [[1/6,2/6,3/6], [4/15,5/15,6/15], [7/24,8/24,9/24]]
    count_matrix[np.arange(num_train), :] = exp_scores[np.arange(num_train), :] / reshaped_sum_exp
    # Use the formula -X[i] + (X[i] * e^(f(yi)) / sum_exp ) = X[i] * (-1 + e^(f(yi)) / sum_exp)
    count_matrix[np.arange(num_train), y] = -1 + ( correct_exp_arr[np.arange(num_train)] / sum_exp_arr[np.arange(num_train)] )

    dW = np.matmul(X.T, count_matrix)

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
