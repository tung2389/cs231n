from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape
        num_train = N # Added line
        num_dims = D  # Added line
        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # X1 holds first fully connected layer scores of all samples. X1[i] represents 10 scores of ten neurons of sample i
        X1 = np.matmul(X, W1) + b1
        # "The network uses a ReLU nonlinearity after the first fully connected layer." Use ReLu activation function
        X1_activation = np.maximum(0, layer1_scores)
        # Output layer
        X2 = np.matmul(X1_activation, W2) + b2
        scores = X2
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # scores = X2
        scores -= np.max(scores) # Avoid numerical instability
        scores_exp = np.exp(scores)
        correct_exp_arr = scores_exp[np.arange(num_train), y]
        sum_exp_arr = np.sum(scores_exp[np.arange(num_train)], axis=1)
        loss = np.sum( -np.log(correct_exp_arr / sum_exp_arr) )

        loss /= num_train
        loss += reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # To calculate dL_dW2, use the same way as we calculate the gradient in softmax vectorized
        # See softmax_vectorized in sotfmax.py for more details
        reshaped_sum_exp = sum_exp_arr[:].reshape(num_train, 1)
        # Calculate e^f(yj) / sum_exp for every j in every training sample.
        count_matrix = scores_exp / reshaped_sum_exp # (1)
        # Add -1 part in both dL_dW[:, yi] = X[i] * (-1 + ( e^(f(yi)) / sum_exp) ) and dL / db2[yi]  = -1 + ( e^f(yi) / sum_exp )
        count_matrix[np.arange(num_train), y] = -1  

        # dl_dX2 calculated based on the formula L = Σ ( -f(yi) + log Σ ( e^(f(yj)) ) ), using partial derivative on every element f of X2
        dL_dX2 = count_matrix
        dX2_dW2 = X1.activation.T
        dX2_dX1act = W2.T
        # Chain rule
        dL_dW2 = np.matmul(dL_dX2, dX2_dW2)
        dL_dX1act = np.matmul(dL_dX2, dX2_dX1act)
        # dL / db2[yi]  = -1 + e^f(yi) / sum_exp. Add the -1 part to the dL / db2[yi] first
        # dL / db2[j] = e^f(yj) / sum_exp
        # every j in dL / db2[j] in a training example are added with e^f(yj) / sum_exp (we do that in (1)). So we just need to sum according to column in count_matrix to calculate dL / db2
        dL_db2 = np.sum(dL_dX2, axis=0)
        
        # (ReLU(X))' =  signum(ReLU(x)). If X > 0, then (ReLU(X))' = 1. Else (ReLU(X))' = 0
        dX1act_dX1 = np.sign(X1_activation)
        dX1_dW1 = X.T
        # Chain rule
        dL_dX1 = np.matmul(dL_dX1act, dX1act_dX1)
        dL_dW1 = np.matmul(dL_dX1, dX1_dW1)
        dL_db1 = np.sum(dL_dX1, axis=0)

        dL_dW1 /= num_train
        dL_dW2 /= num_train
        dL_db1 /= num_train
        dL_db2 /= num_train

        dL_dW1 += 2 * reg * W1
        dL_dW2 += 2 * reg * W2

        grads['W1'] = dL_dW1
        grads['W2'] = dL_dW2
        grads['b1'] = dL_db1
        grads['b2'] = dL_db2

        # Why dL_db2 = np.sum(dL_dX2, axis=0) and dL_db1 = np.sum(dL_dX1, axis=0)?
        # Because of broadcasting rule, we can write X2 = W2 . X1 + np.matmul(np.ones(num_train, 1), b2.reshape(1,output_size))
        # => dX2 / db2 = np.ones(num_train, 1). Then np.matmul(dL_dX2.T, dX2 / db2) is basically the sum of dL_dX2.T along the rows

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
