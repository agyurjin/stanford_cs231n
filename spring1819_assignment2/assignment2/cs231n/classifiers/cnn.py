from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, speed="naive"):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #                           
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.speed = speed
        c, H, W = input_dim
        self.params["W1"] = weight_scale * np.random.randn(num_filters, c, filter_size, filter_size)
        self.params["b1"] = np.zeros(num_filters)

        conv_param = {'stride': 1, 'pad': (filter_size-1)//2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        pad = conv_param["pad"]
        conv_s = conv_param["stride"]

        pool_s = pool_param["stride"]
        pw = pool_param["pool_width"]
        ph = pool_param["pool_height"]

        H_prime1 = 1 + int((H + 2*pad - filter_size)/conv_s)
        W_prime1 = 1 + int((W + 2*pad - filter_size)/conv_s)

        H_prime2 = 1 + int((H_prime1 - ph)/pool_s)
        W_prime2 = 1 + int((W_prime1 - ph)/pool_s)

        flatten_size = num_filters * H_prime2 * W_prime2
        self.params["W2"] = weight_scale * np.random.randn(flatten_size, hidden_dim)
        self.params["b2"] = np.zeros(hidden_dim)

        self.params["W3"] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params["b3"] = np.zeros(num_classes)
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        if self.speed == "naive":
            conv_forward, cache_conv = conv_forward_naive(X, W1, b1, conv_param)
            re_forward, cache_re = relu_forward(conv_forward)
            maxpool_forward, cache_maxpool = max_pool_forward_naive(re_forward, pool_param)
        elif self.speed == "fast":
            conv_re_forward, cache_conv_re = conv_relu_forward(X, W1, b1, conv_param)
            maxpool_forward, cache_maxpool = max_pool_forward_fast(conv_re_forward, pool_param)
        else:
        	raise ValueError('Unrecognized speed "%s"' % self.speed)

        # FCN part
        af1_forward, cache_af1 = affine_forward(maxpool_forward, W2, b2)
        re1_forward, cache_re1 = relu_forward(af1_forward)
        af2_forward, cache_af2 = affine_forward(re1_forward, W3, b3)

        scores = af2_forward

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, softmax_grad = softmax_loss(scores, y)
        loss += 0.5*self.reg*(np.sum(W2*W2) + np.sum(W3*W3))

        # FCN part
        dx3, dW3, db3 = affine_backward(softmax_grad, cache_af2)
        grads["W3"] = dW3 + self.reg * W3
        grads["b3"] = db3

        dr1 = relu_backward(dx3, cache_re1)

        dx2, dW2, db2 = affine_backward(dr1, cache_af1)
        grads["W2"] = dW2 + self.reg * W2
        grads["b2"] = db2

        if self.speed == "naive":
        	dmp = max_pool_backward_naive(dx2, cache_maxpool)
        	dr = relu_backward(dmp, cache_re)
        
        	dx1, dW1, db1 = conv_backward_naive(dr, cache_conv)
        	grads["W1"] = dW1
        	grads["b1"] = db1
        elif self.speed == "fast":
        	dmp = max_pool_backward_fast(dx2, cache_maxpool)
        	dx1, dW1, db1 = conv_relu_backward(dmp, cache_conv_re)
        	grads["W1"] = dW1
        	grads["b1"] = db1
        else:
        	raise ValueError('Unrecognized speed "%s"' % self.speed )

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
