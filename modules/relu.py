import numpy as np

class ReLU:
    '''
    An implementation of rectified linear units(ReLU)
    '''
    def __init__(self):
        self.cache = None
        self.dx= None

    def forward(self, x):
        '''
        The forward pass of ReLU. Save necessary variables for backward
        :param x: input data
        :return: output of the ReLU function
        '''
        out = None
        #############################################################################
        # TODO: Implement the ReLU forward pass.                                    #
        #############################################################################
        relu_rows = []
        for row in x:
            relu = np.maximum(0,row)
            relu_rows.append(relu)
        out = np.array(relu_rows)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        '''

        :param dout: the upstream gradients
        :return:
        '''
        #############################################################################
        # TODO: Implement the ReLU backward pass.                                   #
        #############################################################################
        #https://stackoverflow.com/questions/46411180/implement-relu-derivative-in-python-numpy
        x = self.cache.copy()
        dx = dout * (x >= 0)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.dx = np.array(dx)
