import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        # https://towardsdatascience.com/lets-code-convolutional-neural-network-in-plain-numpy-ce48e732f5d5

        # Get input and kernel dimensions
        N, C, H, W = x.shape
        H_k, W_k = self.kernel_size, self.kernel_size

        # Calculate output dimensions
        H_out = int((H - self.kernel_size)/self.stride + 1)
        W_out = int((W - self.kernel_size)/self.stride + 1)
        out = np.zeros((N, C, H_out, W_out))

        # Stride kernel over input and pool
        for i in range(H_out):
            for j in range(W_out):
                H_start = i * self.stride
                H_end = H_start + H_k
                W_start = j * self.stride
                W_end = W_start + W_k

                # Pool
                receptive_field = x[:, :, H_start:H_end, W_start:W_end]
                out[:,:,i,j] = np.max(receptive_field, axis=(2,3)) #check axes

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x, H_out, W_out
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        '''
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        N,F,H_dout, W_dout = dout.shape
        dx = np.zeros(x.shape)
        for i in range(H_out):
            for j in range(W_out):
                H_start = i * self.stride
                H_end = H_start + self.kernel_size
                W_start = j * self.stride
                W_end = W_start + self.kernel_size

                for f in range(F):
                    for n in range(N):
                        # create mask
                        dout_element = dout[n,f,i,j]
                        # define receptive field
                        receptive_field = x[n, f, H_start:H_end, W_start:W_end]
                        mask = receptive_field == np.max(receptive_field, keepdims=True)
                        dout_masked = np.multiply(dout_element, mask)
                        dx[n, f, H_start: H_end, W_start: W_end] += dout_masked
        self.dx = dx

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

