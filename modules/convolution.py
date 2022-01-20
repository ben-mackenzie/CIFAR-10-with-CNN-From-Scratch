import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        # Get input dimensions
        N, C, H, W = x.shape
        num_filters = self.weight.shape[0]

        # Calculate output dimensions
        H_out = int((H-self.kernel_size+2*self.padding)/self.stride)+1
        W_out = int((W-self.kernel_size+2*self.padding)/self.stride)+1
        out = np.zeros((N, num_filters, H_out, W_out))

        # Pad input
        # https://datascience-enthusiast.com/DL/Convolution_model_Step_by_Stepv2.html
        x_padded = np.pad(x, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)))

        for f in range(num_filters):
            for i in range(H_out):
                for j in range(W_out):

                    # get receptive field dimensions
                    H_start = i * self.stride
                    H_end = H_start + self.kernel_size
                    W_start = j * self.stride
                    W_end = W_start + self.kernel_size

                    # multiply receptive field by kernel
                    receptive_field = x_padded[:,:,H_start:H_end,W_start:W_end]
                    kernel = self.weight[f,:,:,:]
                    product_rf_kernel = receptive_field*kernel

                    out[:,f,i,j] = np.sum(product_rf_kernel, axis =(1,2,3))+self.bias[f]

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x, x_padded, num_filters
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x, x_padded, num_filters = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        # define dimensions
        H_dout, W_dout = dout.shape[2], dout.shape[3]
        N,C,H_dx, W_dx = x_padded.shape

        # define output variables
        self.dw = np.zeros((num_filters,C,self.kernel_size,self.kernel_size))
        self.dx = np.zeros((x.shape))
        dx_pad = np.zeros((N,C, H_dx, W_dx))

        for f in range(num_filters):
            for i in range(H_dout):
                for j in range(W_dout):

                    # define receptive field
                    H_start = i * self.stride
                    H_end = H_start + self.kernel_size
                    W_start = j * self.stride
                    W_end = W_start + self.kernel_size
                    receptive_field = x_padded[:, :, H_start:H_end, W_start:W_end]

                    # select dout elements i,j
                    dout_elements = dout[:,f,i,j]

                    # DW: define matrix to store sum of dout element and receptive field over all training examples
                    dout_rf_matrix = np.zeros((C,self.kernel_size, self.kernel_size))

                    # loop over training examples
                    for k in range(len(dout_elements)):

                        # DW: multiply dout element by receptive field
                        dout_matrix = np.multiply(dout_elements[k],receptive_field[k,:,:,:])
                        dout_rf_matrix += dout_matrix

                        # DX: calculate stamp for one training example
                        element = dout_elements[k]
                        dx_stamp_example = np.multiply(element,self.weight[f,:,:,:])
                        dx_pad[k, :, H_start:H_end, W_start:W_end] += dx_stamp_example

                    # DW: add summed dout_rf matrix for curr receptive field for curr filter
                    self.dw[f,:,:,:]  += dout_rf_matrix

        # DW: update self.dw
        self.dw = np.array(self.dw)

        # DX: remove padding
        # https://github.com/AdalbertoCq/Deep-Learning-Specialization-Coursera/blob/master/Convolutional%20Neural%20Networks/week1/convolution_model.py
        self.dx = dx_pad[:,:,self.padding:-self.padding,self.padding:-self.padding]

        # DB
        self.db = np.sum(dout, axis=(0, 2, 3))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
