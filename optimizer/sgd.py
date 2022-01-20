from ._base_optimizer import _BaseOptimizer
class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum

        # initialize the velocity terms for each weight


    def update(self, model):
        '''
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        '''
        self.apply_regularization(model)

        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for weights                                        #
                #############################################################################
                pass
                # calculate velocity
                v_previous = self.grad_tracker[idx]['dw']
                v = self.momentum * v_previous - self.learning_rate*m.dw

                # add v to weights
                m.weight += v

                # store velocity
                self.grad_tracker[idx]['dw'] = v
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
            if hasattr(m, 'bias'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for bias                                           #
                #############################################################################
                # calculate velocity
                v_previous = self.grad_tracker[idx]['db']
                v = self.momentum * v_previous - self.learning_rate*m.db

                # add velocity to bias
                m.bias += v

                # store velocity
                self.grad_tracker[idx]['db'] = v
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
