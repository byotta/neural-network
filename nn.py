'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np
import matplotlib.pyplot as plt


class NeuralNet:

    def __init__(self, layers, learningRate, regParam, epsilon=0.12, numEpochs=100):
        '''
        Constructor
        Arguments:
                layers - a numpy array of L-2 integers (L is # layers in the network)
                learningRate - the learning rate for backpropagation
                regParam - the regularization parameter for the cost function
                epsilon - one half the interval around zero for setting the initial weights
                numEpochs - the number of epochs to run during training
        '''
        self.layers = layers
        self.learningRate = learningRate
        self.regParam = regParam
        self.epsilon = epsilon
        self.numEpochs = numEpochs
        self.theta = []  # list of matrices of weights, each index will be weights
        self.unique_vals = None

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __forwardProp(self, instance):
        # insert the bias node for the first one
        node_layers = [np.insert(instance, 0, 1)]
        for i in range(len(self.layers)):
            z = np.matmul(node_layers[i].T, self.theta[i])
            next_layer = self.__sigmoid(z)  # sigmoid function
            node_layers.append(np.insert(next_layer, 0, 1)
                               )  # add the bias term
        last_layer = np.matmul(node_layers[-1], self.theta[-1])  # no bias here
        node_layers.append(self.__sigmoid(last_layer))
        return node_layers

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        # first layer input layer
        # we're gonna make our list of matrices of theta
        # there will be (L - 1) elements in this list. L layers and L - 1 bridges between them
        n, d = X.shape
        self.unique_vals = np.unique(y)
        K = len(self.unique_vals)
        # making a better version of the layers array which includes info about all layers not just the hidden ones
        full_layers = self.layers.copy()
        full_layers = np.insert(full_layers, 0, d)
        full_layers = np.append(full_layers, K)
        L = len(full_layers)
        # consider making result a numpy array instead
        for i in range(L - 1):
            np.random.seed(1)
            matrix = np.random.uniform(
                low=-self.epsilon, high=self.epsilon, size=(full_layers[i] + 1, full_layers[i + 1]))
            # future calls of fit may start to add too many elements
            self.theta.append(matrix)
        # we just set up theta

        # now make a list of vectors of our "predictions"
        one_hot = np.zeros((n, K))
        for i in range(n):
            one_hot[i][int(y[i])] = 1

        for epoch in range(self.numEpochs):
            # make our gradient array
            gradients = []
            for q in range(len(self.theta)):
                gradients.append(np.zeros(self.theta[q].shape))
            print(epoch)
            for i in range(n):
                instance = X[i]
                layers = self.__forwardProp(instance)
                # that was forward prop. time for backprop
                last_layer = layers[-1]
                lastLayer_error = last_layer - one_hot[i]
                # we gonna push to beginning of this list
                backprop_errors = [lastLayer_error]
                backprop_errors[0] = np.reshape(
                    backprop_errors[0], (backprop_errors[0].shape[0], 1))
                for j in range(L - 2, 0, -1):
                    prev_error = backprop_errors[0]
                    b = layers[j]
                    fixed_layer = np.reshape(
                        b, (b.shape[0], 1))  # (n,) -> (n, 1)
                    left_prod = np.matmul(self.theta[j][1:, :], prev_error)
                    # left_prod = np.matmul(self.theta[j], prev_error)
                    right_prod = np.multiply(fixed_layer[1:, :], np.subtract(
                        1, fixed_layer[1:, :]))  # currlist[j] should give me a3?
                    # right_prod = np.multiply(fixed_layer, np.subtract(1, fixed_layer))
                    curr_error = np.multiply(left_prod, right_prod)
                    backprop_errors.insert(0, curr_error)
                # just made all our errors

                for q in range(len(self.theta)):
                    # lop off only activation
                    a_l = np.reshape(layers[q], (layers[q].shape[0], 1))
                    a_l_withoutBias = a_l[1:, :]
                    aT = a_l_withoutBias.T  # a_t
                    error_l = backprop_errors[q]  # delta t + 1
                    error_l_fixed = np.reshape(error_l, (error_l.shape[0], 1))
                    product = np.matmul(error_l_fixed, aT)
                    transposed_product = product.T
                    insert_it = np.insert(
                        transposed_product, 0, error_l_fixed.T, axis=0)
                    gradients[q] = np.add(gradients[q], insert_it)

            d = np.divide(gradients, n)
            for layer in range(len(gradients)):
                d[layer] = np.add(d[layer], np.multiply(
                    self.regParam, self.theta[layer]))
                d[layer][0] = np.subtract(d[layer][0], np.multiply(
                    self.regParam, self.theta[layer][0]))
            for i in range(len(self.theta)):
                self.theta[i] = np.subtract(
                    self.theta[i], np.multiply(self.learningRate, d[i]))

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        n, d = X.shape
        res = np.ones((n, 1))
        for i in range(n):
            instance = X[i]
            prediction_array = self.__forwardProp(instance)[-1]
            res[i] = self.unique_vals[np.argmax(prediction_array)]
        return res

    def visualizeHiddenNodes(self, filename):
        '''
        Outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''
        pics = []
        first_layer = self.theta[0][1::]
        n, d = first_layer.shape
        for i in range(d):
            init = first_layer[:, i].reshape(20, 20)
            after = (init - init.min()) / \
                (init.max() - init.min())  # just normalizing
            pics.append(after)
        for i in range(len(pics)):
            plt.imshow(pics[i])
        plt.savefig(filename)
