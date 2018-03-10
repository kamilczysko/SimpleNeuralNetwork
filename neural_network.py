import numpy as np

class NeuralNetwork:

    def __init__(self, inputs, hiddens, outputs):
        self.learning_rate = 0.9
        self.weights_IH = (2*np.random.uniform(size=(inputs, hiddens))-1)
        self.weights_HO = (2*np.random.uniform(size=(hiddens, outputs))-1)
        self.bias_H = (2*np.random.uniform(size=(1, hiddens))-1)
        self.bias_O = (2*np.random.uniform(size=(1, outputs))-1)
        # print(self.weights_IH,' --- ',self.weights_HO)

    def feedforward(self, inputs):

        h_output = sigmoid(np.dot(inputs, self.weights_IH) + self.bias_H)
        output = sigmoid(np.dot(h_output, self.weights_HO) + self.bias_O)
        return output

    def train(self, data, target):

        output_h = sigmoid(np.dot(data, self.weights_IH) + self.bias_H)
        output_o = sigmoid(np.dot(output_h, self.weights_HO) + self.bias_O)
        error_o = target - output_o
        slope_o = derivatives_sigmoid(output_o)
        slope_h = derivatives_sigmoid(output_h)
        err_slo_o = error_o * slope_o
        error_h = err_slo_o.dot(self.weights_HO.transpose())
        err_slo_h = error_h * slope_h

        self.weights_HO += np.transpose(output_h).dot(err_slo_o)*self.learning_rate
        self.bias_O += err_slo_o * self.learning_rate
        self.weights_IH += np.array(data).transpose().dot(err_slo_h) * self.learning_rate
        self.bias_H += err_slo_h * self.learning_rate


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)
