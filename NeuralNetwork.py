import numpy as np

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)
class NeuralNetwork:

    def __init__(self, inputs, hiddens, outputs):
        #Variable initialization
        self.epoch=1 #Setting training iterations
        self.lr=0.1 #Setting learning rate
        self.inputlayer_neurons = inputs #number of features in data set
        self.hiddenlayer_neurons = hiddens #number of hidden layers neurons
        self.output_neurons = outputs #number of neurons at output layer

        #weight and bias initialization
        self.wh= 2*np.random.uniform(size=(self.inputlayer_neurons,self.hiddenlayer_neurons))-1
        self.bh= 2*np.random.uniform(size=(1,self.hiddenlayer_neurons))-1
        self.wout= 2*np.random.uniform(size=(self.hiddenlayer_neurons,self.output_neurons))-1
        self.bout= 2*np.random.uniform(size=(1,self.output_neurons))-1

    def feedforward(self, input):
        hidden_layer_input1=np.dot(input, self.wh)
        hidden_layer_input=hidden_layer_input1 + self.bh
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output_layer_input1=np.dot(hiddenlayer_activations,self.wout)
        output_layer_input= output_layer_input1+ self.bout
        output = sigmoid(output_layer_input)
        return output

    def teach(self, X, y):
        for i in range(self.epoch):

            #Forward Propogation
            hidden_layer_input1=np.dot(X, self.wh)
            hidden_layer_input=hidden_layer_input1 + self.bh
            hiddenlayer_activations = sigmoid(hidden_layer_input)
            output_layer_input1=np.dot(hiddenlayer_activations,self.wout)
            output_layer_input= output_layer_input1+ self.bout
            output = sigmoid(output_layer_input)

            #Backpropagation
            E = y-output
            slope_output_layer = derivatives_sigmoid(output)
            slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
            d_output = E * slope_output_layer
            Error_at_hidden_layer = d_output.dot(self.wout.T)
            d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
            self.wout += hiddenlayer_activations.T.dot(d_output) *self.lr
            self.bout += np.sum(d_output, axis=0,keepdims=True) *self.lr
            self.wh += X.T.dot(d_hiddenlayer) *self.lr
            self.bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *self.lr

#Input array
X=np.array([[1,0],[0,1],[0,0],[1,1]])

#Output
y=np.array([[1],[1],[0], [0]])
nn = NeuralNetwork(2,2,1)
print(nn.feedforward([1,0]))
print(nn.feedforward([1,1]))
print(nn.feedforward([0,0]))
print(nn.feedforward([0,1]))
nn.teach(X,y)
#
print(nn.feedforward([1,0]))
print(nn.feedforward([1,1]))
print(nn.feedforward([0,0]))
print(nn.feedforward([0,1]))