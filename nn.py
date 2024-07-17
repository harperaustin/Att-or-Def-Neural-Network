import numpy as np



def sigmoid(x):
    #This is the activation function that will be used to turn the unbounded input
    #into a predictable output. The function I am using is f(x) = 1/(1+(e^-x)), the sigmoid function.
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    #The deriviate of the sigmoid function.
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    """
    A function that calculates the loss of the neural network by using
    the mean squared error.
    
    y_true and y_pred are numpy arrays of the same length
    - y_true is an array of the true values of an input
    - y_pred is an array of the predicted values of an input
    """
    return ((y_true - y_pred) ** 2).mean()

"""
weights = np.array([0,1])
bias = 4
n = Neuron(weights, bias)
x = np.array([2,3])
print(n.feedforward(x))
"""

class My_NN:
    """
    My neural network is a very simple NN consisting of:
    - 2 inputs
    - 1 hidden layer with 2 neurons (h1, h2)
    - 1 ouput layer with 1 neuron (o1)

    Each neuron will have the same weights and bias: 
    - w = [0,1]
    - b = 0
    """

    def __init__(self) -> None:
        
        

        #The weights and biases for the neural network, generated randomly at first
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()


    def feedforward(self, x):
        #x is a numpy array with 2 elements (the two inputted values)

        #Give the hidden layer the input values
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)

        #Give the output layer the values from the hidden layer
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)

        return o1
    
    def train(self, data, all_y_trues):
        """
        - data is a (n x 2) numpy array, where n is the number of samples in the dataset.
        - all_y_trues is a numpy array with n elements (the true values of the dataset)

        The elements in all_y_true correspond to the elements in data
        """

        learn_rate = 0.01
        
        epochs = 10000 #number of times to loop through the entire dataset

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                #Must do a feedforward, however we can't use our built-in function,
                #as we will need the values for future calculutions.
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w4 * h1 + self.w5 * h2 + self.b3
                o1 = sigmoid(sum_o1)

                y_pred = o1

                #Now calculuate the partial derivates,
                #where dL_dypred represents the parital derivative of L / partial y_pred

                dL_dypred = -2 * (y_true - y_pred)

                #Neuron o1
                dypred_dw5 = h1 * sigmoid_derivative(sum_o1)
                dypred_dw6 = h2 * sigmoid_derivative(sum_o1)
                dypred_db3 = sigmoid_derivative(sum_o1)

                dypred_dh1 = self.w5 * sigmoid_derivative(sum_o1)
                dypred_dh2 = self.w6 * sigmoid_derivative(sum_o1)

                #Neuron h1
                dh1_dw1 = x[0] * sigmoid_derivative(sum_h1)
                dh1_dw2 = x[1] * sigmoid_derivative(sum_h1)
                dh1_db1 = sigmoid_derivative(sum_h1)

                #Neuron h2
                dh2_dw3 = x[0] * sigmoid_derivative(sum_h2)
                dh2_dw4 = x[1] * sigmoid_derivative(sum_h2)
                dh2_db2 = sigmoid_derivative(sum_h2)

                #Now we must update the weights and biases:

                #Updating Neuron h1:
                self.w1 -= learn_rate * dL_dypred * dypred_dh1 * dh1_dw1
                self.w2 -= learn_rate * dL_dypred * dypred_dh1 * dh1_dw2
                self.b1 -= learn_rate * dL_dypred * dypred_dh1 * dh1_db1

                #Updating Neuron h2:
                self.w3 -= learn_rate * dL_dypred * dypred_dh2 * dh2_dw3
                self.w4 -= learn_rate * dL_dypred * dypred_dh2 * dh2_dw4
                self.b2 -= learn_rate * dL_dypred * dypred_dh2 * dh2_db2

                #Updating Neuron o1:

                self.w5 -= learn_rate * dL_dypred * dypred_dw5
                self.w6 -= learn_rate * dL_dypred * dypred_dw6
                self.b3 -= learn_rate * dL_dypred * dypred_db3


                #Calculate the total loss at the end of each epoch
                if epoch % 250 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1, data)
                    loss = mse_loss(all_y_trues, y_preds)
                    print("EPOCH %d loss: %.3f" % (epoch, loss))

    




data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])

all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

network = My_NN()
network.train(data, all_y_trues)




