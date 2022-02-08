import os
import sys
import utils as ut
from sklearn.metrics import accuracy_score
import pandas as pd
import timeit
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('letter-recognition.csv')
enc = LabelEncoder()
enc.fit(df['letter'])
df['letter'] = enc.transform(df['letter'])
print(len(df.columns))

x = np.array(df.iloc[:,1:])
y = np.array(df.iloc[:,0])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)


x_train[0]
def classify(X_test, parameters, class_dict, layers):
        obj2 = NeuralNet()
        y_test, res = obj2.predict(X_test, parameters, class_dict, layers)
        res = np.asarray(res)
        return res



class NeuralNet:

    def fit(self, X_train, y_train, X_test, y_test, layers, alpha, epochs):

        # Initialize Weights and Biases
        parameters = self.initialize(layers)

        cost_function, learning_curve, val_losses, val_accuracy = [], [], [], []

        # Binarize Labels
        classes = list(set(y_train))
        y_bin = ut.label_binarize(y_train, classes)
        y_val_bin = ut.label_binarize(y_test, classes)

        for j in range(epochs):
            print('Start of %d epoch'%(j+1))
            # ---------------------------------------------------
            y_hat, parameters = self.forward_prop(X_train, parameters, layers)
            log_loss = ut.log_loss(y_bin, y_hat)
            
            
            # Back Propagation
            parameters = self.back_prop(X_train, y_bin, parameters, layers)

            # Prep variables for adam optimizer
            params, grads = self.prep_vars(layers, parameters)

            # Initialize constructor of adam optimizer
            learning_rate_init = alpha
            optimizer = AdamOptimizer(params, learning_rate_init)

            # updates weights with grads
            params = optimizer.update_params(grads)  # update weights

            # Unpack results from Adam Optimizer
            parameters = self.params_unpack(params, parameters, layers)

            # Append log loss, to plot curve later
            cost_function.append(log_loss)

            # ---------------------------------------------------
            # Mapping
            if j == 0:
                class_dict = dict()
                for i in range(len(y_bin)):
                    class_dict[str(y_bin[i])] = y_train[i]
            # ---------------------------------------------------
            
            _, result_ = self.predict(X_train, parameters, class_dict, layers)
            learning_curve.append(accuracy_score(y_train,result_))
            
            
            y_pred, res_val = self.predict(X_test, parameters, class_dict, layers)
            # print(y_pred[0])
            val_loss = ut.log_loss(y_val_bin, y_pred)
            val_losses.append(val_loss)
            val_accuracy.append(accuracy_score(y_test,res_val))
            
            print('\t Train loss : {}, val loss : {}'.format(cost_function[-1],val_losses[-1]))
            print('\t Train accuracy : {}, val accuracy : {}'.format(learning_curve[-1],val_accuracy[-1]))

        # Making plots
        
        plt.plot(range(epochs),learning_curve,label = 'Train')
        plt.plot(range(epochs),val_accuracy,label = 'Test')
        plt.legend()
        plt.title("Learning Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.savefig("{}-{}.png".format(epochs,alpha))
        plt.show()
        
        print()
        plt.plot(range(epochs),cost_function, label = 'Train')
        plt.plot(range(epochs),val_losses, label = 'Test')
        plt.legend()
        plt.title("Logistic Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig("{}_{}.png".format(epochs,alpha))
        plt.show()

        return parameters, class_dict

    def prep_vars(self, layers, parameters):

        params, grads = [], []

        for j in "W", "B":
            for i in range(1, len(layers)):
                params.append(parameters[j + str(i)])
                grads.append(parameters['d' + j + str(i)])

        params = np.asarray(params)
        grads = np.asarray(grads)

        return params, grads

    def params_unpack(self, params, parameters, layers):

        j = 0
        for i in range(1, len(layers)):
            parameters['W' + str(i)] = params[j]
            j += 1

        for i in range(1, len(layers)):
            parameters['B' + str(i)] = params[j]
            j += 1

        return parameters

    def initialize(self, layers):
        """
        Xavier Initialization of weights and biases
        :param layers:
        :return: Initialized weights and biases
        """
        # We will use a dictionary throughout the code to refer to values
        parameters = {}

        # Random seed
        rand_state = np.random.RandomState(42)

        for i in range(1, len(layers)):
            bound = np.sqrt(6. / (layers[i - 1] + layers[i]))
            parameters['W' + str(i)] = rand_state.uniform(-bound, bound, (layers[i - 1], layers[i]))
            parameters['B' + str(i)] = rand_state.uniform(-bound, bound, layers[i])

        return parameters

    def forward_prop(self, data, parameters, layers):
        
        parameters['A' + str(0)] = data

        for i in range(1, len(layers)):
            parameters['Z' + str(i)] = np.add(np.dot(parameters['A' + str(i - 1)], parameters['W' + str(i)]),
                                              parameters['B' + str(i)])

            if i != len(layers) - 1:
                parameters['A' + str(i)] = ut.relu(parameters['Z' + str(i)])
            else:
                # Final Activation is Softmax
                parameters['A' + str(i)] = ut.softmax(parameters['Z' + str(i)])

        return parameters['A' + str(len(layers) - 1)], parameters

    def back_prop(self, X_train, Y, parameters, layers):
        m = X_train.shape[0]  # Number of values; used for averaging

        parameters['dZ' + str(len(layers) - 1)] = (1 / m) * (parameters['A' + str(len(layers) - 1)] - Y)
        parameters['dW' + str(len(layers) - 1)] = np.dot(np.transpose(parameters['A' + str(len(layers) - 2)]),
                                                         parameters['dZ' + str(len(layers) - 1)])
        parameters['dB' + str(len(layers) - 1)] = parameters['dZ' + str(len(layers) - 1)].sum()

        for i in range(len(layers) - 2, 0, -1):
            parameters['dZ' + str(i)] = (1 / m) * (np.dot(parameters['dZ' + str(i + 1)],
                                                          np.transpose(parameters['W' + str(i + 1)])) *
                                                   (self.relu_derivative(parameters['Z' + str(i)])))
            parameters['dW' + str(i)] = np.dot(np.transpose(parameters['A' + str(i - 1)]), parameters['dZ' + str(i)])
            parameters['dB' + str(i)] = parameters['dZ' + str(i)].sum()

        return parameters
    
    def tanh(self, x):
        return np.tanh(x)

    def relu_derivative(self, x):
        """
        Derivative of ReLU Activation
        :param x: input data
        :return: 0 if value <= 0, else 1
        """
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def predict(self, X_test, parameters, class_dict, layers):
        # Call forward Prop
        y_test, parameters = self.forward_prop(X_test, parameters, layers)

        # Binarize probabilities
        y_test1 = (y_test == y_test.max(axis=1)[:, None]).astype(float)

        res = []
        # Map binarized probabilities to relevant classes
        for i in range(y_test1.shape[0]):
            res.append(class_dict[str(y_test1[i])])

        return y_test, res


# -------------------------------------------------------------------------
class AdamOptimizer():

    def __init__(self, params, learning_rate_init=0.001, beta_1=0.9,
                 beta_2=0.999, epsilon=1e-8):
        self.params = [param for param in params]
        self.learning_rate_init = float(learning_rate_init)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.ms = [np.zeros_like(param) for param in params]
        self.vs = [np.zeros_like(param) for param in params]

    def update_params(self, grads):
        """Update parameters with given gradients
        Parameters
        ----------
        grads : list, length = len(params)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params
        """
        updates = self._get_updates(grads)
        for param, update in zip(self.params, updates):
            param += update

        return self.params

    def _get_updates(self, grads):
        self.t += 1
        self.ms = [self.beta_1 * m + (1 - self.beta_1) * grad
                   for m, grad in zip(self.ms, grads)]
        self.vs = [self.beta_2 * v + (1 - self.beta_2) * (grad ** 2)
                   for v, grad in zip(self.vs, grads)]
        self.learning_rate = (self.learning_rate_init *
                              np.sqrt(1 - self.beta_2 ** self.t) /
                              (1 - self.beta_1 ** self.t))
        updates = [-self.learning_rate * m / (np.sqrt(v) + self.epsilon)
                   for m, v in zip(self.ms, self.vs)]
        return updates


obj = NeuralNet()

input_layer = x_train[0].shape[0]  # Number of attributes
output_layer = np.unique(y_train).size  # Number of classes
layers = [input_layer, 64, 32, output_layer]
parameters, class_dict = obj.fit(x_train, y_train, x_test, y_test, layers, 0.0001, 2000)

res = classify(x_test, parameters, class_dict, layers)


print('Test accuracy',accuracy_score(y_test, res))

