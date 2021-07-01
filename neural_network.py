import numpy as np
import json


class NeuralNet:
    # Neural Network class, used to create and update creatures neural network
    def __init__(self, num_inputs):
        # Load parameters
        settings_file = open("settings.json")
        self.settings = json.load(settings_file)["neural_network_parameters"]
        settings_file.close()

        # Define NN parameters
        self.parameters = None          # NN parameters, weights and biases

        # Define NN parameters here
        self.activation_function = self.settings["activation_function"]     # desired activation function, than or sigmoid
        self.num_outputs = self.settings["num_outputs"]                     # number of desired outputs from the nn
        self.num_hidden_layers = self.settings["num_hidden_layers"]         # number of nodes in hidden layers
        self.bounds = self.settings["bounds"]                               # upper and lower bounds for parameters
        self.noise = self.settings["noise"]                                 # desired noise in NN

        # Create NN
        self.parameters = self.initialize_parameters(num_inputs, self.num_outputs, self.num_hidden_layers, self.bounds)

    def forward_propagation(self, inputs):
        outputs = self.forward_propagation(inputs, self.parameters, self.activation_function)
        return outputs

    def update_neural_net(self):
        # Update parameters
        self.parameters = self.update_params_gaussian(self.parameters, self.bounds, self.noise)

    def sigmoid(self, X):
        # calculates sigmoid
        # ARGUMENTS:
        # X - scalar or numpy array of any size

        # RETURNS:
        # S - scalar or array of X

        # assert correct input
        assert isinstance(X, int) or isinstance(X, np.ndarray)

        S = 1/(1 + np.exp(-X))

        return S

    def initialize_parameters(self, n_x, n_y, n_h, bounds):
        # Sets initial weight matrixes to random values between plus and minus the bounds.
        # Biases initialised to zeros.

        # ARGUMENTS:
        # n_x: length of input layer X.
        # n_y: length of output layer Y.
        # n_h: length of hidden layer A.
        # bounds: upper and lower bounds for parameters

        # RETURNS:
        # Dictionary containing:
        # w1: weight matrix with shape (n_h, n_x)
        # w2: weight matrix with shape (n_y, n_h)
        # b1: bias matrix with shape (n_h, 1)
        # b2: bias matrix with shape (n_y, 1)

        # Assert correct inputs
        assert len(bounds) == 2
        assert isinstance(n_x, int)
        assert isinstance(n_h, int)
        assert isinstance(n_y, int)

        # set random seed for repeatability
        # np.random.seed(5)

        # Initialize matrixes:
        w1 = np.random.uniform(low=bounds[0], high=bounds[1], size=(n_h, n_x))
        w2 = np.random.uniform(low=bounds[0], high=bounds[1], size=(n_y, n_h))
        b1 = np.zeros((n_h, 1))
        b2 = np.zeros((n_y, 1))

        # assert that the shapes of matrixes are correct
        assert (w1.shape == (n_h, n_x))
        assert (b1.shape == (n_h, 1))
        assert (w2.shape == (n_y, n_h))
        assert (b2.shape == (n_y, 1))

        init_params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

        return init_params

    def forward_propagation(self, X, parameters, af):
        # Retrieves parameters and inputs and computes the forward propagation

        # ARGUMENTS:
        # X - input data size (n_x, 1)
        # Parameters - python dictionary with weight and biases
        # af - either t or s, used to choose which activation fucntion to use

        # RETURNS:
        # Y - Output array of size (n_y, 1)
        # cache - python dictionary containing (Z1, Z2, A1, Y)

        # Assert correct inputs
        if isinstance(X, tuple):
            X = list(X)
        if isinstance(X, list):
            X = np.array(X).reshape(len(X), 1)
        assert isinstance(X, np.ndarray)
        assert isinstance(parameters, dict)
        assert af.lower() == "s" or af.lower() == "t"

        # retrive parameters
        w1 = parameters["w1"]
        b1 = parameters["b1"]
        w2 = parameters["w2"]
        b2 = parameters["b2"]

        # Forward propagation
        Z1 = np.dot(w1, X) + b1

        if af.lower() == "s":
            A1 = sigmoid(Z1)
            Z2 = np.dot(w2, A1) + b2
            Y = sigmoid(Z2)
        elif af.lower() == "t":
            A1 = np.tanh(Z1)
            Z2 = np.dot(w2, A1) + b2
            Y = np.tanh(Z2)

        assert (Y.shape == (w2.shape[0], X.shape[1]))

        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "Y": Y}

        return Y[0][0], cache

    def update_params_gaussian(self, parameters, bounds, noise):
        # Updates with random gaussian noise
        # ARGUMENTS:
        # Parameters - python dictionary of weight and bias parameters
        # bounds: upper and lower values for parameters
        # noise -  Gaussian noise level

        #  RETURNS:
        # updated_parameters - python dictionary of updated parameters

        # Assert correct inputs
        assert isinstance(parameters, dict)
        assert len(bounds) == 2
        assert isinstance(noise, float)

        # retrive parameters
        w1 = parameters["w1"]
        b1 = parameters["b1"]
        w2 = parameters["w2"]
        b2 = parameters["b2"]

        # Add noise to parameters
        w1 = w1 + np.multiply(np.random.uniform(low=bounds[0], high=bounds[1], size=list(w1.shape)), noise)
        w2 = w2 + np.multiply(np.random.uniform(low=bounds[0], high=bounds[1], size=list(w2.shape)), noise)
        b1 = b1 + np.multiply(np.random.uniform(low=bounds[0], high=bounds[1], size=list(b1.shape)), noise)
        b2 = b2 + np.multiply(np.random.uniform(low=bounds[0], high=bounds[1], size=list(b2.shape)), noise)

        updated_params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

        return updated_params
