import numpy as np
import json


class NeuralNet:
    # Neural Network class, used to create and update creatures neural network
    def __init__(self, num_inputs):
        # Load parameters
        settings_file = open("settings.json")
        self.settings = json.load(settings_file)["nn_parameters"]
        settings_file.close()

        # Define NN parameters
        self.parameters = None          # NN parameters, weights and biases

        # Define NN parameters here
        self.activation_function = self.settings["activation_function"]     # desired activation function, than or sigmoid
        self.n_x = self.settings["num_inputs"]                             # number of desired inputs to the nn
        self.n_y = self.settings["num_outputs"]                             # number of desired outputs from the nn
        self.n_h = self.settings["num_hidden_layers"]                       # number of nodes in hidden layers
        self.bounds = self.settings["bounds"]                               # upper and lower bounds for parameters
        self.noise = self.settings["noise"]                                 # desired noise in NN

        # Create NN
        self.parameters = self.initialize_parameters()

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

    def initialize_parameters(self):
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
        assert len(self.bounds) == 2
        assert isinstance(self.n_x, int)
        assert isinstance(self.n_h, int)
        assert isinstance(self.n_y, int)

        # set random seed for repeatability
        # np.random.seed(5)

        # Initialize matrixes:
        w1 = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=(self.n_h, self.n_x))
        w2 = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=(self.n_y, self.n_h))
        b1 = np.zeros((self.n_h, 1))
        b2 = np.zeros((self.n_y, 1))

        # assert that the shapes of matrixes are correct
        assert (w1.shape == (self.n_h, self.n_x))
        assert (b1.shape == (self.n_h, 1))
        assert (w2.shape == (self.n_y, self.n_h))
        assert (b2.shape == (self.n_y, 1))

        init_params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

        return init_params

    def forward_propagation(self, *X):
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
        assert isinstance(self.parameters, dict)
        assert self.activation_function.lower() == "s" or self.activation_function.lower() == "t"

        # retrive parameters
        w1 = self.parameters["w1"]
        b1 = self.parameters["b1"]
        w2 = self.parameters["w2"]
        b2 = self.parameters["b2"]

        # Forward propagation
        Z1 = np.dot(w1, X) + b1

        if self.activation_function.lower() == "s":
            A1 = self.sigmoid(Z1)
            Z2 = np.dot(w2, A1) + b2
            Y = self.sigmoid(Z2)
        elif self.activation_function.lower() == "t":
            A1 = np.tanh(Z1)
            Z2 = np.dot(w2, A1) + b2
            Y = np.tanh(Z2)

        assert (Y.shape == (w2.shape[0], X.shape[1]))

        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "Y": Y}

        return Y[0][0], cache

    def update_neural_net(self):
        # Updates NN with random gaussian noise
        # ARGUMENTS:
        # Parameters - python dictionary of weight and bias parameters
        # bounds: upper and lower values for parameters
        # noise -  Gaussian noise level forward

        #  RETURNS:
        # updated_parameters - python dictionary of updated parameters

        # Assert correct inputs
        assert isinstance(self.parameters, dict)
        assert len(self.bounds) == 2
        assert isinstance(self.noise, float)

        # retrive parameters
        w1 = self.parameters["w1"]
        b1 = self.parameters["b1"]
        w2 = self.parameters["w2"]
        b2 = self.parameters["b2"]

        # Add noise to parameters
        w1 = w1 + np.multiply(np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=list(w1.shape)), self.noise)
        w2 = w2 + np.multiply(np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=list(w2.shape)), self.noise)
        b1 = b1 + np.multiply(np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=list(b1.shape)), self.noise)
        b2 = b2 + np.multiply(np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=list(b2.shape)), self.noise)

        updated_params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

        return updated_params
