import numpy as np
import json


class NeuralNet:
    # Neural Network class, used to create and update creatures neural network
    def __init__(self):
        # Load parameters
        settings_file = open("settings.json")
        self.settings = json.load(settings_file)["nn_parameters"]
        settings_file.close()

        # Define NN parameters
        self.parameters = None          # NN parameters, weights and biases

        # Define NN parameters here
        self.activation_function = self.settings["activation_function"]     # desired activation function
        self.n_x = self.settings["num_inputs"]                              # number of desired inputs to the nn
        self.n_y = self.settings["num_outputs"]                             # number of desired outputs from the nn
        self.n_h = self.settings["num_hidden_layers"]                       # number of nodes in hidden layers
        self.bounds = self.settings["bounds"]                               # upper and lower bounds for parameters
        self.noise = self.settings["noise"]                                 # desired noise in NN

        # Create NN
        self.parameters = self.initialize_parameters()

    @staticmethod
    def sigmoid(x_inputs):
        # calculates sigmoid
        # ARGUMENTS:
        # x_inputs - scalar or numpy array of any size
        #
        # RETURNS:
        # sig - scalar or array of x_inputs

        # assert correct input
        assert isinstance(x_inputs, int) or isinstance(x_inputs, np.ndarray)

        sig = 1/(1 + np.exp(-x_inputs))

        return sig

    def initialize_parameters(self):
        # Sets initial weight matrixes to random values between plus and minus the bounds.
        # Biases initialised to zeros.

        # ARGUMENTS:
        # n_x: length of input layer x_inputs.
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

    def forward_propagation(self, *x_inputs):
        # Retrieves parameters and inputs and computes the forward propagation

        # ARGUMENTS:
        # x_inputs - input data size (n_x, 1)
        # Parameters - python dictionary with weight and biases
        # af - either t or s, used to choose which activation function to use

        # RETURNS:
        # Y - Output array of size (n_y, 1)
        # cache - python dictionary containing (Z1, Z2, A1, Y)

        # Assert correct inputs
        if isinstance(x_inputs, tuple):
            x_inputs = list(x_inputs)
        if isinstance(x_inputs, list):
            x_inputs = np.array(x_inputs).reshape(len(x_inputs), 1)
        assert isinstance(x_inputs, np.ndarray)
        assert isinstance(self.parameters, dict)
        assert self.activation_function.lower() == "sigmoid" or self.activation_function.lower() == "tanh"

        # retrieve parameters
        w1 = self.parameters["w1"]
        b1 = self.parameters["b1"]
        w2 = self.parameters["w2"]
        b2 = self.parameters["b2"]

        # Forward propagation
        Z1 = np.dot(w1, x_inputs) + b1

        if self.activation_function.lower() == "sigmoid":
            A1 = self.sigmoid(Z1)
            Z2 = np.dot(w2, A1) + b2
            Y = self.sigmoid(Z2)
        elif self.activation_function.lower() == "tanh":
            A1 = np.tanh(Z1)
            Z2 = np.dot(w2, A1) + b2
            Y = np.tanh(Z2)
        else:
            raise Exception("ERROR: Unknown activation function.")

        assert (Y.shape == (w2.shape[0], x_inputs.shape[1]))

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

        # retrieve parameters
        w1 = self.parameters["w1"]
        b1 = self.parameters["b1"]
        w2 = self.parameters["w2"]
        b2 = self.parameters["b2"]

        # Create a multiplier to increase, decrease or leave parameter alone by given amount
        w1_change = np.multiply(self.settings["parameter_change"],
                                np.multiply(np.random.randint(low=-1, high=2, size=w1.shape), self.bounds[1]))
        w2_change = np.multiply(self.settings["parameter_change"],
                                np.multiply(np.random.randint(low=-1, high=2, size=w2.shape), self.bounds[1]))
        b1_change = np.multiply(self.settings["parameter_change"],
                                np.multiply(np.random.randint(low=-1, high=2, size=b1.shape), self.bounds[1]))
        b2_change = np.multiply(self.settings["parameter_change"],
                                np.multiply(np.random.randint(low=-1, high=2, size=b2.shape), self.bounds[1]))

        # Increase parameters by multiplier and add noise
        w1 = np.add(w1, w1_change) + np.multiply(np.random.uniform(low=self.bounds[0], high=self.bounds[1],
                                                                   size=list(w1.shape)), self.noise)
        w2 = np.add(w2, w2_change) + np.multiply(np.random.uniform(low=self.bounds[0], high=self.bounds[1],
                                                                   size=list(w2.shape)), self.noise)
        b1 = np.add(b1, b1_change) + np.multiply(np.random.uniform(low=self.bounds[0], high=self.bounds[1],
                                                                   size=list(b1.shape)), self.noise)
        b2 = np.add(b2, b2_change) + np.multiply(np.random.uniform(low=self.bounds[0], high=self.bounds[1],
                                                                   size=list(b2.shape)), self.noise)

        # dict of updated parameters
        updated_params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

        # If parameters go outside of bounds set them back to bounds
        for key, param in updated_params.items():
            updated_params[key] = np.where(param < self.settings["bounds"][0], self.settings["bounds"][0], param)
            updated_params[key] = np.where(param > self.settings["bounds"][1], self.settings["bounds"][1], param)

        self.parameters = updated_params

