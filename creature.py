import csv
import json

import numpy as np

import voxelCreator as vC
import nn

# The following file contains class parameters for creatures and their neural networks. To edit default values refer to
# settings.json.


class CreatureFile:
    def __init__(self, genome, morphology, structure, index):
        # When called initializes creature file, setting default values to creature.
        # ARGUMENTS
        # - genome      float, creature genome
        # - morphology  np.ndarray, creature starting morphology
        # - structure   (1,3) list, list with dimensions of creature morphology. i.e. (6, 6, 6)
        # - index       int, index item for creature naming

        settings_file = open("settings.json")
        mat_defaults = json.load(settings_file)["mat_defaults"]
        settings_file.close()

        # Creature structure properties
        self.structure = structure          # (1,3) list, creatures structure
        self.morphology = morphology        # (z, x*y) np.ndarray, creatures morphology (where x, y, z are values from structure list)
        self.genome = genome                # int, creatures genome

        # Basic creature information
        self.index = index                  # int, index used to create creature name
        self.episode = None                 # int, saves creatures current episode number
        self.generation = None              # int, saves creatures current generation number
        self.name = self.name = "_creature" + str(self.index)  # string, Set creature's name

        # Creature material properties (Initially set to defaults), please refer to settings.json to see definitions


        self.number_of_materials = mat_defaults["number_of_materials"]
        self.integration = mat_defaults["integration"]
        self.damping = mat_defaults["damping"]
        self.collision = mat_defaults["collision"]
        self.features = mat_defaults["features"]
        self.stopConditions = mat_defaults["stopConditions"]
        self.drawSmooth = mat_defaults["drawSmooth"]
        self.write_fitness = mat_defaults["write_fitness"]
        self.QhullTmpFile = mat_defaults["QhullTmpFile"]
        self.CurvaturesTmpFile = mat_defaults["CurvaturesTmpFile"]
        self.numFixed = mat_defaults["numFixed"]
        self.numForced = mat_defaults["numForced"]
        self.gravity = mat_defaults["gravity"]
        self.thermal = mat_defaults["thermal"]
        self.version = mat_defaults["version"]
        self.lattice = mat_defaults["lattice"]
        self.voxel = mat_defaults["voxel"]
        self.mat_type = mat_defaults["mat_type"]
        self.mat_colour = mat_defaults["mat_colour"]
        self.mechanical_properties = mat_defaults["mechanical_properties"]
        self.compression_type = mat_defaults["compression_type"]
        self.phase_offset = mat_defaults["phase_offset"]
        self.stiffness_array = mat_defaults["stiffness_array"]

        # file name variables
        self.current_file_name = None       # string, current VXA file name
        self.fitness_file_name = None       # string, current fitness file name
        self.pressures_file_name = None     # string, current pressures file name
        self.ke_file_name = None            # string, current ke file name
        self.strain_file_name = None        # string, current strain file name

        # Fitness variables
        self.previous_fitness = 0.0         # float, previous fitness, used to calculate fitness between episodes
        self.fitness_xyz = None             # (1,3) list, fitness
        self.fitness_eval = 0.0             # float, creatures evaluated fitness
        self.displacement_delta = 0.0       # float, creatures displacement
        self.average_forces = None          # (z, x*y) list, average forces acting on voxels. (where x, y, z are values from structure list)

        # evolution
        self.neural_net = None              # class, uses NeuralNet class to create nn for creature
        self.evolution = {}                 # dict, used to save creatures evolutionary history

    def update_creature_info(self, generation, episode):
        # Updates basic creature information
        # ARGUMENTS
        # - generation      int, used to update creature generation number
        # - episode         int, used to update creature episode number

        # Update information
        if not self.generation == generation:
            self.generation = generation
        self.episode = episode
        self.current_file_name = self.name + "_gen" + str(self.generation) + "_ep" + str(self.episode)
        self.fitness_file_name = self.current_file_name + "_fitness.xml"
        self.pressures_file_name = "pressures" + self.fitness_file_name + ".csv"
        self.ke_file_name = "ke" + self.fitness_file_name + ".csv"
        self.strain_file_name = "strain" + self.fitness_file_name + ".csv"

    def createVXA(self, generation, episode, number_of_materials=None, integration=None, damping=None, collision=None,
                  features=None, stopConditions=None, drawSmooth=None, write_fitness=None, QhullTmpFile=None,
                  CurvaturesTmpFile=None, numFixed=None, numForced=None, gravity=None, thermal=None, version=None,
                  lattice=None, voxel=None, mat_type=None, mat_colour=None, mechanical_properties=None,
                  compression_type=None, phase_offset=None, stiffness_array=None):

        # Using creature material properties, creates readable VXA file for execution on Voxelyze. See settings.json
        # for more explanation on parameters.
        #
        # ARGUMENTS
        # - generation              int, new creature generation number
        # - episode                 int, new creature episode number
        # - number_of_materials     int, used to create a list of creature materials
        # - integration             (1, 2) list, used to set integration parameters in vxa file
        # - damping                 (1, 3) list, used to set damping parameters in vxa file
        # - collision               (1, 3) list, used to set collision parameters in vxa file
        # - features                (1, 3) list, used to set features parameters in vxa file
        # - stopConditions          (1, 3) list, used to set stopConditions parameters in vxa file
        # - drawSmooth              int, used to set drawSmooth parameter in vxa file
        # - write_fitness           int, used to set write_fitness parameter in vxa file
        # - QhullTmpFile            string, used to set QhullTmpFile parameter in vxa file
        # - CurvaturesTmpFile       string, used to set CurvaturesTmpFile parameter in vxa file
        # - numFixed                int, used to set numFixed parameter in vxa file
        # - numForced               int, used to set numForced parameter in vxa file
        # - gravity                 (1, 6) list, used to set gravity parameters in vxa file
        # - thermal                 (1, 5) list, used to set thermal parameters in vxa file
        # - version                 string, used to set version parameter in vxa file
        # - lattice                 (1, 8) list, used to set lattice parameters in vxa file
        # - voxel                   (1, 4) list, used to set voxel parameters in vxa file
        # - mat_type                int, used to set mat_type parameter in vxa file
        # - mat_colour              (1, number_of_materials) list, used to set mat_colour parameters in vxa file
        # - mechanical_properties   (1, 13) list, used to set mechanical_properties parameters in vxa file
        # - compression_type        string, used to set compression_type parameter in vxa file
        # - phase_offset            int, used to set phase_offset parameter in vxa file
        # - stiffness_array         np.ndarray, used to set stiffness_array in vxa file
        #
        # RETURNS
        # - vxa_file                string, containing vxa file

        # update file name before creating vxa
        self.update_creature_info(generation, episode)

        # If property given change self.PROPERTY_NAME
        if number_of_materials is not None:
            self.number_of_materials = number_of_materials
        if integration is not None:
            self.integration = integration
        if damping is not None:
            self.damping = damping
        if collision is not None:
            self.collision = collision
        if features is not None:
            self.features = features
        if stopConditions is not None:
            self.stopConditions = stopConditions
        if drawSmooth is not None:
            self.drawSmooth = drawSmooth
        if write_fitness is not None:
            self.write_fitness = write_fitness
        if QhullTmpFile is not None:
            self.QhullTmpFile = QhullTmpFile
        if CurvaturesTmpFile is not None:
            self.CurvaturesTmpFile = CurvaturesTmpFile
        if numFixed is not None:
            self.numFixed = numFixed
        if numForced is not None:
            self.numForced = numForced
        if gravity is not None:
            self.gravity = thermal
        if version is not None:
            self.version = version
        if lattice is not None:
            self.lattice = lattice
        if voxel is not None:
            self.voxel = voxel
        if mat_type is not None:
            self.mat_type = mat_type
        if mat_colour is not None:
            self.mat_colour = mat_colour
        if mechanical_properties is not None:
            self.mechanical_properties = mechanical_properties
        if compression_type is not None:
            self.compression_type = compression_type
        if phase_offset is not None:
            self.phase_offset = phase_offset
        if stiffness_array is not None:
            self.stiffness_array = stiffness_array

        # Set Genetic Algorithm variable
        GA = [self.write_fitness, self.fitness_file_name, self.QhullTmpFile, self.CurvaturesTmpFile]

        # Call VXA_CREATE and get VXA file text
        init_text = vC.init_creator(self.integration, self.damping, self.collision, self.features, self.stopConditions,
                                    self.drawSmooth, GA)

        env_text = vC.environment_creator(self.numFixed, self.numForced, self.gravity, self.thermal)

        vxc_text = vC.vxc_creator(self.version, self.lattice, self.voxel)

        for mat_ID in range(1, self.number_of_materials + 1):
            mat_name = "mat_" + str(mat_ID)
            voxel_text = vC.voxel_creator(mat_ID, self.mat_type, mat_name, self.mat_colour, self.mechanical_properties)

        struc_text = vC.structure_creator(self.compression_type, self.structure, self.morphology)

        stiff_text = vC.stiffness_creator(self.stiffness_array)

        offset_text = vC.offset_creator(self.structure, self.phase_offset)

        end_text = vC.end_creator()

        # save vxa file
        self.vxa_file = init_text + env_text + vxc_text + voxel_text + struc_text + offset_text + stiff_text + end_text

        return self.vxa_file

    def update_fitness(self):
        # Evaluates creatures fitness by reading the saved fitness file and saves fitness evaluation. Punishes creature
        # for displacement in y axis

        mod = 3  # modifier for severity of punishment when creature has y displacement

        # Open file and retrieve fitness values
        tag_y = "<normDistY>"
        tag_x = "<normDistX>"
        tag_z = "<normDistZ>"
        with open(self.fitness_file_name) as fit_file:
            for line in fit_file:
                if tag_y in line:
                    result_Y = abs(float(line.replace(tag_y, "").replace("</" + tag_y[1:], "")))
                if tag_x in line:
                    result_X = float(line.replace(tag_x, "").replace("</" + tag_x[1:], ""))
                if tag_z in line:
                    result_Z = abs(float(line.replace(tag_z, "").replace("</" + tag_z[1:], "")))
        fit_file.close()

        # Save fitness values
        self.fitness_xyz = [result_X, result_Y, result_Z]

        # Calculate fitness, punish for locomotion that is not in a straight line
        self.fitness_eval = self.fitness_xyz[0] - self.fitness_xyz[1]*mod

    def update_evolution(self):
        # Updates creatures evolutionary history ands saves key information within its self.evolution dictionary.
        # Allows historical values to be retried at any point after simulations have completed.

        if ("gen_" + str(self.generation)) not in self.evolution:
            self.evolution[("gen_" + str(self.generation))] = {}

        if ("ep_" + str(self.episode)) not in self.evolution[("gen_" + str(self.generation))]:
            self.evolution["gen_" + str(self.generation)][("ep_" + str(self.episode))] = {}

        # Update morphology and stiffness change
        self.evolution[("gen_" + str(self.generation))][("ep_" + str(self.episode))]["morphology"] = self.morphology
        if self.stiffness_array is not None:
            self.evolution["gen_" + str(self.generation)]["ep_" + str(self.episode)].update({"stiffness": self.stiffness_array})

        # Update fitness values in evolution
        self.evolution["gen_" + str(self.generation)]["ep_" + str(self.episode)].update({"fitness_xyz": self.fitness_xyz})
        self.evolution["gen_" + str(self.generation)]["ep_" + str(self.episode)].update({"fitness_eval": self.fitness_eval})
        self.evolution["gen_" + str(self.generation)]["ep_" + str(self.episode)].update({"fitness_eval": self.fitness_eval})
        self.evolution["gen_" + str(self.generation)]["ep_" + str(self.episode)].update({"average_forces": self.average_forces})
        self.evolution["gen_" + str(self.generation)].update({"nn_parameters": self.neural_net.parameters})

    def update_stiffness(self):
        # Uses artificial neural network to update the creatures morphology and stiffness array.

        settings_file = open("settings.json")
        struc_params = json.load(settings_file)["creature_structure"]
        settings_file.close()

        # Calculate displacement since last evaluation
        self.displacement_delta = (self.fitness_eval - self.previous_fitness)/10

        # If fitness is getting worse, ensure delta is negative
        if (self.previous_fitness > self.fitness_eval) and self.displacement_delta > 0:
            self.displacement_delta = self.displacement_delta * -1

        # Reset average force
        average_forces = np.zeros((self.structure[2], self.structure[0]*self.structure[1]))

        # Format KE file
        with open(self.ke_file_name) as kef:
            ke_data = csv.reader(kef)
            for row in ke_data:
                row_data = np.multiply(np.array(row[:-1], dtype=np.float), 10)
                row_array = np.reshape(row_data, (self.structure[2], self.structure[0]*self.structure[1]))
                average_forces += row_array
        kef.close()

        # set self.average_forces vector and calculate ultimate avg
        self.average_forces = np.divide(average_forces, np.prod(self.structure))
        ultimate_average = np.sum(self.average_forces)/np.prod(self.structure)

        input_len = 3  # Set the len of the inputs
        if self.neural_net is None:
            self.neural_net = NeuralNet(input_len)

        # Calculate difference between ultimate avg and the average force on each voxel
        ke_delta = np.multiply(np.subtract(ultimate_average, self.average_forces), 10)

        # Update evolutionary history of creature before calculation and updating stiffness
        self.update_evolution()

        # Use NN to update the stiffness array of the creature
        updated_stiffness = []
        for row in ke_delta:
            updated_stiffness.append([self.neural_net.forward_propagation(
                (elem, self.displacement_delta, self.genome))[0]*struc_params["min_stiffness"] for elem in row])
        updated_stiffness = np.array(updated_stiffness)
        new_stiffness = self.stiffness_array + updated_stiffness

        # Use stiffness array to create new morphology
        new_morphology = np.ones(self.morphology.shape)*struc_params["morph_min"]
        new_morphology = np.where(new_stiffness > struc_params["max_stiffness"], struc_params["morph_max"], new_morphology)
        new_morphology = np.where(np.logical_and(new_stiffness > struc_params["min_stiffness"],
                                                 new_stiffness < struc_params["max_stiffness"]),
                                  struc_params["morph_between"], new_morphology)

        # Set limits to stiffness
        new_stiffness[new_stiffness < struc_params["min_stiffness"]] = struc_params["min_stiffness"]
        new_stiffness[new_stiffness > struc_params["max_stiffness"]] = struc_params["max_stiffness"]

        # Get range of cells where the morphology was previously 4 (actuator)
        new_morphology = np.where((self.morphology == struc_params["actuator_morph"]), struc_params["actuator_morph"], new_morphology)
        new_stiffness = np.where(self.morphology == struc_params["actuator_morph"], struc_params["actuator_stiffness"], new_stiffness)

        # Update stiffness
        self.morphology = new_morphology
        self.stiffness_array = new_stiffness

        # Update old fitness with new fitness
        self.previous_fitness = self.fitness_eval
    
    def evolve(self):
        # Evolves creatures artificial neural network. (Varies biases and weights)
        # RETURNS
        # self          class, creature

        # Update neural network
        self.neural_net.update_neural_net()
        return self


class NeuralNet:
    # Neural Network class, used to create and update creatures neural network
    def __init__(self, num_inputs):
        # Load parameters
        settings_file = open("settings.json")
        nn_params = json.load(settings_file)["neural_network_parameters"]
        settings_file.close()

        # Define NN parameters
        self.parameters = None          # NN parameters, weights and biases

        # Define NN parameters here
        self.activation_function = nn_params["activation_function"]     # desired activation function, than or sigmoid
        self.num_outputs = nn_params["num_outputs"]                     # number of desired outputs from the nn
        self.num_hidden_layers = nn_params["num_hidden_layers"]         # number of nodes in hidden layers
        self.bounds = nn_params["bounds"]                               # upper and lower bounds for parameters
        self.noise = nn_params["noise"]                                 # desired noise in NN

        # Create NN
        self.parameters = nn.initialize_params(num_inputs, self.num_outputs, self.num_hidden_layers, self.bounds)

    def forward_propagation(self, inputs):
        outputs = nn.forward_propagation(inputs, self.parameters, self.activation_function)
        return outputs

    def update_neural_net(self):
        # Update parameters
        self.parameters = nn.update_params_gaussian(self.parameters, self.bounds, self.noise)
