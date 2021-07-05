import csv
import json
import random

import numpy as np

from evosoro import Evosoro
from neural_network import NeuralNet


class Creature:
    def __init__(self, genome, index):
        # When called initializes creature file, setting default values to creature.
        # ARGUMENTS
        # - genome      float, creature genome
        # - index       int, index item for creature naming

        # import settings
        settings_file = open("settings.json")
        self.settings = json.load(settings_file)
        settings_file.close()

        # Creature genotype and phenotype
        self.genome = genome                # int, creatures genome
        self.phenotype = Evosoro()          # class, creature phenotype

        # Creature initial stiffness
        self.stiffness_array = np.multiply(np.ones((self.phenotype.structure[2], self.phenotype.structure[0]
                                                    * self.phenotype.structure[1])), self.settings["structure"]["base_stiffness"])

        # Basic creature information
        self.index = index                  # int, index used to create creature name
        self.episode = None                 # int, saves creatures current episode number
        self.generation = None              # int, saves creatures current generation number
        self.name = self.name = "_creature" + str(self.index)  # string, Set creature's name

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

    def update_vxa(self, generation, episode):

        # update file name before creating vxa
        self.update_creature_info(generation, episode)

        # Update vxa file
        self.phenotype.update_vxa_file(self)

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
        self.evolution[("gen_" + str(self.generation))][("ep_" + str(self.episode))]["morphology"] = self.phenotype.morphology
        if self.stiffness_array is not None:
            self.evolution["gen_" + str(self.generation)]["ep_" + str(self.episode)].update({"stiffness": self.stiffness_array})

        # Update fitness values in evolution
        self.evolution["gen_" + str(self.generation)]["ep_" + str(self.episode)].update({"fitness_xyz": self.fitness_xyz})
        self.evolution["gen_" + str(self.generation)]["ep_" + str(self.episode)].update({"fitness_eval": self.fitness_eval})
        self.evolution["gen_" + str(self.generation)]["ep_" + str(self.episode)].update({"average_forces": self.average_forces})
        self.evolution["gen_" + str(self.generation)].update({"nn_parameters": self.neural_net.parameters})

    def update_stiffness(self):
        # Uses artificial neural network to update the creatures morphology and stiffness array.

        # Calculate displacement since last evaluation
        self.displacement_delta = (self.fitness_eval - self.previous_fitness)/10

        # If fitness is getting worse, ensure delta is negative
        if (self.previous_fitness > self.fitness_eval) and self.displacement_delta > 0:
            self.displacement_delta = self.displacement_delta * -1

        # Reset average force
        average_forces = np.zeros((self.phenotype.structure[2], self.phenotype.structure[0]*self.phenotype.structure[1]))

        # Format KE file
        with open(self.ke_file_name) as kef:
            ke_data = csv.reader(kef)
            for row in ke_data:
                row_data = np.multiply(np.array(row[:-1], dtype=np.float), 10)
                row_array = np.reshape(row_data, (self.phenotype.structure[2],
                                                  self.phenotype.structure[0]*self.phenotype.structure[1]))
                average_forces += row_array
        kef.close()

        # set self.average_forces vector and calculate ultimate avg
        self.average_forces = np.divide(average_forces, np.prod(self.phenotype.structure))
        ultimate_average = np.sum(self.average_forces)/np.prod(self.phenotype.structure)

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
                (elem, self.displacement_delta, self.genome))[0]*self.settings["structure"]["min_stiffness"] for elem in row])
        updated_stiffness = np.array(updated_stiffness)
        new_stiffness = self.stiffness_array + updated_stiffness

        # Use stiffness array to create new morphology
        new_morphology = np.ones(self.phenotype.morphology.shape)*self.settings["structure"]["morph_min"]
        new_morphology = np.where(new_stiffness > self.settings["structure"]["max_stiffness"],
                                  self.settings["structure"]["morph_max"], new_morphology)
        new_morphology = np.where(np.logical_and(new_stiffness > self.settings["structure"]["min_stiffness"],
                                                 new_stiffness < self.settings["structure"]["max_stiffness"]),
                                  self.settings["structure"]["morph_between"], new_morphology)

        # Set limits to stiffness
        new_stiffness[new_stiffness < self.settings["structure"]["min_stiffness"]]\
            = self.settings["structure"]["min_stiffness"]
        new_stiffness[new_stiffness > self.settings["structure"]["max_stiffness"]]\
            = self.settings["structure"]["max_stiffness"]

        # Get range of cells where the morphology was previously 4 (actuator)
        new_morphology = np.where((self.phenotype.morphology == self.settings["structure"]["actuator_morph"]),
                                  self.settings["structure"]["actuator_morph"], new_morphology)
        new_stiffness = np.where(self.phenotype.morphology == self.settings["structure"]["actuator_morph"],
                                 self.settings["structure"]["actuator_stiffness"], new_stiffness)

        # Update stiffness
        self.phenotype.morphology = new_morphology
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
    
    def fix_morphology(self):
        # While evolving the creature, voxels can become isolated from any adjacent voxels. This causes errors in the simulater as these voxels
        # drop off the creature and to the floor. If this occurs, the isolated voxels should be removed and the morphology and stiffness_array updated accordingly.
        
        # Create copy of morphology with additional zeros around the morphology 
        temp_morph = np.zeros((self.phenotype.morphology.shape[0] + 2, self.phenotype.morphology.shape[1] + self.phenotype.structure[1]*2))
        temp_morph[1:-1, self.phenotype.structure[1]:-self.phenotype.structure[1]] = self.phenotype.morphology
        
        # Iterate through each element of creatures morphology
        for indx, elem in np.ndenumerate(self.phenotype.morphology):
            # Turn index into array and offset index to iterate through temp_morphology
            indx = np.array(list(indx))
            temp_indx = indx + [1, self.phenotype.structure[1]]

            # Create empty array of the values of all adjacent voxels.
            adj_voxels = []
            # append adjacent voxel values to list
            adj_voxels.append(adj_voxels[temp_indx[0], temp_indx[1]-1])
            adj_voxels.append(adj_voxels[temp_indx[0], temp_indx[1]+1])
            adj_voxels.append(adj_voxels[temp_indx[0], temp_indx[1]-6])
            adj_voxels.append(adj_voxels[temp_indx[0], temp_indx[1]+6])
            adj_voxels.append(adj_voxels[temp_indx[0]+1, temp_indx[1]])
            adj_voxels.append(adj_voxels[temp_indx[0]-1, temp_indx[1]])

            # if all the values of the adj_voxels are 0 (i.e. they don't exist) remove voxel from morphology
            if not any(adj_voxels):
                # Update morphology and stiffness
                self.phenotype.morphology[indx[0], indx[1]] = 0
                self.stiffness_array[indx[0], indx[1]] = self.settings["structure"]["min_stiffness"]

    def remove_eighths(self):
        # Removes eight sections of the creature. Updates self.eight_damage_morph dictionary
        # ARGUMENTS:
        # - creature                class (creature), creature information

        # get voxel dimensions
        x_voxels = self.phenotype.structure[0]
        y_voxels = self.phenotype.structure[1]
        z_voxels = self.phenotype.structure[2]

        # set empty arrays for the four quarters of the damage
        damage_1 = np.ones([x_voxels, y_voxels], dtype=int)
        damage_2 = np.ones([x_voxels, y_voxels], dtype=int)
        damage_3 = np.ones([x_voxels, y_voxels], dtype=int)
        damage_4 = np.ones([x_voxels, y_voxels], dtype=int)

        # set empty arrays for damage of the eighths
        damaged_morph_1 = []
        damaged_morph_2 = []
        damaged_morph_3 = []
        damaged_morph_4 = []
        damaged_morph_5 = []
        damaged_morph_6 = []
        damaged_morph_7 = []
        damaged_morph_8 = []

        # setting damaged areas
        for ii in range(y_voxels):
            if ii < y_voxels / 2:
                for jj in range(x_voxels):
                    if jj < x_voxels / 2:
                        damage_1[ii, jj] = 0
                    else:
                        damage_2[ii, jj] = 0
            else:
                for jj in range(x_voxels):
                    if jj < x_voxels / 2:
                        damage_3[ii, jj] = 0
                    else:
                        damage_4[ii, jj] = 0

        if x_voxels % 2 == 0 and y_voxels % 2 == 0 and z_voxels % 2 == 0:
            # Cube region with even lengths x_voxels, y_voxels and z_voxels
            for k in range(z_voxels):
                row = self.phenotype.morphology[k]

                if k < z_voxels / 2:
                    # convert to array
                    row_array = np.array(np.array_split(row, x_voxels))

                    # Apply damage to each row
                    row_damage_1 = np.multiply(row_array, damage_1)
                    row_damage_2 = np.multiply(row_array, damage_2)
                    row_damage_3 = np.multiply(row_array, damage_3)
                    row_damage_4 = np.multiply(row_array, damage_4)

                    # Convert to list
                    row_damg_1 = [elem for lst in row_damage_1.tolist() for elem in lst]
                    row_damg_2 = [elem for lst in row_damage_2.tolist() for elem in lst]
                    row_damg_3 = [elem for lst in row_damage_3.tolist() for elem in lst]
                    row_damg_4 = [elem for lst in row_damage_4.tolist() for elem in lst]

                    # Apply damage to layers
                    damaged_morph_1.append(row_damg_1)
                    damaged_morph_2.append(row_damg_2)
                    damaged_morph_3.append(row_damg_3)
                    damaged_morph_4.append(row_damg_4)
                    # Append unaffected layers
                    damaged_morph_5.append(row.tolist())
                    damaged_morph_6.append(row.tolist())
                    damaged_morph_7.append(row.tolist())
                    damaged_morph_8.append(row.tolist())

                else:
                    # convert to array
                    row_array = np.array(np.array_split(row, x_voxels))

                    # Apply damage to each row
                    row_damage_5 = np.multiply(row_array, damage_1)
                    row_damage_6 = np.multiply(row_array, damage_2)
                    row_damage_7 = np.multiply(row_array, damage_3)
                    row_damage_8 = np.multiply(row_array, damage_4)

                    # Convert to list
                    row_damg_5 = [elem for lst in row_damage_5.tolist() for elem in lst]
                    row_damg_6 = [elem for lst in row_damage_6.tolist() for elem in lst]
                    row_damg_7 = [elem for lst in row_damage_7.tolist() for elem in lst]
                    row_damg_8 = [elem for lst in row_damage_8.tolist() for elem in lst]

                    # Append unaffected layers
                    damaged_morph_1.append(row.tolist())
                    damaged_morph_2.append(row.tolist())
                    damaged_morph_3.append(row.tolist())
                    damaged_morph_4.append(row.tolist())
                    # Apply damage to layers
                    damaged_morph_5.append(row_damg_5)
                    damaged_morph_6.append(row_damg_6)
                    damaged_morph_7.append(row_damg_7)
                    damaged_morph_8.append(row_damg_8)

        creatures_eighths_damage = {"eighths_creature_1": damaged_morph_1,
                                    "eighths_creature_2": damaged_morph_2,
                                    "eighths_creature_3": damaged_morph_3,
                                    "eighths_creature_4": damaged_morph_4,
                                    "eighths_creature_5": damaged_morph_5,
                                    "eighths_creature_6": damaged_morph_6,
                                    "eighths_creature_7": damaged_morph_7,
                                    "eighths_creature_8": damaged_morph_8
                                    }

        # Update creature morphology and set stiffness to min_stiffness if morphology is at 0
        self.phenotype.morphology = np.array(random.choice(list(creatures_eighths_damage.values())))
        self.stiffness_array = np.where(self.phenotype.morphology == 0, self.settings["structure"]["min_stiffness"],
                                        self.stiffness_array)



    def remove_halfs(self):

        # get voxel dimensions
        x_voxels = self.phenotype.structure[0]
        y_voxels = self.phenotype.structure[1]
        z_voxels = self.phenotype.structure[2]

        # set empty arrays for the four quarters of the damage
        damage_1 = np.ones([x_voxels, y_voxels], dtype=int)
        damage_2 = np.ones([x_voxels, y_voxels], dtype=int)
        damage_3 = np.ones([x_voxels, y_voxels], dtype=int)
        damage_4 = np.ones([x_voxels, y_voxels], dtype=int)
        damage_5 = np.zeros([x_voxels, y_voxels], dtype=int)

        # set empty arrays for damage of the halfs
        damaged_morph_1 = []
        damaged_morph_2 = []
        damaged_morph_3 = []
        damaged_morph_4 = []
        damaged_morph_5 = []
        damaged_morph_6 = []

        for ii in range(y_voxels):
            if ii < y_voxels / 2:
                for jj in range(x_voxels):
                    if jj < x_voxels / 2:
                        damage_1[ii, jj] = 0
                    else:
                        damage_2[ii, jj] = 0
            else:
                for jj in range(x_voxels):
                    if jj < x_voxels / 2:
                        damage_3[ii, jj] = 0
                    else:
                        damage_4[ii, jj] = 0

        if x_voxels % 2 == 0 and y_voxels % 2 == 0 and z_voxels % 2 == 0:
            # Cube region with even lengths x_voxels, y_voxels and z_voxels
            for k in range(z_voxels):
                row = self.phenotype.morphology[k]
                if k < z_voxels / 2:
                    row_array = np.array(np.array_split(row, x_voxels))

                    # Apply damage to each row
                    row_damage_1 = np.multiply(row_array, damage_5)
                    row_damage_2 = np.multiply(np.multiply(row_array, damage_1), damage_2)
                    row_damage_3 = np.multiply(np.multiply(row_array, damage_4), damage_3)
                    row_damage_4 = np.multiply(np.multiply(row_array, damage_1), damage_3)
                    row_damage_5 = np.multiply(np.multiply(row_array, damage_2), damage_4)

                    # Convert to list
                    row_damg_1 = [elem for lst in row_damage_1.tolist() for elem in lst]
                    row_damg_2 = [elem for lst in row_damage_2.tolist() for elem in lst]
                    row_damg_3 = [elem for lst in row_damage_3.tolist() for elem in lst]
                    row_damg_4 = [elem for lst in row_damage_4.tolist() for elem in lst]
                    row_damg_5 = [elem for lst in row_damage_5.tolist() for elem in lst]

                    # Apply damage to layers
                    damaged_morph_1.append(row_damg_1)
                    damaged_morph_2.append(row_damg_2)
                    damaged_morph_3.append(row_damg_3)
                    damaged_morph_4.append(row_damg_4)
                    damaged_morph_5.append(row_damg_5)
                    # Append unaffected layers
                    damaged_morph_6.append(row.tolist())

                else:
                    # convert to array
                    row_array = np.array(np.array_split(row, x_voxels))

                    # Apply damage to each row
                    row_damage_2 = np.multiply(np.multiply(row_array, damage_1), damage_2)
                    row_damage_3 = np.multiply(np.multiply(row_array, damage_4), damage_3)
                    row_damage_4 = np.multiply(np.multiply(row_array, damage_1), damage_3)
                    row_damage_5 = np.multiply(np.multiply(row_array, damage_2), damage_4)
                    row_damage_6 = np.multiply(row_array, damage_5)

                    # Convert to list
                    row_damg_2 = [elem for lst in row_damage_2.tolist() for elem in lst]
                    row_damg_3 = [elem for lst in row_damage_3.tolist() for elem in lst]
                    row_damg_4 = [elem for lst in row_damage_4.tolist() for elem in lst]
                    row_damg_5 = [elem for lst in row_damage_5.tolist() for elem in lst]
                    row_damg_6 = [elem for lst in row_damage_6.tolist() for elem in lst]

                    # Append unaffected layers
                    damaged_morph_1.append(row.tolist())
                    # Apply damage to layers
                    damaged_morph_2.append(row_damg_2)
                    damaged_morph_3.append(row_damg_3)
                    damaged_morph_4.append(row_damg_4)
                    damaged_morph_5.append(row_damg_5)
                    damaged_morph_6.append(row_damg_6)

        creatures_halfs_damage = {"half_creature_1": damaged_morph_1,
                                  "half_creature_2": damaged_morph_2,
                                  "half_creature_3": damaged_morph_3,
                                  "half_creature_4": damaged_morph_4,
                                  "half_creature_5": damaged_morph_5,
                                  "half_creature_6": damaged_morph_6,
                                  }

        # Update creature morphology and set stiffness to min_stiffness if morphology is at 0
        self.phenotype.morphology = np.array(random.choice(list(creatures_halfs_damage.values())))
        self.stiffness_array = np.where(self.phenotype.morphology == 0, self.settings["structure"]["min_stiffness"],
                                        self.stiffness_array)

    def remove_quarters(self):

        # get voxel dimensions
        x_voxels = self.phenotype.structure[0]
        y_voxels = self.phenotype.structure[1]
        z_voxels = self.phenotype.structure[2]

        # set empty arrays for the four quarters of the damage
        damage_1 = np.ones([x_voxels, y_voxels], dtype=int)
        damage_2 = np.ones([x_voxels, y_voxels], dtype=int)
        damage_3 = np.ones([x_voxels, y_voxels], dtype=int)
        damage_4 = np.ones([x_voxels, y_voxels], dtype=int)

        # set empty arrays for damage of the quarters
        damaged_morph_1 = []
        damaged_morph_2 = []
        damaged_morph_3 = []
        damaged_morph_4 = []
        damaged_morph_5 = []
        damaged_morph_6 = []
        damaged_morph_7 = []
        damaged_morph_8 = []
        damaged_morph_9 = []
        damaged_morph_10 = []
        damaged_morph_11 = []
        damaged_morph_12 = []

        # setting damaged areas
        for ii in range(y_voxels):
            if ii < y_voxels / 2:
                for jj in range(x_voxels):
                    if jj < x_voxels / 2:
                        damage_1[ii, jj] = 0
                    else:
                        damage_2[ii, jj] = 0
            else:
                for jj in range(x_voxels):
                    if jj < x_voxels / 2:
                        damage_3[ii, jj] = 0
                    else:
                        damage_4[ii, jj] = 0

        if x_voxels % 2 == 0 and y_voxels % 2 == 0 and z_voxels % 2 == 0:
            # Cube region with even lengths x_voxels, y_voxels and z_voxels
            for k in range(z_voxels):
                row = self.phenotype.morphology[k]

                if k < z_voxels / 2:
                    # convert to array
                    row_array = np.array(np.array_split(row, x_voxels))

                    # Apply damage to each row
                    row_damage_1 = np.multiply(row_array, damage_1)
                    row_damage_2 = np.multiply(row_array, damage_2)
                    row_damage_3 = np.multiply(row_array, damage_3)
                    row_damage_4 = np.multiply(row_array, damage_4)
                    row_damage_5 = np.multiply(np.multiply(row_array, damage_1), damage_3)
                    row_damage_6 = np.multiply(np.multiply(row_array, damage_2), damage_4)
                    row_damage_9 = np.multiply(np.multiply(row_array, damage_1), damage_2)
                    row_damage_10 = np.multiply(np.multiply(row_array, damage_3), damage_4)

                    # Convert to list
                    row_damg_1 = [elem for lst in row_damage_1.tolist() for elem in lst]
                    row_damg_2 = [elem for lst in row_damage_2.tolist() for elem in lst]
                    row_damg_3 = [elem for lst in row_damage_3.tolist() for elem in lst]
                    row_damg_4 = [elem for lst in row_damage_4.tolist() for elem in lst]
                    row_damg_5 = [elem for lst in row_damage_5.tolist() for elem in lst]
                    row_damg_6 = [elem for lst in row_damage_6.tolist() for elem in lst]
                    row_damg_9 = [elem for lst in row_damage_9.tolist() for elem in lst]
                    row_damg_10 = [elem for lst in row_damage_10.tolist() for elem in lst]

                    # Apply damage to layers
                    damaged_morph_1.append(row_damg_1)
                    damaged_morph_2.append(row_damg_2)
                    damaged_morph_3.append(row_damg_3)
                    damaged_morph_4.append(row_damg_4)
                    damaged_morph_5.append(row_damg_5)
                    damaged_morph_6.append(row_damg_6)
                    damaged_morph_9.append(row_damg_9)
                    damaged_morph_10.append(row_damg_10)
                    # Append unaffected layers
                    damaged_morph_7.append(row.tolist())
                    damaged_morph_8.append(row.tolist())
                    damaged_morph_11.append(row.tolist())
                    damaged_morph_12.append(row.tolist())

                else:
                    # convert to array
                    row_array = np.array(np.array_split(row, x_voxels))

                    # Apply damage to each row
                    row_damage_1 = np.multiply(row_array, damage_1)
                    row_damage_2 = np.multiply(row_array, damage_2)
                    row_damage_3 = np.multiply(row_array, damage_3)
                    row_damage_4 = np.multiply(row_array, damage_4)
                    row_damage_7 = np.multiply(np.multiply(row_array, damage_1), damage_3)
                    row_damage_8 = np.multiply(np.multiply(row_array, damage_2), damage_4)
                    row_damage_11 = np.multiply(np.multiply(row_array, damage_1), damage_2)
                    row_damage_12 = np.multiply(np.multiply(row_array, damage_3), damage_4)

                    # Convert to list
                    row_damg_1 = [elem for lst in row_damage_1.tolist() for elem in lst]
                    row_damg_2 = [elem for lst in row_damage_2.tolist() for elem in lst]
                    row_damg_3 = [elem for lst in row_damage_3.tolist() for elem in lst]
                    row_damg_4 = [elem for lst in row_damage_4.tolist() for elem in lst]
                    row_damg_7 = [elem for lst in row_damage_7.tolist() for elem in lst]
                    row_damg_8 = [elem for lst in row_damage_8.tolist() for elem in lst]
                    row_damg_11 = [elem for lst in row_damage_11.tolist() for elem in lst]
                    row_damg_12 = [elem for lst in row_damage_12.tolist() for elem in lst]

                    # Append unaffected layers
                    damaged_morph_5.append(row.tolist())
                    damaged_morph_6.append(row.tolist())
                    damaged_morph_9.append(row.tolist())
                    damaged_morph_10.append(row.tolist())
                    # Apply damage to layers
                    damaged_morph_1.append(row_damg_1)
                    damaged_morph_2.append(row_damg_2)
                    damaged_morph_3.append(row_damg_3)
                    damaged_morph_4.append(row_damg_4)
                    damaged_morph_7.append(row_damg_7)
                    damaged_morph_8.append(row_damg_8)
                    damaged_morph_11.append(row_damg_11)
                    damaged_morph_12.append(row_damg_12)

        creatures_quarters_damage = {"quarters_creature_1": damaged_morph_1,
                                     "quarters_creature_2": damaged_morph_2,
                                     "quarters_creature_3": damaged_morph_3,
                                     "quarters_creature_4": damaged_morph_4,
                                     "quarters_creature_5": damaged_morph_5,
                                     "quarters_creature_6": damaged_morph_6,
                                     "quarters_creature_7": damaged_morph_7,
                                     "quarters_creature_8": damaged_morph_8,
                                     "quarters_creature_9": damaged_morph_9,
                                     "quarters_creature_10": damaged_morph_10,
                                     "quarters_creature_11": damaged_morph_11,
                                     "quarters_creature_12": damaged_morph_12
                                     }

        # Update creature morphology and set stiffness to min_stiffness if morphology is at 0
        self.phenotype.morphology = np.array(random.choice(list(creatures_quarters_damage.values())))
        self.stiffness_array = np.where(self.phenotype.morphology == 0, self.settings["structure"]["min_stiffness"],
                                        self.stiffness_array)

