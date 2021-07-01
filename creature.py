import csv
import json
import os
import subprocess as sub
import time
import datetime as dt
import shutil
import operator
import random
from copy import deepcopy

import numpy as np

from evosoro import Evosoro
from neural_network import NeuralNet

# The following file contains class parameters for creatures and their neural networks. To edit default values refer to
# settings.json.


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
        self.damaged = False                # bool, has creature been damaged?

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


class Population:
    # Allows for the creation of a population of creatures
    def __init__(self):
        # Import settings
        settings_file = open("settings.json")
        self.settings = json.load(settings_file)
        settings_file.close()

        # When called for first time, creates a new population of creatures.
        self.population = {}                                                        # Evaluation population
        self.full_population = {}                                                   # Population of all creatures
        self.create_new_population((0, self.settings["parameters"]["pop_size"]))    # Create starting population
        self.damaged_population = {}                                                # Damaged population

    def create_new_population(self, population_range):
        # ARGUMENTS:
        # - Range: 1x2 list, (a, b) where a is lower & b upper bound of range of creatures (for naming index)

        # RETURNS:
        # population: dictionary of created creature.

        # Set range parameters
        a, b = population_range

        # Create population
        for i in range(a, b):
            # set genome to a random float between -2.0 and 2.0
            genome = np.round(np.random.uniform(-2, 2, 1), decimals=1)[0]
            # Create creature
            creature = Creature(genome, i)

            # Append creature to creature dictionary
            self.population["creature_" + str(i)] = creature
            # Add created creatures to full population
            self.full_population["creature_" + str(i)] = creature

    def run_genetic_algorithm(self):
        # Retrieve parameters
        gen_size = self.settings["parameters"]["gen_size"]

        # Initialize genetic algorithm
        for gen_num in range(gen_size):
            # Provide user with generation number
            print(str(dt.datetime.now()) + " CURRENT GENERATION NUMBER: " + str(gen_num))

            # Evaluate population
            self.eval_pop(gen_num)

            if not gen_num == gen_size - 1:
                # Create new population and retrieve top performing creature
                top_creature = self.new_population()
            else:
                sorted_pop, top_creature = self.sort_population()

            # Print gen top performers details
            print(str(dt.datetime.now()) + " Top performer:" + top_creature.name + ". Fitness: " + str(top_creature.fitness_eval))

        print("FINISHED SIMULATIONS FOR UNDAMAGED CREATURES.")

    def eval_pop(self, generation_number):
        # ARGUMENTS
        # - gen_num:        int, Current generation
        # - population:     dict, creatures to evaluate
        # - episode_size:   int, Number of episodes used to evaluate creatures

        # Working directories variables
        cwd = os.getcwd()
        gfd = os.path.join(os.getcwd(), "generated_files")  # Generated files directory

        # Start simulations, after each simulation robot undergoes morphological change
        for episode in range(self.settings["parameters"]["ep_size"]):
            for creature in self.population.values():

                # Create VXA file for creature
                creature.update_vxa(generation=generation_number, episode=episode)

                # Get file path variables and save vxa
                vxa_fp = os.path.join(cwd, creature.current_file_name + ".vxa")
                new_file = open(vxa_fp, "w")
                new_file.write(creature.phenotype.vxa_file)
                new_file.close()

                # launch simulation
                sub.Popen(self.settings["evosoro_path"] + " -f  " + creature.current_file_name + ".vxa", shell=True)

                # wait for fitness and pressure file existence
                ffp = os.path.join(cwd, creature.fitness_file_name)     # ffp
                pf = os.path.join(cwd, creature.pressures_file_name)    # pressure file path
                kefp = os.path.join(cwd, creature.ke_file_name)         # ke file path
                sfp = os.path.join(cwd, creature.strain_file_name)      # strain file path

                while not os.path.exists(pf) or not os.path.exists(ffp):
                    time.sleep(1)

                # Update creature fitness
                creature.update_fitness()

                # occasionally an error occurs and results return 0, if so, re-run for up to 60s
                t = time.time()
                toc = 0
                while creature.update_fitness == 0 and toc < 60:
                    creature.update_fitness()
                    toc = time.time() - t

                # Update creature stiffness, uses ANN
                creature.update_stiffness()

                # Create new folders and move files
                ccf = os.path.join(gfd, creature.name)  # current creature folder
                if not os.path.exists(ccf):
                    os.mkdir(ccf)

                cgf = os.path.join(ccf, "gen_" + str(generation_number))  # current generation folder
                if not os.path.exists(cgf):
                    os.mkdir(cgf)

                cef = os.path.join(cgf, "ep_" + str(episode))   # current episode folder
                if not os.path.exists(cef):
                    os.mkdir(cef)

                # Move created creature files to corresponding episode folder
                shutil.move(vxa_fp, cef)
                shutil.move(ffp, cef)
                shutil.move(pf, cef)
                shutil.move(kefp, cef)
                shutil.move(sfp, cef)

    def new_population(self):
        # Function sorts previously evaluated population and selects top performers, evolves the neural network of a
        # a selected few and creates new creatures. These are joined into one dictionary for further evaluation
        #
        # RETURNS
        # - top_creature        class (creature), top performing creature of the last evaluation

        # Retrieve parameters
        top = self.settings["parameters"]["top"]
        evolve = self.settings["parameters"]["evolve"]

        # Sort population
        sorted_pop, top_creature = self.sort_population()

        # Create new population with completely random genomes
        num_creatures = len(self.full_population)                                   # Number of creatures created so far
        new_pop_size = len(self.population) - top - evolve                          # Number of new creatures to make
        self.population = {}                                                        # Reset population
        self.create_new_population((num_creatures, num_creatures + new_pop_size))   # Create new creatures

        # Add top preforming creatures to new population
        self.population.update({crt.name: crt for crt in sorted_pop[0:top]})

        # From sorted pop grab the next =evolved (num) creatures
        self.population.update({crt.name: crt.evolve() for crt in sorted_pop[top:evolve+top]})

        # Print list
        print(str(dt.datetime.now()) + " CURRENT POPULATION: ")
        print([str(ctr.name) + ":" + str(ctr.genome) for ctr in self.population.values()])

        # Top creature
        top_creature = sorted_pop[0]

        return top_creature

    def sort_population(self, population=None):
        # If pop not specified used self.population
        if population is None:
            population = self.population

        # Sort creatures by fitness size
        sorted_population = sorted(population.values(), key=operator.attrgetter("fitness_eval"), reverse=True)

        # Top creature
        top_creature = sorted_population[0]

        return sorted_population, top_creature

    def save_population(self, population=None):
        # Save data of the top preforming creatures throughout generations.
        # If pop not specified used self.population
        if population is None:
            population = self.full_population

        # Sorted by creature performance
        sorted_pop, top_creature = self.sort_population(population)

        # Save files
        dict_sort_crts = [{ctr.name: ctr.fitness_eval} for ctr in sorted_pop]
        with open("generated_files/creature_evolution.json", "w") as ctr_file:
            json.dump(dict_sort_crts, ctr_file, sort_keys=True, indent=4)
        ctr_file.close()

    def damage_population(self, num_creatures_to_damage=None):
        # Inflicts damage to each member of the population, if number_of_creatures_to_damage is given,
        # the full_population is sorted and inputed to select the best creatures.
        # If pop not specified used self.full_population
        if num_creatures_to_damage is None:
            damage_pop = self.full_population
        else:
            damage_pop_list, top_creature = self.sort_population(self.full_population)
            damage_pop = {crt.name: crt for crt in damage_pop_list[0:num_creatures_to_damage]}

        for crt_name, crt in damage_pop.items():
            # Copy class
            eighths_ctr = deepcopy(crt)
            quarter_ctr = deepcopy(crt)
            half_ctr = deepcopy(crt)

            # rename copied creature
            eighths_ctr.name = crt_name + "_eighths_damage"
            quarter_ctr.name = crt_name + "_quarter_damage"
            half_ctr.name = crt_name + "_half_damage"

            # Inflict damage to creature (creates new morphology and stiffness array)
            eighths_ctr.remove_eighths()
            quarter_ctr.remove_halfs()
            half_ctr.remove_halfs()

            # Save into self.damage_population dict
            self.damaged_population[eighths_ctr.name] = eighths_ctr
            self.damaged_population[quarter_ctr.name] = quarter_ctr
            self.damaged_population[half_ctr.name] = half_ctr



