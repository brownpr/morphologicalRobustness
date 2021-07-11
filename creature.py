import csv
import json

import numpy as np

from evosoro import Evosoro
from voxel import Voxel
from neural_network import NeuralNet


class Creature:
    def __init__(self, index=None, name=None):
        # When called initializes creature file, setting default values to creature.
        # ARGUMENTS
        # - index                                                   int, index item for creature naming

        # import settings
        settings_file = open("settings.json")
        self.settings = json.load(settings_file)
        settings_file.close()

        # create creature phenotype
        self.phenotype = Evosoro()                                  # class, creature phenotype

        # Create dictionary of voxels                               # dict of classes, dictionary of creatures voxels
        self.voxels = {coords: Voxel(list(coords), mat_number) for coords, mat_number
                       in dict(np.ndenumerate(self.phenotype.morphology)).items()}

        # Basic creature information
        self.episode = None                                         # int, saves creatures current episode number
        self.generation = None                                      # int, saves creatures current generation number
        if name:
            self.name = name                                        # string, Set creature's name
        elif not index:
            raise Warning("SIMULATION STOPPED. When creating a creature, you must provide either name or index!")
        else:
            self.name = "_creature" + str(index)                    # string, Set creature's name

        # file name variables
        self.current_file_name = None                               # string, current VXA file name
        self.fitness_file_name = None                               # string, current fitness file name
        self.pressures_file_name = None                             # string, current pressures file name
        self.ke_file_name = None                                    # string, current ke file name
        self.strain_file_name = None                                # string, current strain file name

        # Fitness variables
        self.previous_fitness = 0.0                                 # float, previous fitness
        self.fitness_xyz = None                                     # (1,3) list, fitness
        self.fitness_eval = 0.0                                     # float, creatures evaluated fitness
        self.average_forces = None                                  # (z, x*y) list, average forces acting on voxels.
        #                                                             (where x, y, z are values from structure list)

        # genome (neural network of creature
        self.neural_net = None                                      # class, neural network of the creature

        # evolution
        self.evolution = {}                                         # dict, save creatures evolutionary history

        # Create stiffness array and update it
        self.stiffness_array = np.zeros(self.phenotype.morphology.shape)  # np.array of stiffness at each voxel
        for index, voxel in self.voxels.items():
            self.stiffness_array[index[0]][index[1]][index[2]] = voxel.stiffness

    def set_neural_network(self):
        self.neural_net = NeuralNet()

    def set_neighbours_and_sections(self):
        # Get region areas
        x_sections, y_sections, z_sections = self.settings["structure"]["sections"]
        range_x_section, range_y_sections, range_z_sections = np.divide(self.settings["structure"]["creature_structure"],
                                                                        self.settings["structure"]["sections"])
        section_counter = 0
        section_range_dictionary = {}
        for z in range(0, z_sections):
            z_range = [z * range_z_sections, (z + 1) * range_z_sections]
            for y in range(0, y_sections):
                y_range = [y * range_z_sections, (y + 1) * range_z_sections]
                for x in range(0, x_sections):
                    x_range = [x * range_z_sections, (x + 1) * range_z_sections]
                    section_range_dictionary[str(section_counter)] = [x_range, y_range, z_range]
                    section_counter += 1

        # Iterate thorough voxels and add their neighbours and the section they belong to
        for index, voxel in self.voxels.items():
            z, y, x = index
            # Get list of possible neighbours and find which voxels are within this range
            neighbours_list = [[z, y, x - 1], [z, y, x + 1], [z, y - 1, x], [z, y + 1, x], [z - 1, y, x], [z + 1, y, x]]
            for neighbour in self.voxels.values():
                if neighbour.coordinates in neighbours_list:
                    voxel.neighbours.append(neighbour)

            # Find which section the x, y, z coordinates for this voxel
            for section_num, section_range in section_range_dictionary.items():
                if (x >= section_range[0][0]) and (x < section_range[0][1]) and (y >= section_range[1][0]) and \
                        (y < section_range[1][1]) and (z >= section_range[2][0]) and (z < section_range[2][1]) and \
                        voxel.section is None:
                    voxel.section = int(section_num)

    def update_creature_info(self, generation, episode):
        # Updates basic creature information
        # ARGUMENTS
        # - generation                                              int, used to update creature generation number
        # - episode                                                 int, used to update creature episode number

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
        self.phenotype.update_vxa_file(self, stiffness_array=self.stiffness_array)

    def update_evolution(self):
        # Updates creatures evolutionary history ands saves key information within its self.evolution dictionary.
        # Allows historical values to be retried at any point after simulations have completed.

        if ("gen_" + str(self.generation)) not in self.evolution:
            self.evolution[("gen_" + str(self.generation))] = {}
            # Update neural network parameters for said generation
            self.evolution["gen_" + str(self.generation)].update(
                {"nn_parameters": {key: str(value) for key, value in self.neural_net.parameters.items()}})

        if ("ep_" + str(self.episode)) not in self.evolution[("gen_" + str(self.generation))]:
            self.evolution["gen_" + str(self.generation)][("ep_" + str(self.episode))] = {}

        # Update morphology for episode
        self.evolution[("gen_" + str(self.generation))][("ep_" + str(self.episode))]["morphology"] = \
            str(np.reshape(self.phenotype.morphology, (self.phenotype.structure[2], self.phenotype.structure[0]
                                                       * self.phenotype.structure[1])).tolist())

        # Update stiffness array for current episode
        if self.stiffness_array is not None:
            self.evolution["gen_" + str(self.generation)]["ep_" + str(self.episode)]["stiffness"] = \
                str(np.reshape(self.stiffness_array, (self.phenotype.structure[2], self.phenotype.structure[0]
                                                      * self.phenotype.structure[1])).tolist())

        # Update fitness values in evolution
        self.evolution["gen_" + str(self.generation)]["ep_" + str(self.episode)].update({"fitness_xyz": str(self.fitness_xyz)})
        self.evolution["gen_" + str(self.generation)]["ep_" + str(self.episode)].update({"fitness_eval": self.fitness_eval})
        self.evolution["gen_" + str(self.generation)]["ep_" + str(self.episode)].update({"average_forces": str(self.average_forces.tolist())})

    def update_morphology(self, new_stiffness_array=None):
        # Updates creatures stiffness and morphology dependant on stiffness values
        # Additionally, as voxels are removed voxels can become isolated from any adjacent voxels. This causes errors
        # in the simulator. Hence, these isolated voxels must be removed

        # Update creatures voxel stiffness and morphology and update creatures stiffness_array,
        # if none is given this part is skipped
        if new_stiffness_array is not None:
            for index, voxel in self.voxels.items():
                voxel.update_with_stiffness(new_stiffness_array[index[0]][index[1]][index[2]])
                self.stiffness_array[index[0]][index[1]][index[2]] = voxel.stiffness

        # remove isolated voxels
        for voxel in self.voxels.values():
            # only run if the voxel can change and is not already removed
            if voxel.can_be_changed and voxel.material_number:
                # grab list of the material numbers for neighbouring voxels
                neighbour_states = [neighbour.material_number for neighbour in voxel.neighbours]
                # if all neighbouring voxels are 0 delete current cube
                if not any(neighbour_states):
                    voxel.remove()

        # Update creatures morphology
        for index, voxel in self.voxels.items():
            # Although not needed (as no change will have occurred) this will save iterations
            if voxel.can_be_changed:
                self.phenotype.morphology[index[0]][index[1]][index[2]] = voxel.material_number

    def update_neural_network(self, return_self=False):
        # Update neural network
        self.neural_net.update_neural_net()

        # Return self?
        if return_self:
            return self

    def calculate_fitness(self):
        # Evaluates creatures fitness by reading the saved fitness file and saves fitness evaluation. Punishes creature
        # for displacement in y axis

        mod = 3  # int, modifier for severity of punishment in y displacement

        # Update old fitness with current fitness
        self.previous_fitness = self.fitness_eval

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
        self.fitness_eval = self.fitness_xyz[0] - self.fitness_xyz[1] * mod

    def calculate_stiffness(self):
        # Uses artificial neural network to update the creatures morphology and stiffness array.

        # Calculate displacement since last evaluation
        displacement_delta = (self.fitness_eval - self.previous_fitness)/10

        # Average forces
        average_forces = np.zeros((self.phenotype.structure[2], self.phenotype.structure[0], self.phenotype.structure[1]))

        # Format KE file
        with open(self.ke_file_name) as kef:
            ke_data = csv.reader(kef)
            for row in ke_data:
                row_data = np.multiply(np.array(row[:-1], dtype=np.float), 10)
                row_array = np.reshape(row_data, (self.phenotype.structure[2],
                                                  self.phenotype.structure[0], self.phenotype.structure[1]))
                average_forces += row_array
        kef.close()

        # set self.average_forces vector and calculate ultimate avg
        self.average_forces = np.divide(average_forces, np.prod(self.phenotype.structure))
        ultimate_average = np.sum(self.average_forces)/np.prod(self.phenotype.structure)

        # Calculate difference between ultimate avg and the average force on each voxel
        ke_delta = np.multiply(np.subtract(ultimate_average, self.average_forces), 10)

        # Update evolutionary history of creature before calculation and updating stiffness
        self.update_evolution()

        # Use NN to update the stiffness array of the creature
        vectorized_nn = np.vectorize(self.neural_net.forward_propagation)
        stiffness_delta, cache = vectorized_nn(ke_delta, displacement_delta)
        stiffness_delta = np.multiply(stiffness_delta, 10000)

        # Update stiffness of voxels
        new_stiffness_array = np.add(self.stiffness_array, stiffness_delta)

        # Fix morphology, restore actuator voxels stiffness and material number and remove isolated voxels
        self.update_morphology(new_stiffness_array)

    def find_voxel_by_coordinates(self, coordinates):
        for index, voxel in self.voxels.items():
            if coordinates == index:
                return voxel

        raise Exception("ERROR: Voxel not found, please ensure given coordinates are formatted correctly.")

    def find_voxels_in_radius(self, centre_voxel_coordinates, radius):
        if radius == 1:
            raise Warning("WARNING: remove_spherical_region radius input of 1 will only remove the centre voxel and "
                          "none of its neighbours.")
        elif radius < 1:
            raise Exception("ERROR: Cannot have a radius less than 1.")

        # Find centre voxel
        centre_voxel = self.find_voxel_by_coordinates(centre_voxel_coordinates)

        # Create list of voxels that are in the desired area
        voxels_in_radius = {centre_voxel}
        while radius > 1:
            affected_voxels = list(voxels_in_radius)
            for i in range(len(affected_voxels)):
                voxel = affected_voxels[i]
                voxels_in_radius.update(voxel.neighbours)
            radius = radius - 1

        return voxels_in_radius

    def remove_voxels_sections(self, sections):
        if isinstance(sections, tuple):
            sections = list(sections)
        elif isinstance(sections, int):
            sections = [sections]
        assert isinstance(sections, list)

        for voxel in self.voxels.values():
            if voxel.section in sections:
                if voxel.can_be_changed and voxel.material_number:
                    voxel.remove()

        # Update creature name
        self.name = self.name + "_removed_sections_" + "_".join(str(elem) for elem in sections)

        # update the morphology of the creature
        self.update_morphology()

    def stiffness_change_sections(self, sections, multiply_stiffness=None, divide_stiffness=None,
                                  set_new_stiffness=None, reduce_stiffness=None, increase_stiffness=None):
        # If no stiffness change provided stop simulation
        if not any((multiply_stiffness, divide_stiffness, set_new_stiffness, reduce_stiffness, increase_stiffness)):
            raise Exception("ERROR: You must provide some for of stiffness change for spherical_region_stiffness_change")

        if isinstance(sections, tuple):
            sections = list(sections)
        elif isinstance(sections, int):
            sections = [sections]
        assert isinstance(sections, list)

        # Multiplies stiffness of affected region by input
        if multiply_stiffness:
            for voxel in self.voxels.values():
                if voxel.section in sections:
                    if voxel.can_be_changed:
                        voxel.update_with_stiffness(voxel.stiffness*multiply_stiffness)

            # Update creature name
            self.name = self.name + "_stiffness_multiplied_sections_" + "_".join(str(elem) for elem in sections)

        # Divides stiffness of affected region by input
        if divide_stiffness:
            for voxel in self.voxels.values():
                if voxel.section in sections:
                    if voxel.can_be_changed:
                        voxel.update_with_stiffness(voxel.stiffness/divide_stiffness)

            # Update creature name
            self.name = self.name + "_stiffness_divided_sections_" + "_".join(str(elem) for elem in sections)

        # Changes stiffness to given input
        if set_new_stiffness:
            for voxel in self.voxels.values():
                if voxel.section in sections:
                    if voxel.can_be_changed:
                        voxel.update_with_stiffness(set_new_stiffness)

            # Update creature name
            self.name = self.name + "_stiffness_changed_sections_" + "_".join(str(elem) for elem in sections)

        # Add to stiffness of affected region by input
        if increase_stiffness:
            for voxel in self.voxels.values():
                if voxel.section in sections:
                    if voxel.can_be_changed:
                        voxel.update_with_stiffness(voxel.stiffness + increase_stiffness)

            # Update creature name
            self.name = self.name + "_stiffness_increased_sections_" + "_".join(str(elem) for elem in sections)

        # reduce stiffness of affected region by input
        if reduce_stiffness:
            for voxel in self.voxels.values():
                if voxel.section in sections:
                    if voxel.can_be_changed:
                        voxel.update_with_stiffness(voxel.stiffness - reduce_stiffness)

            # Update creature name
            self.name = self.name + "_stiffness_reduced_sections_" + "_".join(str(elem) for elem in sections)

        # update the morphology of the creature
        self.update_morphology()

    def remove_voxels_spherical_region(self, centre_voxel_coordinates, radius):

        # Get list of voxels in radius
        voxels_in_radius = self.find_voxels_in_radius(centre_voxel_coordinates, radius)

        # Set material number for voxels in radius to zero
        for voxel in voxels_in_radius:
            if voxel.can_be_changed and voxel.material_number:
                voxel.remove()

        # Update creature name
        self.name = self.name + "_sphere_removed_radius_" + str(radius)

        # update the morphology of the creature
        self.update_morphology()

    def stiffness_change_spherical_region(self, centre_voxel_coordinates, radius, multiply_stiffness=None,
                                          divide_stiffness=None, set_new_stiffness=None, reduce_stiffness=None,
                                          increase_stiffness=None):

        # If no stiffness change provided stop simulation
        if not any((multiply_stiffness, divide_stiffness, set_new_stiffness, reduce_stiffness, increase_stiffness)):
            raise Exception("ERROR: You must provide some for of stiffness change for spherical_region_stiffness_change")

        # Get list of voxels in radius
        voxels_in_radius = self.find_voxels_in_radius(centre_voxel_coordinates, radius)

        # Multiplies stiffness of affected region by input
        if multiply_stiffness:
            for voxel in voxels_in_radius:
                if voxel.can_be_changed:
                    voxel.update_with_stiffness(voxel.stiffness*multiply_stiffness)

            # Update creature name
            self.name = self.name + "_sphere_multiplied_radius_" + str(radius)

        # Divides stiffness of affected region by input
        if divide_stiffness:
            for voxel in voxels_in_radius:
                if voxel.can_be_changed:
                    voxel.update_with_stiffness(voxel.stiffness/divide_stiffness)

            # Update creature name
            self.name = self.name + "_sphere_divided_radius_" + str(radius)

        # Changes stiffness to given input
        if set_new_stiffness:
            for voxel in voxels_in_radius:
                if voxel.can_be_changed:
                    voxel.update_with_stiffness(set_new_stiffness)

            # Update creature name
            self.name = self.name + "_sphere_changed_radius_" + str(radius)

        # Add to stiffness of affected region by input
        if increase_stiffness:
            for voxel in voxels_in_radius:
                if voxel.can_be_changed:
                    voxel.update_with_stiffness(voxel.stiffness + increase_stiffness)

            # Update creature name
            self.name = self.name + "_sphere_increased_radius_" + str(radius)

        # reduce stiffness of affected region by input
        if reduce_stiffness:
            for voxel in voxels_in_radius:
                if voxel.can_be_changed:
                    voxel.update_with_stiffness(voxel.stiffness - reduce_stiffness)

            # Update creature name
            self.name = self.name + "_sphere_divided_radius_" + str(radius)

        # update the morphology of the creature
        self.update_morphology()
