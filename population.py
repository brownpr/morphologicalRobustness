import json
import os
import datetime as dt
import operator
import subprocess as sub
import time
import shutil
from copy import deepcopy

from creature import Creature


class Population:
    def __init__(self, population=None, damaged_population=False, reset_evolution=False):
        # Import settings
        settings_file = open("settings.json")
        self.settings = json.load(settings_file)
        settings_file.close()

        # When called for first time, creates a new population of creatures.
        self.population = {}                            # dict, Evaluation population
        self.full_population = {}                       # dict, Full population
        self.damaged_population = damaged_population    # bool, has population been damaged?
        self.damage_type = None                         # string, what type of damage has been inflicted on creature
        self.damage_arguments = None                    # Previously used arguments (all creatures created in a damaged
        #                                                   population will receive the same damage as the originals)

        # Create an example creature to copy from (as many parameters don't change when creating an additional creature)
        self.base_creature = Creature(name="base_creature")
        self.base_creature.set_neighbours_and_sections()

        # If population introduced into system add to self.population, else create new starting population
        if population is not None:
            self.population = population
        else:
            self.create_new_population((0, self.settings["parameters"]["pop_size"]))

        # If reset_evolution, reset evolutionary history
        if reset_evolution:
            for creature in self.population.values():
                creature.evolution = {}

        # Set full population
        self.full_population = self.population          # Full population is initial population

    def create_new_population(self, population_range):
        # ARGUMENTS:
        # - Range: 1x2 list, (a, b) where a is lower & b upper bound of range of creatures (for naming index)

        # RETURNS:
        # population: dictionary of created creature.

        # Set range parameters
        a, b = population_range

        # Create population
        for i in range(a, b):
            # Create creature
            creature = deepcopy(self.base_creature)
            creature.name = "_creature" + str(i)

            # Get ANN for creature
            creature.set_neural_network()

            # Append creature to creature dictionary
            self.population[creature.name] = creature
            # Add created creatures to full population
            self.full_population[creature.name] = creature

        # If the creature to be added to a damaged population class, damage said creatures before adding them
        if self.damaged_population:
            self.population = self.inflict_damage(self.damage_type, self.damage_arguments)

    def run_genetic_algorithm(self, generation_size=None):
        # Runs genetic algorithm by evaluating each creature and then changing their morphology accordingly

        # Retrieve parameters
        if generation_size is None:
            generation_size = self.settings["parameters"]["gen_size"]

        # Print Creature Genomes
        if self.damaged_population:
            print(str(dt.datetime.now()) + " INITIAL DAMAGED POPULATION: ")
        else:
            print(str(dt.datetime.now()) + " INITIAL UNDAMAGED POPULATION: ")

        print([str(creature.name) for creature in self.population.values()])

        # Initialize genetic algorithm
        for generation_num in range(generation_size):
            # Provide user with generation number
            print(str(dt.datetime.now()) + " CURRENT GENERATION NUMBER: " + str(generation_num))
            # Print list of creatures under evaluation
            print(str(dt.datetime.now()) + " Population under evaluation:")
            print([creature.name for creature in self.population.values()])

            # Evaluate population
            self.evaluate_population(generation_num)

            if not generation_num == generation_size - 1:
                # Create new population and retrieve top performing creature
                top_creature = self.new_population()
            else:
                sorted_pop, top_creature = self.sort_population()

            # Print generation top performers details
            print(str(dt.datetime.now()) + " Finished evaluating population, top performing creature:"
                  + top_creature.name + ". Fitness: " + str(top_creature.fitness_eval))

        if self.damaged_population:
            print(str(dt.datetime.now()) + " FINISHED SIMULATIONS FOR " + self.damage_type.upper() +
                  " DAMAGED CREATURES.")
        else:
            print(str(dt.datetime.now()) + " FINISHED SIMULATIONS FOR UNDAMAGED CREATURES.")

    def evaluate_population(self, generation_number):
        # ARGUMENTS
        # - generation_num:        int, Current generation
        # - population:     dict, creatures to evaluate
        # - episode_size:   int, Number of episodes used to evaluate creatures

        # Working directories variables
        cwd = os.getcwd()
        gfd = os.path.join(os.getcwd(), "generated_files")  # Generated files directory

        # Start simulations, after each simulation robot undergoes morphological change
        for episode in range(self.settings["parameters"]["ep_size"]):
            for creature in self.population.values():

                # Create VXA file for creature
                creature.update_vxa(generation_number, episode)

                # Get file path variables and save vxa
                vxa_fp = os.path.join(cwd, creature.current_file_name + ".vxa")
                new_file = open(vxa_fp, "w")
                new_file.write(creature.phenotype.vxa_file)
                new_file.close()

                # launch simulation
                sub.Popen(self.settings["evosoro_path"] + " -f  " + creature.current_file_name + ".vxa", shell=True)

                # wait for fitness and pressure file existence
                ffp = os.path.join(cwd, creature.fitness_file_name)  # fitness file path
                pf = os.path.join(cwd, creature.pressures_file_name)  # pressure file path
                kefp = os.path.join(cwd, creature.ke_file_name)  # ke file path
                sfp = os.path.join(cwd, creature.strain_file_name)  # strain file path

                # wait for file to appear, if two minutes passes and there is no file raise exception
                t = time.time()
                while not os.path.exists(pf) or not os.path.exists(ffp):
                    time.sleep(1)
                    toc = time.time() - t
                    if toc > 120:
                        raise Exception("ERROR: No pressure file or fitness file found after 120 seconds. "
                                        "This error is commonly due to errors in the written vxa file.")

                # Update creature fitness
                creature.calculate_fitness()

                # occasionally an error occurs and results return 0, if so, re-run for up to 60s
                t = time.time()
                toc = 0
                while creature.fitness_eval == 0 and toc < 60:
                    creature.calculate_fitness()
                    toc = time.time() - t

                # Update creature stiffness, uses ANN
                creature.calculate_stiffness()

                # Create new folders and move files
                ccf = os.path.join(gfd, creature.name)  # current creature folder
                if not os.path.exists(ccf):
                    os.mkdir(ccf)

                cgf = os.path.join(ccf, "gen_" + str(generation_number))  # current generation folder
                if not os.path.exists(cgf):
                    os.mkdir(cgf)

                cef = os.path.join(cgf, "ep_" + str(episode))  # current episode folder
                if not os.path.exists(cef):
                    os.mkdir(cef)

                # Move created creature files to corresponding episode folder
                shutil.move(vxa_fp, cef)
                shutil.move(ffp, cef)
                shutil.move(pf, cef)
                shutil.move(kefp, cef)
                shutil.move(sfp, cef)

                # If at last episode, reset morphology and stiffness
                if episode == self.settings["parameters"]["ep_size"] - 1:
                    creature.reset_morphology()

    def new_population(self):
        # Function sorts previously evaluated population and selects top performers, evolves the neural network of a
        # a selected few and creates new creatures. These are joined into one dictionary for further evaluation
        #
        # RETURNS
        # - top_creature        class (creature), top performing creature of the last evaluation

        # Retrieve parameters
        top = self.settings["parameters"]["top"]
        evolve = self.settings["parameters"]["evolve"]
        population_size = self.settings["parameters"]["pop_size"]

        if (top + evolve) > population_size:
            raise Exception("ERROR in settings.json: please ensure that the sum of 'top' and 'evolve' is less than or "
                            "equal to the population size.")
        elif (top + evolve) == population_size:
            raise Warning("WARNING: in settings.json the sum of 'top' and 'evolve' equals population size, no new"
                          "creatures will be generated.")

        # Sort population
        sorted_pop, top_creature = self.sort_population()

        # Create new population with completely random genomes
        num_creatures = len(self.full_population)                   # Number of creatures created so far
        new_pop_size = len(self.population) - top - evolve          # Number of new creatures to make

        # Reset population
        self.population = {}
        # Create new creatures (must be ran before adding top preforming creatures
        self.create_new_population((num_creatures, num_creatures + new_pop_size))

        # Add top preforming creatures to new population
        self.population.update({creature.name: creature for creature in sorted_pop[0:top]})

        # From sorted pop grab the next =evolved (num) creatures
        self.population.update({creature.name: creature.update_neural_network(return_self=True)
                                for creature in sorted_pop[top:evolve + top]})

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
        # Save data of the top preforming creatures throughout generations. Run at the end of evaluation
        # If pop not specified used self.population
        if population is None:
            population = self.full_population

        # Sorted by creature performance
        sorted_pop, top_creature = self.sort_population(population)

        # Set file name dependant on damage
        if self.damaged_population:
            file_name = "damaged_evolution_" + self.damage_type
        else:
            file_name = "undamaged_evolution"

        # Save fitness rank of top preforming creatures for said population

        dict_sort_creatures = [{creature.name: creature.fitness_eval} for creature in sorted_pop]

        with open("generated_files/" + "performance_" + file_name + ".json", "w") as population_file:
            json.dump(dict_sort_creatures, population_file, sort_keys=True, indent=4)
        population_file.close()

        # Save evolutionary history of creatures
        for creature in self.population.values():
            with open("generated_files/" + creature.name + "/evolution.json", "w") as creature_file:
                json.dump(creature.evolution, creature_file, sort_keys=True, indent=4)
            creature_file.close()

    def inflict_damage(self, damage_type, damage_arguments, population_to_damage=None, damage_base_creature=False):

        if population_to_damage is None:
            population_to_damage = self.population

        assert isinstance(damage_type, str)
        assert isinstance(damage_arguments, list)
        assert isinstance(damage_arguments, list)

        # SECTION DAMAGES
        if damage_type == "remove_sect":
            assert len(damage_arguments) == 1
            assert isinstance(damage_arguments[0], tuple)
            for creature in population_to_damage.values():
                creature.remove_voxels_sections(damage_arguments[0])

        elif damage_type == "stiff_sect_mult":
            assert len(damage_arguments) == 2
            assert isinstance(damage_arguments[0], tuple)
            assert isinstance(damage_arguments[1], float) or isinstance(damage_arguments[1], int)
            for creature in population_to_damage.values():
                creature.stiffness_change_sections(damage_arguments[0], multiply_stiffness=damage_arguments[1])

        elif damage_type == "stiff_sect_div":
            assert len(damage_arguments) == 2
            assert isinstance(damage_arguments[0], tuple)
            assert isinstance(damage_arguments[1], float) or isinstance(damage_arguments[1], int)
            for creature in population_to_damage.values():
                creature.stiffness_change_sections(damage_arguments[0], divide_stiffness=damage_arguments[1])

        elif damage_type == "stiff_sect_set":
            assert len(damage_arguments) == 2
            assert isinstance(damage_arguments[0], tuple)
            assert isinstance(damage_arguments[1], float) or isinstance(damage_arguments[1], int)
            for creature in population_to_damage.values():
                creature.stiffness_change_sections(damage_arguments[0], set_new_stiffness=damage_arguments[1])

        elif damage_type == "stiff_sect_add":
            assert len(damage_arguments) == 2
            assert isinstance(damage_arguments[0], tuple)
            assert isinstance(damage_arguments[1], float) or isinstance(damage_arguments[1], int)
            for creature in population_to_damage.values():
                creature.stiffness_change_sections(damage_arguments[0], increase_stiffness=damage_arguments[1])

        elif damage_type == "stiff_sect_red":
            assert len(damage_arguments) == 2
            assert isinstance(damage_arguments[0], tuple)
            assert isinstance(damage_arguments[1], float) or isinstance(damage_arguments[1], int)
            for creature in population_to_damage.values():
                creature.stiffness_change_sections(damage_arguments[0], reduce_stiffness=damage_arguments[1])

        # SPHERICAL DAMAGES
        elif damage_type == "remove_spher":
            assert len(damage_arguments) == 2
            assert len(damage_arguments[0]) == 3
            assert isinstance(damage_arguments[0], tuple)
            assert isinstance(damage_arguments[1], int)

            for creature in population_to_damage.values():
                creature.remove_voxels_spherical_region(damage_arguments[0], damage_arguments[1])

        elif damage_type == "stiff_spher_mult":
            assert len(damage_arguments) == 3
            assert len(damage_arguments[0]) == 3
            assert isinstance(damage_arguments[0], tuple)
            assert isinstance(damage_arguments[1], int)
            assert isinstance(damage_arguments[2], float) or isinstance(damage_arguments[2], int)

            for creature in population_to_damage.values():
                creature.stiffness_change_spherical_region(damage_arguments[0], damage_arguments[1],
                                                           multiply_stiffness=damage_arguments[2])

        elif damage_type == "stiff_spher_div":
            assert len(damage_arguments) == 3
            assert len(damage_arguments[0]) == 3
            assert isinstance(damage_arguments[0], tuple)
            assert isinstance(damage_arguments[1], int)
            assert isinstance(damage_arguments[2], float) or isinstance(damage_arguments[2], int)

            for creature in population_to_damage.values():
                creature.stiffness_change_spherical_region(damage_arguments[0], damage_arguments[1],
                                                           divide_stiffness=damage_arguments[2])

        elif damage_type == "stiff_spher_set":
            assert len(damage_arguments) == 3
            assert len(damage_arguments[0]) == 3
            assert isinstance(damage_arguments[0], tuple)
            assert isinstance(damage_arguments[1], int)
            assert isinstance(damage_arguments[2], float) or isinstance(damage_arguments[2], int)

            for creature in population_to_damage.values():
                creature.stiffness_change_spherical_region(damage_arguments[0], damage_arguments[1],
                                                           set_new_stiffness=damage_arguments[2])

        elif damage_type == "stiff_spher_add":
            assert len(damage_arguments) == 3
            assert len(damage_arguments[0]) == 3
            assert isinstance(damage_arguments[0], tuple)
            assert isinstance(damage_arguments[1], int)
            assert isinstance(damage_arguments[2], float) or isinstance(damage_arguments[2], int)

            for creature in population_to_damage.values():
                creature.stiffness_change_spherical_region(damage_arguments[0], damage_arguments[1],
                                                           increase_stiffness=damage_arguments[2])

        elif damage_type == "stiff_spher_red":
            assert len(damage_arguments) == 3
            assert len(damage_arguments[0]) == 3
            assert isinstance(damage_arguments[0], tuple)
            assert isinstance(damage_arguments[1], int)
            assert isinstance(damage_arguments[2], float) or isinstance(damage_arguments[2], int)

            for creature in population_to_damage.values():
                creature.stiffness_change_spherical_region(damage_arguments[0], damage_arguments[1],
                                                           reduce_stiffness=damage_arguments[2])

        else:
            raise Exception("ERROR: Unknown damage type.")

        # Update damage type and damage arguments
        self.damage_type = damage_type
        self.damage_arguments = damage_arguments

        # Inflict damage on the base creature.
        if damage_base_creature:
            self.inflict_damage(damage_type, damage_arguments, {self.base_creature.name: self.base_creature})
