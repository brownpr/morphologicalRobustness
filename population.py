import json
import os
import datetime as dt
import operator
import subprocess as sub
import time
import shutil
from copy import deepcopy

import numpy as np

from creature import Creature


class Population:
    # Allows for the creation of a population of creatures
    def __init__(self, population=None):
        # Import settings
        settings_file = open("settings.json")
        self.settings = json.load(settings_file)
        settings_file.close()

        # When called for first time, creates a new population of creatures.
        self.population = {}                # Evaluation population
        self.full_population = {}           # Population of all creatures
        self.damaged_population = False     # is population been damaged?

        # If population introduced into system add to self.population, else create new starting population
        if population is not None:
            self.population = population
        else:
            self.create_new_population((0, self.settings["parameters"]["pop_size"]))

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

        # If the creature to be added to a damaged population class, damage said creatures before adding them
        if self.damaged_population:
            self.population = self.inflict_damage()

    def run_genetic_algorithm(self):
        # Retrieve parameters
        gen_size = self.settings["parameters"]["gen_size"]

        # Print Creature Genomes
        if self.damaged_population:
            print(str(dt.datetime.now()) + " INITIAL DAMAGED POPULATION: ")
        else:
            print(str(dt.datetime.now()) + " INITIAL UNDAMAGED POPULATION: ")

        print([str(ctr.name) + ":" + str(ctr.genome) for ctr in self.population.values()])

        # Initialize genetic algorithm
        for gen_num in range(gen_size):
            # Provide user with generation number
            print(str(dt.datetime.now()) + " CURRENT GENERATION NUMBER: " + str(gen_num))

            # Evaluate population
            self.evaluate_population(gen_num)

            if not gen_num == gen_size - 1:
                # Create new population and retrieve top performing creature
                top_creature = self.new_population()
            else:
                sorted_pop, top_creature = self.sort_population()

            # Print gen top performers details
            print(str(dt.datetime.now()) + " Top performer:" + top_creature.name + ". Fitness: " + str(
                top_creature.fitness_eval))

        if self.damaged_population:
            print("FINISHED SIMULATIONS FOR DAMAGED CREATURES.")
        else:
            print("FINISHED SIMULATIONS FOR UNDAMAGED CREATURES.")

    def evaluate_population(self, generation_number):
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
                ffp = os.path.join(cwd, creature.fitness_file_name)  # ffp
                pf = os.path.join(cwd, creature.pressures_file_name)  # pressure file path
                kefp = os.path.join(cwd, creature.ke_file_name)  # ke file path
                sfp = os.path.join(cwd, creature.strain_file_name)  # strain file path

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

                cef = os.path.join(cgf, "ep_" + str(episode))  # current episode folder
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
        num_creatures = len(self.full_population)  # Number of creatures created so far
        new_pop_size = len(self.population) - top - evolve  # Number of new creatures to make

        # Reset population
        self.population = {}
        # Create new creatures (must be ran before adding top preforming creatures
        self.create_new_population((num_creatures, num_creatures + new_pop_size))

        # Add top preforming creatures to new population
        self.population.update({crt.name: crt for crt in sorted_pop[0:top]})

        # From sorted pop grab the next =evolved (num) creatures
        self.population.update({crt.name: crt.evolve() for crt in sorted_pop[top:evolve + top]})

        # Print list
        print(str(dt.datetime.now()) + " CURRENT POPULATION: ")
        print([str(ctr.name) + ":" + str(ctr.genome) for ctr in self.population.values()])

        # Top creaturetop_creature
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
        if self.damaged_population:
            file_name = "damaged_creature_evolution"
        else:
            file_name = "creature_evolution"

        with open("generated_files/" + file_name + ".json", "w") as ctr_file:
            json.dump(dict_sort_crts, ctr_file, sort_keys=True, indent=4)
        ctr_file.close()

    def inflict_damage(self, num_creatures_to_damage=None):
        # Inflicts damage to each member of the population, if number_of_creatures_to_damage is given,
        # the full_population is sorted and inputed to select the best creatures.
        # If pop not specified used self.full_population
        if num_creatures_to_damage is None:
            damage_pop = self.full_population
        else:
            damage_pop_list, top_creature = self.sort_population(self.full_population)
            damage_pop = {crt.name: crt for crt in damage_pop_list[0:num_creatures_to_damage]}

        damaged_pop = {}         # Dictionary to store damaged population
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

            # Save into self.inflict_damage dict
            damaged_pop[eighths_ctr.name] = eighths_ctr
            damaged_pop[quarter_ctr.name] = quarter_ctr
            damaged_pop[half_ctr.name] = half_ctr

        # With damaged population, create new class of damage population
        damaged_population = Population(damaged_pop)

        # set self.damaged_population to true
        damaged_population.damaged_population = True

        return damaged_population

