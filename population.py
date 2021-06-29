import os
import shutil
import time
import subprocess as sub
import numpy as np
import operator
import json
import datetime as dt

from creature import CreatureFile as CF


def create_new_creatures(rng):
    # ARGUMENTS:
    # - Range: 1x2 list, (a, b) where a is lower & b upper bound of range of creatures (for naming index)

    # RETURNS:
    # population: dictionary of created creatures

    # Import creature_structure parameters
    # Load parameters
    settings_file = open("settings.json")
    struc_params = json.load(settings_file)["creature_structure"]
    settings_file.close()

    # Set range parameters
    a, b = rng

    # Set morphological parameters
    population = {}  # Create empty dictionary for population
    starting_morphology = np.array([
                      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                  ])
    creature_strc = struc_params["creature_structure"]
    base_stiff = struc_params["base_stiffness"]  # Set base stiffness multiplier
    # Create initial stiffness array
    init_stiff = np.multiply(np.ones((creature_strc[2], creature_strc[0] * creature_strc[1])), base_stiff)

    # Create staring population
    for i in range(a, b):
        # set genome to a random float between -2.0 and 2.0
        genome = np.round(np.random.uniform(-2, 2, 1), decimals=1)[0]
        # Create creature
        creature = CF(genome, starting_morphology, creature_strc, i)
        # Update stillness to initial values
        creature.stiffness_array = init_stiff  # Initial Stiffness

        # Append creature to creature dictionary
        population["creature_" + str(i)] = creature

    return population


def eval_pop(gen_num, population, episode_size):
    # ARGUMENTS
    # - gen_num:        int, Current generation
    # - population:     dict, creatures to evaluate
    # - episode_size:   int, Number of episodes used to evaluate creatures

    # Working directories variables
    cwd = os.getcwd()
    gfd = os.path.join(os.getcwd(), "generated_files")  # Generated files directory

    init_distances = np.zeros(len(population))  # Initialize distance traveled to 0

    # Start simulations, after each simulation robot undergoes morphological change
    for episode in range(episode_size):
        for c_name, creature in population.items():

            # Create VXA file for creature
            creature.vxa_file = creature.createVXA(generation=gen_num, episode=episode)

            # Get file path variables and save vxa
            vxa_fp = os.path.join(cwd, creature.current_file_name + ".vxa")
            new_file = open(vxa_fp, "w")
            new_file.write(creature.vxa_file)
            new_file.close()

            # Load voxelize path location
            settings_file = open("settings.json")
            vox_path = json.load(settings_file)["evosoro_path"]
            settings_file.close()

            # launch simulation
            sub.Popen(vox_path + " -f  " + creature.current_file_name + ".vxa", shell=True)

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

            # Create new folders and move files there
            cgf = os.path.join(gfd, "gen_" + str(gen_num))  # current generation folder
            if not os.path.exists(cgf):
                os.mkdir(cgf)

            ccf = os.path.join(cgf, creature.name)  # current creature folder
            if not os.path.exists(ccf):
                os.mkdir(ccf)

            cef = os.path.join(ccf, creature.current_file_name)   # current episode folder
            if not os.path.exists(cef):
                os.mkdir(cef)

            shutil.move(vxa_fp, cef)
            shutil.move(ffp, cef)
            shutil.move(pf, cef)
            shutil.move(kefp, cef)
            shutil.move(sfp, cef)


def new_population(population, comb_pop, top, evolve):
    # Function sorts previously evaluated population and selects top performers, evolves the neural network of a
    # a selected few and creates new creatures. These are joined into one dictionary for further evaluation
    #
    # ARGUMENTS
    # - population      dict, creatures which have been evaluated
    # - comb_pop        dict, all creatures that have been created
    # - top             int, number of top preforming creatures to keep
    # - evolve          int, number of creatures to evolve their neural network
    # RETURNS
    # - new_pop         dict, new set of creatures to evaluate in next generation
    # - comb_pop        dict, updated dict of all creatures

    # Sort creatures by fitness size
    sorted_pop = sorted(population.values(), key=operator.attrgetter("fitness_eval"), reverse=True)

    # Add top preforming creatures to new population
    top_performers = {crt.name: crt for crt in sorted_pop[0:top]}

    # Top creature
    top_creature = sorted_pop[0]

    # From sorted pop grab the next =evolved (num) creatures
    evolved_pop = {crt.name: crt.evolve() for crt in sorted_pop[top:evolve+top]}

    # Create new population with completely random genomes
    num_creatures = len(comb_pop)  # Number of creatures created so far
    new_pop_size = len(population) - top - evolve  # Number of new creatures to make
    new_creatures = create_new_creatures((num_creatures, num_creatures + new_pop_size))  # Create new creatures
    print(str(dt.datetime.now()) + " NEW CREATURES: ")
    print([str(ctr.name) + ":" + str(ctr.genome) for ctr in new_creatures.values()])

    # join all dictionaries together
    new_pop = top_performers.copy()
    new_pop.update(evolved_pop)
    new_pop.update(new_creatures)

    # Update combined population dict with full list of creatures
    comb_pop.update(new_creatures)

    return new_pop, comb_pop, top_creature


if __name__ == "__main__":
    pass
    # for i in range(10):
    #     eval_pop([1, 2], i)
    # print("hello world")
