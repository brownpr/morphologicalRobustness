import datetime as dt
import json
import os
import sys
import operator

import population as pop


if __name__ == "__main__":
    # Following code launches the evolutionary procedure of a set of creatures, please refer to settings.json to
    # edit launch parameters and default values

    try:
        # Load parameters
        settings_file = open("settings.json")
        parameters = json.load(settings_file)["parameters"]
        settings_file.close()

        population = pop.create_new_creatures((0, parameters["pop_size"]))
        # Add population to combined population
        comb_pop = population.copy()  # combined population

        print(str(dt.datetime.now()) + " INITIAL POPULATION: ")
        print([str(ctr.name) + ":" + str(ctr.genome) for ctr in population.values()])

        # Create file to save creature files
        if not os.path.exists("generated_files"):
            os.mkdir("generated_files")
        else:
            print("STOPPING SIMULATION ... creature files may already exist. Please delete or rename 'generated_files' folder.")
            sys.exit()

        # Initialize genetic algorithm
        for gen_num in range(parameters["gen_size"]):
            # Provide user with generation number
            print(str(dt.datetime.now()) + " CURRENT GENERATION NUMBER: " + str(gen_num))

            # Evaluate population
            pop.eval_pop(gen_num, population, parameters["ep_size"])

            # Create new population, returns comb_pop and new population (both lists)
            population, comb_pop, top_creature = pop.new_population(population, comb_pop, parameters["top"], parameters["evolve"])

            print(str(dt.datetime.now()) + " Top performer:" + top_creature.name + ". Fitness: " + str(top_creature.fitness_eval))
        print("FINISHED EVOLVING")

        # Save data of the top preforming creatures throughout generations.
        # Sorted by creature performance
        comb_sorted_pop = sorted(comb_pop.values(), key=operator.attrgetter("fitness_eval"), reverse=True)
        with open("generated_files/creature_evolution.json", "w") as ctr_file:
            for creature in comb_sorted_pop:
                json.dump(creature.evolution, ctr_file, sort_keys=False, indent=4)
        ctr_file.close()

    except KeyboardInterrupt:  # Allow for keybord interupt of script
        exit()
