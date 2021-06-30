import datetime as dt
import os
import sys


from creature import Population


if __name__ == "__main__":
    # Following code launches the evolutionary procedure of a set of creatures, please refer to settings.json to
    # edit launch parameters and default values

    try:
        # Create population
        pop = Population()

        print(str(dt.datetime.now()) + " INITIAL POPULATION: ")
        print([str(ctr.name) + ":" + str(ctr.genome) for ctr in pop.population.values()])

        # Create file to save creature files
        if not os.path.exists("generated_files"):
            os.mkdir("generated_files")
        else:
            print("STOPPING SIMULATION ... creature files may already exist. Please delete or rename 'generated_files' folder.")
            sys.exit()

        # Start Genetic Algorithm
        pop.run_genetic_algorithm()

        # Damage population
        pop.damage_population()

        print("Hello world")
    except KeyboardInterrupt:  # Allow for keybord interupt of script
        exit()
