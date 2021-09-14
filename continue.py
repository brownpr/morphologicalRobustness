import datetime
import sys

from population_async import Population
import os

if __name__ == "__main__":
    try:

        pkl_file_name = "previous_population.pkl"

        if not os.path.exists("generated_files"):
            raise Exception("STOPPING SIMULATION: The 'generated_files' folder does not exist. "
                            "Please run 'initialize.py' before 'continue.py'.")

        if not os.path.exists(pkl_file_name):
            raise Exception("No .pkl file found. Within 'continue.py' ensure that the pkl_file_name var. string"
                            "matches the population you wish to damage and that the file is found within the cwd.")

        loaded_population = Population(load_population=True)

        # Create new population to evaluate
        loaded_population.new_population()

        # Evaluate population
        loaded_population.run_genetic_algorithm()
        loaded_population.save_population()

        print(str(datetime.datetime.now()) + "-----FINISHED EVALUATION-----")

    except KeyboardInterrupt:
        sys.exit()
