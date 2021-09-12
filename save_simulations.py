import datetime
import sys

from population_async import Population
import os

if __name__ == "__main__":
    try:
        # Create file to save creature files, if file exists stop sim
        if not os.path.exists("generated_files"):
            raise Exception("STOPPING SIMULATION: The 'generated_files' folder does not exist. Please copy folder into "
                            "cwd or initialize a simulation before running save_simulations.py")

        loaded_population = Population(load_population=True)
        loaded_population.save_evolutionary_history()

        print(str(datetime.datetime.now()) + "Saved evolutionary history for "
              + str(len(loaded_population.full_population)) + " creatures")

    except KeyboardInterrupt:
        sys.exit()
