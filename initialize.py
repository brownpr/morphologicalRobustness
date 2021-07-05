import datetime as dt
import os
import sys


from population import Population


if __name__ == "__main__":
    # Following code launches the evolutionary procedure of a set of creatures, please refer to settings.json to
    # edit launch parameters and default values

    try:
        # Create file to save creature files, if file exists stop sim
        if not os.path.exists("generated_files"):
            os.mkdir("generated_files")
        else:
            print("STOPPING SIMULATION: Creature files may already exist. Please delete or "
                  "rename the 'generated_files' folder.")
            sys.exit()

        # Create population
        pop = Population()

        # Start Genetic Algorithm
        pop.run_genetic_algorithm()

        # Save population
        pop.save_population()

        # Damage population
        damaged_pop = pop.inflict_damage(10)

        # Run genetic Algorithm on damaged population
        damaged_pop.run_genetic_algorithm()

        # Save damaged population
        damaged_pop.save_population()

        print("Hello world")  # Used as a
    except KeyboardInterrupt:  # Allow for keybord interupt of script
        exit()
