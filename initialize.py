from population_async import Population
import os


if __name__ == "__main__":
    try:
        # Create folder to save creature files, if file exists stop sim
        if not os.path.exists("generated_files"):
            os.mkdir("generated_files")
        else:
            raise Exception("STOPPING SIMULATION: The 'generated_files' folder already exist. Please delete or "
                  "rename the 'generated_files' folder.")

        # Create initial undamaged population
        undamaged_population = Population()

        # Start Genetic Algorithm
        undamaged_population.run_genetic_algorithm()

        # Save population to pickle file
        undamaged_population.save_population()

    except KeyboardInterrupt:
        exit()
