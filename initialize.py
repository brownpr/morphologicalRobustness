from population import Population


if __name__ == "__main__":
    try:
        # Create initial undamaged population
        undamaged_population = Population()

        # Start Genetic Algorithm
        undamaged_population.run_genetic_algorithm()

        # Save population
        undamaged_population.save_population()

        # Damage population
        num_creatures_to_damage = 20
        undamaged_population_sorted, top_creature = undamaged_population.sort_population(undamaged_population
                                                                                         .full_population)
        damaged_population_dict = {crt.name: crt for crt in undamaged_population_sorted[0:num_creatures_to_damage]}

        damaged_population = Population(population=damaged_population_dict, damaged_population=False)

        # Inflict damage on each creature (choose damage type)
        for creature in damaged_population.population.values():
            creature.remove_voxels_sections((1, 3))

        print("HelloWorld")

    except KeyboardInterrupt:
        exit()