import datetime

from population_async import Population
import os

if __name__ == "__main__":
    try:
        # Damage population
        undamaged_population = Population(load_population=True)
        num_creatures_to_damage = 20
        undamaged_population_sorted, top_creature = undamaged_population.sort_population(undamaged_population
                                                                                         .full_population)
        damaged_population_dict = {crt.name: crt for crt in undamaged_population_sorted[0:num_creatures_to_damage]}

        damaged_population = Population(population=damaged_population_dict, is_damaged=True, reset_evolution=True)

        # List of damage types and their arguments:
        #
        # SECTION DAMAGES:
        # DAMAGE_TYPE               ARGUMENTS (length -- types -- definition) (Input arguments must be placed in a list)
        # remove_sect               1 -- tuple -- list of section numbers to remove
        # stiff_sect_mult           2 -- tuple, float -- list of section numbers to affect, multiply stiffness by input
        # stiff_sect_div            2 -- tuple, float -- list of section numbers to affect, divide stiffness by input
        # stiff_sect_set            2 -- tuple, float -- list of section numbers to affect, change stiffness to input
        # stiff_sect_add            2 -- tuple, float -- list of section numbers to affect, increase stiffness by input
        # stiff_sect_red            2 -- tuple, float -- list of section numbers to affect, reduce stiffness by input
        #
        # SPHERICAL DAMAGES:
        # DAMAGE_TYPE               ARGUMENTS (length -- types -- definition)
        # remove_spher              2 -- tuple, int -- centre voxel coordinates, affected radius
        # stiff_spher_mult          3 -- tuple, int, float -- centre voxel coordinates, affected radius, multiply stiffness by input
        # stiff_spher_div           3 -- tuple, int, float -- centre voxel coordinates, affected radius, divide stiffness by input
        # stiff_spher_set           3 -- tuple, int, float -- centre voxel coordinates, affected radius, change stiffness to input
        # stiff_spher_add           3 -- tuple, int, float -- centre voxel coordinates, affected radius, increase stiffness by input
        # stiff_spher_red           3 -- tuple, int, float -- centre voxel coordinates, affected radius, reduce stiffness by input
        #

        # Inflict damage on each creature (choose damage type)
        damaged_population.inflict_damage(damage_type="remove_sect", damage_arguments=[(1, 3)], damage_base_creature=True)
        # damaged_population.inflict_damage(damage_type="stiff_spher_red", damage_arguments=[(0, 0, 0), 3, 200000],
        #                                   damage_base_creature=True)

        # Evaluate creatures for n evaluations
        damaged_population.run_genetic_algorithm()

        # Save population
        # damaged_population.save_evolutionary_history()

        print(str(datetime.datetime.now()) + "-----FINISHED EVALUATION-----")

    except KeyboardInterrupt:
        exit()
