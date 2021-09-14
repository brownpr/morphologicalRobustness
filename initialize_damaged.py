import os
from utls import rename_gf_folder

from population_async import Population

if __name__ == "__main__":
    try:
        pkl_file_name = "previous_population.pkl"

        if os.path.exists("generated_files"):
            rename_gf_folder("undamaged_evaluations")
        else:
            os.mkdir("generated_files")

        if not os.path.exists(pkl_file_name):
            raise Exception("No .pkl file found. In 'initialize_damaged.py' check that the pkl_file_name var. string"
                            "matches the population you wish to damage and that the file is found within the cwd.")

        # Load previous population
        undamaged_population = Population(load_population=True)

        # Create a new population dictionary from the top performing creatures from the loaded population
        num_creatures_to_damage = len(undamaged_population.population)  # Set size of damaged population
        undamaged_population_sorted, _ = undamaged_population.sort_population(undamaged_population.full_population)
        damaged_population_dict = {crt.name: crt for crt in undamaged_population_sorted[0:num_creatures_to_damage]}

        # Use population to delete loaded population
        population_to_damage = Population(population=damaged_population_dict, is_damaged=True, reset_evolution=True)
        del undamaged_population
        del damaged_population_dict
        del undamaged_population_sorted

        # COPY POPULATIONS, DAMAGE THEM, EVALUATE, SAVE AND DELETE
        
        # HALF DAMAGED CREATURES:
        # Damaged right half
        damaged_half_0 = population_to_damage.deepcopy()
        damaged_half_0.inflict_damage(damage_type="remove_sect", damage_arguments=[(0, 2, 4, 6)])
        damaged_half_0.run_genetic_algorithm(generation_size=1)
        damaged_half_0.save_population(name="damaged_right_half")
        del damaged_half_0  # Save memory
        rename_gf_folder("damaged_right_half")  # rename gf folder and create new one
        # Remove left half
        damaged_half_1 = population_to_damage.deepcopy()
        damaged_half_1.inflict_damage(damage_type="remove_sect", damage_arguments=[(1, 3, 5, 7)])
        damaged_half_1.run_genetic_algorithm(generation_size=1)
        damaged_half_1.save_population(name="damaged_left_half")
        del damaged_half_1  # Save memory
        rename_gf_folder("damaged_left_half")
        # Remove front half
        damaged_half_2 = population_to_damage.deepcopy()
        damaged_half_2.inflict_damage(damage_type="remove_sect", damage_arguments=[(2, 3, 6, 7)])
        damaged_half_2.run_genetic_algorithm(generation_size=1)
        damaged_half_2.save_population(name="damaged_front_half")
        del damaged_half_2  # Save memory
        rename_gf_folder("damaged_front_half")
        # Remove back half
        damaged_half_3 = population_to_damage.deepcopy()
        damaged_half_3.inflict_damage(damage_type="remove_sect", damage_arguments=[(0, 1, 4, 5)])
        damaged_half_3.run_genetic_algorithm(generation_size=1)
        damaged_half_3.save_population(name="damaged_back_half")
        del damaged_half_3  # Save memory
        rename_gf_folder("damaged_back_half")
        # Remove bottom halfright
        damaged_half_4 = population_to_damage.deepcopy()
        damaged_half_4.inflict_damage(damage_type="remove_sect", damage_arguments=[(0, 1, 2, 3)])
        damaged_half_4.run_genetic_algorithm(generation_size=1)
        damaged_half_4.save_population(name="damaged_bottom_half")
        del damaged_half_4  # Save memory
        rename_gf_folder("damaged_bottom_half")
        # Remove top half
        damaged_half_5 = population_to_damage.deepcopy()
        damaged_half_5.inflict_damage(damage_type="remove_sect", damage_arguments=[(4, 5, 6, 7)])
        damaged_half_5.run_genetic_algorithm(generation_size=1)
        damaged_half_5.save_population(name="damaged_top_half")
        del damaged_half_5  # Save memory
        rename_gf_folder("damaged_top_half")

        # QUARTER DAMAGED CREATURES
        # Lower back quarter
        damaged_quarter_0 = population_to_damage.deepcopy()
        damaged_quarter_0.inflict_damage(damage_type="remove_sect", damage_arguments=[(0, 1)])
        damaged_quarter_0.run_genetic_algorithm(generation_size=1)
        damaged_quarter_0.save_population(name="lower_back_quarter")
        del damaged_quarter_0  # Save memory
        rename_gf_folder("lower_back_quarter")
        # lower_right_quarter
        damaged_quarter_1 = population_to_damage.deepcopy()
        damaged_quarter_1.inflict_damage(damage_type="remove_sect", damage_arguments=[(0, 2)])
        damaged_quarter_1.run_genetic_algorithm(generation_size=1)
        damaged_quarter_1.save_population(name="lower_right_quarter")
        del damaged_quarter_1  # Save memory
        rename_gf_folder("lower_right_quarter")
        # lower_left_quarter
        damaged_quarter_3 = population_to_damage.deepcopy()
        damaged_quarter_3.inflict_damage(damage_type="remove_sect", damage_arguments=[(1, 3)])
        damaged_quarter_3.run_genetic_algorithm(generation_size=1)
        damaged_quarter_3.save_population(name="lower_left_quarter")
        del damaged_quarter_3  # Save memory
        rename_gf_folder("lower_left_quarter")
        # lower_front_quarter
        damaged_quarter_5 = population_to_damage.deepcopy()
        damaged_quarter_5.inflict_damage(damage_type="remove_sect", damage_arguments=[(2, 3)])
        damaged_quarter_5.run_genetic_algorithm(generation_size=1)
        damaged_quarter_5.save_population(name="lower_front_quarter")
        del damaged_quarter_5  # Save memory
        rename_gf_folder("lower_front_quarter")
        # back_right_quarter
        damaged_quarter_2 = population_to_damage.deepcopy()
        damaged_quarter_2.inflict_damage(damage_type="remove_sect", damage_arguments=[(0, 4)])
        damaged_quarter_2.run_genetic_algorithm(generation_size=1)
        damaged_quarter_2.save_population(name="back_right_quarter")
        del damaged_quarter_2  # Save memory
        rename_gf_folder("back_right_quarter")
        # back_left_quarter
        damaged_quarter_4 = population_to_damage.deepcopy()
        damaged_quarter_4.inflict_damage(damage_type="remove_sect", damage_arguments=[(1, 5)])
        damaged_quarter_4.run_genetic_algorithm(generation_size=1)
        damaged_quarter_4.save_population(name="back_left_quarter")
        del damaged_quarter_4  # Save memory
        rename_gf_folder("back_left_quarter")
        # front_right_quarter
        damaged_quarter_6 = population_to_damage.deepcopy()
        damaged_quarter_6.inflict_damage(damage_type="remove_sect", damage_arguments=[(2, 6)])
        damaged_quarter_6.run_genetic_algorithm(generation_size=1)
        damaged_quarter_6.save_population(name="front_right_quarter")
        del damaged_quarter_6  # Save memory
        rename_gf_folder("front_right_quarter")
        # front_left_quarter
        damaged_quarter_7 = population_to_damage.deepcopy()
        damaged_quarter_7.inflict_damage(damage_type="remove_sect", damage_arguments=[(3, 7)])
        damaged_quarter_7.run_genetic_algorithm(generation_size=1)
        damaged_quarter_7.save_population(name="front_left_quarter")
        del damaged_quarter_7  # Save memory
        rename_gf_folder("front_left_quarter")
        # top_back_quarter
        damaged_quarter_8 = population_to_damage.deepcopy()
        damaged_quarter_8.inflict_damage(damage_type="remove_sect", damage_arguments=[(4, 5)])
        damaged_quarter_8.run_genetic_algorithm(generation_size=1)
        damaged_quarter_8.save_population(name="top_back_quarter")
        del damaged_quarter_8  # Save memory
        rename_gf_folder("top_back_quarter")
        # top_right_quarter
        damaged_quarter_9 = population_to_damage.deepcopy()
        damaged_quarter_9.inflict_damage(damage_type="remove_sect", damage_arguments=[(4, 6)])
        damaged_quarter_9.run_genetic_algorithm(generation_size=1)
        damaged_quarter_9.save_population(name="top_right_quarter")
        del damaged_quarter_9  # Save memory
        rename_gf_folder("top_right_quarter")
        # top_left_quarter
        damaged_quarter_10 = population_to_damage.deepcopy()
        damaged_quarter_10.inflict_damage(damage_type="remove_sect", damage_arguments=[(5, 7)])
        damaged_quarter_10.run_genetic_algorithm(generation_size=1)
        damaged_quarter_10.save_population(name="top_left_quarter")
        del damaged_quarter_10  # Save memory
        rename_gf_folder("top_left_quarter")
        # top_front_quarter
        damaged_quarter_11 = population_to_damage.deepcopy()
        damaged_quarter_11.inflict_damage(damage_type="remove_sect", damage_arguments=[(6, 7)])
        damaged_quarter_11.run_genetic_algorithm(generation_size=1)
        damaged_quarter_11.save_population(name="top_front_quarter")
        del damaged_quarter_11  # Save memory
        rename_gf_folder("top_front_quarter")

        # EIGHTH DAMAGED CREATURES
        # lower_back_right_eighth
        damaged_eighth_0 = population_to_damage.deepcopy()
        damaged_eighth_0.inflict_damage(damage_type="remove_sect", damage_arguments=[(0)])
        damaged_eighth_0.run_genetic_algorithm(generation_size=1)
        damaged_eighth_0.save_population(name="lower_back_right_eighth")
        del damaged_eighth_0  # Save memory
        rename_gf_folder("lower_back_right_eighth")
        # lower_back_left_eighth
        damaged_eighth_1 = population_to_damage.deepcopy()
        damaged_eighth_1.inflict_damage(damage_type="remove_sect", damage_arguments=[(1)])
        damaged_eighth_1.run_genetic_algorithm(generation_size=1)
        damaged_eighth_1.save_population(name="lower_back_left_eighth")
        del damaged_eighth_1  # Save memory
        rename_gf_folder("lower_back_left_eighth")
        # lower_front_right_eighth
        damaged_eighth_1 = population_to_damage.deepcopy()
        damaged_eighth_1.inflict_damage(damage_type="remove_sect", damage_arguments=[(2)])
        damaged_eighth_1.run_genetic_algorithm(generation_size=1)
        damaged_eighth_1.save_population(name="lower_front_right_eighth")
        del damaged_eighth_1  # Save memory
        rename_gf_folder("lower_front_right_eighth")
        # lower_front_left_eighth
        damaged_eighth_2 = population_to_damage.deepcopy()
        damaged_eighth_2.inflict_damage(damage_type="remove_sect", damage_arguments=[(3)])
        damaged_eighth_2.run_genetic_algorithm(generation_size=1)
        damaged_eighth_2.save_population(name="lower_front_left_eighth")
        del damaged_eighth_2  # Save memory
        rename_gf_folder("lower_front_left_eighth")
        # top_back_right_eighth
        damaged_eighth_3 = population_to_damage.deepcopy()
        damaged_eighth_3.inflict_damage(damage_type="remove_sect", damage_arguments=[(4)])
        damaged_eighth_3.run_genetic_algorithm(generation_size=1)
        damaged_eighth_3.save_population(name="top_back_right_eighth")
        del damaged_eighth_3  # Save memory
        rename_gf_folder("top_back_right_eighth")
        # top_back_left_eighth
        damaged_eighth_4 = population_to_damage.deepcopy()
        damaged_eighth_4.inflict_damage(damage_type="remove_sect", damage_arguments=[(5)])
        damaged_eighth_4.run_genetic_algorithm(generation_size=1)
        damaged_eighth_4.save_population(name="top_back_left_eighth")
        del damaged_eighth_4  # Save memory
        rename_gf_folder("top_back_left_eighth")
        # top_front_right_eighth
        damaged_eighth_5 = population_to_damage.deepcopy()
        damaged_eighth_5.inflict_damage(damage_type="remove_sect", damage_arguments=[(6)])
        damaged_eighth_5.run_genetic_algorithm(generation_size=1)
        damaged_eighth_5.save_population(name="top_front_right_eighth")
        del damaged_eighth_5  # Save memory
        rename_gf_folder("top_front_right_eighth")
        # top_front_left_eighth
        damaged_eighth_6 = population_to_damage.deepcopy()
        damaged_eighth_6.inflict_damage(damage_type="remove_sect", damage_arguments=[(7)])
        damaged_eighth_6.run_genetic_algorithm(generation_size=1)
        damaged_eighth_6.save_population(name="top_front_left_eighth")
        del damaged_eighth_6  # Save memory
        rename_gf_folder("top_front_left_eighth")

    except KeyboardInterrupt:
        exit()


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