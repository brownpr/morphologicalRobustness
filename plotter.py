import matplotlib.pyplot as plt
import os
import json
from collections import OrderedDict
from operator import getitem

UNDAMAGED_PERFORMANCE_FILE = 'performance_undamaged_evolution.json'
DAMAGED_PERFORMANCE_FILE = 'performance_damaged_evolution_remove_sect.json'
CREATURE_FILE = "evolution.json"
LAST_EPISODE_NUMBER = "14"
cwd = os.getcwd()
DATA_FOLDER_PATH = os.path.join(cwd, "generated_files")


def import_top_performers():

    generations = {}
    creatures = {}
    # Data extraction
    for subdir, dirs, files in os.walk(DATA_FOLDER_PATH):
        if CREATURE_FILE in files:
            creatures_file = open(os.path.join(subdir, CREATURE_FILE))
            creature_data = json.load(creatures_file)
            creatures_file.close()

            creature_name = creatures_file.name.replace(CREATURE_FILE, "").replace(DATA_FOLDER_PATH, "") \
                .replace("\\", "").replace("/", "")
            creature_performance_list = []
            creature_max_performance = 0

            for gen in creature_data:

                # rename generations
                if len(gen) == 5:  # gen number under 10
                    generation = gen.replace("_", "_00")
                elif len(gen) == 6:  # between gen number 10 and 99
                    generation = gen.replace("_", "_0")
                # If gen number exceeds 999 add a zero to the above replaces and uncomment this bellow
                # elif len(gen) == 7:  # between gen number 100 and 999
                #     generation = gen.replace("_", "_0")
                else:
                    generation = gen

                # get creature performance
                creature_performance = creature_data[gen]["ep_" + LAST_EPISODE_NUMBER]["fitness_eval"]

                if creature_name not in creatures:
                    creatures.update({creature_name: {"max": creature_max_performance,
                                                      "fitnesses": [{gen: creature_performance}]}})
                else:
                    if creature_performance > creature_max_performance:
                        creatures[creature_name]["max"] = creature_performance

                    creatures[creature_name]["fitnesses"].append({gen: creature_performance})

                # If no section for generation, create one
                if generation not in generations:
                    # As this data is the only one, give it default values for this creature
                    generations.update({generation: {"max": [creature_performance, creature_name],
                                                     "min": [creature_performance, creature_name]}})
                else:
                    if generations[generation]["max"][0] < creature_performance:
                        generations[generation]["max"][0] = creature_performance
                        generations[generation]["max"][1] = creature_name

                    if generations[generation]["min"][0] > creature_performance:
                        generations[generation]["min"][0] = creature_performance
                        generations[generation]["min"][1] = creature_name



    return generations, creatures


def import_creature_performance(creature_name):

    creature_performances = {}
    for subdir, dirs, files in os.walk(os.path.join(DATA_FOLDER_PATH, creature_name)):
        if CREATURE_FILE in files:
            creatures_file = open(os.path.join(subdir, CREATURE_FILE))
            creature_data = json.load(creatures_file)
            creatures_file.close()

            for gen in creature_data:

                # rename generations
                if len(gen) == 5:  # gen number under 10
                    generation = gen.replace("_", "_00")
                elif len(gen) == 6:  # between gen number 10 and 99
                    generation = gen.replace("_", "_0")
                # If gen number exceeds 999 add a zero to the above replaces and uncomment this bellow
                # elif len(gen) == 7:  # between gen number 100 and 999
                #     generation = gen.replace("_", "_0")
                else:
                    generation = gen

                # get creature performance
                creature_performance = creature_data[gen]["ep_" + LAST_EPISODE_NUMBER]["fitness_eval"]

                # If no section for generation, create one
                if generation not in creature_performances:
                    # As this data is the only one, give it default values for this creature
                    creature_performances.update({generation: creature_performance})

    return creature_performances


def plot_performance(data):

    performance_min_values = []
    performance_max_values = []
    performance_act_values = []
    sorted_generations = sorted(data.keys())
    
    max_performance = None
    for generation in sorted_generations:
        performance_min_values.append(data[generation]["min"][0])
        
        if max_performance is None:
            max_performance = data[generation]["max"][0]
        elif max_performance < data[generation]["max"][0]:
            max_performance = data[generation]["max"][0]

        performance_max_values.append(max_performance)

    plt.plot(sorted_generations, performance_max_values, label="max_performance")
    plt.plot(sorted_generations, performance_min_values, label="min_performance")
    plt.xticks(sorted_generations[::25], sorted_generations[::25], rotation=70)
    plt.grid(color="0.95")
    plt.legend()
    plt.show()
    pass


def plot_top_performers(data):

    top_performers = []
    sorted_generations = sorted(data.keys())

    for generation in sorted_generations:
        top_performer = data[generation]["max"][1]

        if top_performer not in top_performers:
            top_performers.append(top_performer)

    for creature in top_performers:
        creature_data = import_creature_performance(creature)

        creature_fitness = []

        for generation in sorted_generations:
            if generation not in creature_data:
                creature_fitness.append(None)
            else:
                creature_fitness.append(creature_data[generation])

        plt.plot(sorted_generations, creature_fitness, label=creature.replace("_", ""))

    plt.xticks(sorted_generations[::25], sorted_generations[::25], rotation=70)
    plt.grid(color="0.95")
    plt.legend()
    plt.show()


def plot_best_performers(creatures, num_creatures, num_gen):
    a = creatures.items()
    top_performers = sorted(creatures.items(), key=lambda x: getitem(x[1], "max"))[-num_creatures:]
    sorted_generations = []
    gen_str = "gen_" + str(num_gen - 1)
    for i in range(num_gen):
        i_str = "gen_" + str(i)
        while len(i_str) < len(gen_str):
            i_str = i_str.replace("_", "_0")
        sorted_generations.append(i_str)

    for creature in top_performers:
        creature_performance = creature[1]["fitnesses"]
        for i in range(len(creature_performance)):
            i_str = creature_performance[i].keys()[0]
            init_str = i_str
            a = len(i_str)
            b = len(gen_str)
            while len(i_str) < len(gen_str):
                i_str = i_str.replace("_", "_0")
            creature_performance[i][i_str] = creature_performance[i].pop(init_str)

        creature_performance = sorted(creature_performance)
        creature_lifespan = [x.keys()[0] for x in creature_performance]
        creature_fitness = [x.values()[0] for x in creature_performance]

        plt.plot(creature_lifespan, creature_fitness, label=creature[0].replace("_", ""))

    plt.xticks(sorted_generations[::25], sorted_generations[::25], rotation=70)
    plt.grid(color="0.95")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    performance_data, creature_data = import_top_performers()

    plot_best_performers(creature_data, 5, 150)
    plot_performance(performance_data)
    plot_top_performers(performance_data)
