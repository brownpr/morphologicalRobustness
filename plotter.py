import matplotlib.pyplot as plt
import os
import json

UNDAMAGED_PERFORMANCE_FILE = 'performance_undamaged_evolution.json'
DAMAGED_PERFORMANCE_FILE = 'performance_damaged_evolution_remove_sect.json'
CREATURE_FILE = "evolution.json"
LAST_EPISODE_NUMBER = "14"


def creature_plots(data_folder_fp):

    performances = {}
    neural_networks = {}
    morphological_evolution = {}
    stiffness_evolution = {}

    for subdir, dirs, files in os.walk(data_folder_fp):

        if UNDAMAGED_PERFORMANCE_FILE in files:
            undamaged_creatures_file = open(os.path.join(subdir, UNDAMAGED_PERFORMANCE_FILE))
            undamaged_creatures_data = json.load(undamaged_creatures_file)
            undamaged_creatures_file.close()

            # turn list of dictionaries into dictionary
            y_values = []
            x_labels = []
            for item in undamaged_creatures_data:
                y_values.append(list(item.values())[0])
                x_labels.append(list(item.keys())[0])

            x_labels.reverse()
            y_values.reverse()
            x_values = range(len(undamaged_creatures_data))

            plt.plot(x_values, y_values)
            plt.xticks(x_values, x_labels, rotation=70)
            plt.show()

        if DAMAGED_PERFORMANCE_FILE in files:
            damaged_creatures_file = open(os.path.join(subdir, DAMAGED_PERFORMANCE_FILE))
            damaged_creatures_data = json.load(damaged_creatures_file)
            damaged_creatures_file.close()

            # turn list of dictionaries into dictionary
            y_values = []
            x_labels = []
            for item in damaged_creatures_data:
                y_values.append(list(item.values())[0])
                x_labels.append(list(item.keys())[0])

            x_labels.reverse()
            y_values.reverse()
            x_values = range(len(damaged_creatures_data))

            plt.plot(x_values, y_values)
            plt.xticks(x_values, x_labels, rotation=70)
            plt.show()

        if CREATURE_FILE in files:
            creatures_file = open(os.path.join(subdir, CREATURE_FILE))
            creature_data = json.load(creatures_file)
            creatures_file.close()

            creature_name = creatures_file.name.replace(CREATURE_FILE, "").replace(data_folder_fp, "").replace("\\", "")

            if creature_name not in neural_networks:
                neural_networks.update({creature_name: {}})
            if creature_name not in morphological_evolution:
                morphological_evolution.update({creature_name: {}})
            if creature_name not in stiffness_evolution:
                stiffness_evolution.update({creature_name: {}})

            for generation in creature_data.keys():

                if len(generation) == 5:
                    gen = generation.replace("_", "_0")
                else:
                    gen = generation

                if gen not in morphological_evolution[creature_name]:
                    morphological_evolution[creature_name].update({gen: {}})

                if gen not in stiffness_evolution[creature_name]:
                    stiffness_evolution[creature_name].update({gen: {}})

                for episode in creature_data[generation].keys():
                    if episode == "nn_parameters":
                        neural_networks[creature_name].update({gen: creature_data[generation]["nn_parameters"]})
                        break
                    elif len(episode) == 4:
                        ep = episode.replace("_", "_0")
                    else:
                        ep = episode

                    if gen not in performances:
                        performances.update({gen: {
                            "max": creature_data[generation][episode]["fitness_eval"],
                            "min": creature_data[generation][episode]["fitness_eval"]}})

                    if creature_data[generation][episode]["fitness_eval"] > performances[gen]["max"]:
                        performances[gen]["max"] = creature_data[generation][episode]["fitness_eval"]

                    if creature_data[generation][episode]["fitness_eval"] < performances[gen]["min"]:
                        performances[gen]["min"] = creature_data[generation][episode]["fitness_eval"]

                    if ep not in morphological_evolution[creature_name][gen]:
                        morphological_evolution[creature_name][gen].update({
                            ep: creature_data[generation][episode]["morphology"]})

                    if ep not in stiffness_evolution[creature_name][gen]:
                        stiffness_evolution[creature_name][gen].update({
                            ep: creature_data[generation][episode]["stiffness"]})

    performance_min_values = []
    performance_max_values = []
    performance_x_values = []

    sorted_performances = {}
    for i in sorted(performances):
        sorted_performances[i] = performances[i]
    for generation, performance in sorted_performances.items():
        performance_x_values.append(generation)
        performance_max_values.append(performance["max"])
        performance_min_values.append(performance["min"])

    plt.plot(performance_x_values, performance_max_values, label="max_performance")
    plt.plot(performance_x_values, performance_min_values, label="min_performance")
    plt.xticks(performance_x_values, performance_x_values, rotation=70)
    plt.grid(color="0.95")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    cwd = os.getcwd()
    folder_path = os.path.join(cwd, "generated_files")

    creature_plots(folder_path)
