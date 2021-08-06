import matplotlib.pyplot as plt
import os
import json
import numpy


def creature_plots(data_folder_fp, undamaged_creatures_file_name, damaged_creatures_file_name=None, creature_files="evolution.json"):
    for subdir, dirs, files in os.walk(data_folder_fp):
        if undamaged_creatures_file_name in files:
            undamaged_creatures_file = open(os.path.join(subdir, undamaged_creatures_file_name))
            undamaged_creatures_data = json.load(undamaged_creatures_file)
            undamaged_creatures_file.close()

            # turn list of dictionaries into dictionary
            y_values = []
            x_labels = []
            for item in undamaged_creatures_data:
                y_values.append(item.values()[0])
                x_labels.append(item.keys()[0])

            x_labels.reverse()
            y_values.reverse()
            x_values = range(len(undamaged_creatures_data))

            plt.plot(x_values, y_values)
            plt.xticks(x_values, x_labels, rotation=70)
            plt.show()
            pass
        if damaged_creatures_file_name in files:
            damaged_creatures_file = open(os.path.join(subdir, damaged_creatures_file_name))
            damaged_creatures_data = json.load(damaged_creatures_file)
            damaged_creatures_file.close()

            # turn list of dictionaries into dictionary
            y_values = []
            x_labels = []
            for item in damaged_creatures_data:
                y_values.append(item.values()[0])
                x_labels.append(item.keys()[0])

            x_labels.reverse()
            y_values.reverse()
            x_values = range(len(damaged_creatures_data))

            plt.plot(x_values, y_values)
            plt.xticks(x_values, x_labels, rotation=70)
            plt.show()
            pass


if __name__ == "__main__":
    cwd = os.getcwd()
    folder_path = os.path.join(cwd, "old/160721")

    undamaged_creatures_performance_file = 'performance_undamaged_evolution.json'
    damaged_creatures_performance_file = 'performance_damaged_evolution_remove_sect.json'

    creature_plots(folder_path, undamaged_creatures_performance_file, damaged_creatures_performance_file)