import json
import os
import time
import re
import subprocess as sub
import multiprocessing

# import settings
settings_file = open("settings.json")
SETTINGS = json.load(settings_file)
settings_file.close()


def evaluate_creature(creature_vxa):
    sub.Popen(SETTINGS["evosoro_path"] + " -f  " + creature_vxa, shell=True)


def evaluate_specific_vxa(creature_name, number_of_evaluations=1):
    # Run evaluation for specific a singular specific vxa files. This is repeated "number_of_evaluations" times.

    if ".vxa" not in creature_name:
        raise Exception("Please provide the file path to the .vxa file to be evaluated.")

    fitness_evaluations = []
    for _ in range(number_of_evaluations):
        evaluate_creature(creature_name)

        fitness_file_name = creature_name.replace(".vxa", "_fitness.xml")
        ffp = os.path.join(os.getcwd(), fitness_file_name)  # fitness file path
        pfp = os.path.join(os.getcwd(), "pressures" + fitness_file_name + ".csv")
        kefp = os.path.join(os.getcwd(), "ke" + fitness_file_name + ".csv")
        sfp = os.path.join(os.getcwd(), "strain" + fitness_file_name + ".csv")

        # wait for file to appear, if two minutes passes and there is no file raise exception
        t = time.time()
        while not os.path.exists(pfp) or not os.path.exists(ffp):
            time.sleep(1)
            toc = time.time() - t
            if toc > 300:
                raise Exception("ERROR: No pressure file or fitness file for " + creature_name +
                                " found after 300 seconds. This error is commonly due to problems in the created"
                                " vxa file.")

        tag_x = "<normDistX>(.*)</normDistX>"
        tag_y = "<normDistY>(.*)</normDistY>"
        tag_z = "<normDistZ>(.*)</normDistZ>"
        
        while True:
            tic = time.time()
            with open(fitness_file_name, "r") as fitness_file:
                read_file = fitness_file.read()
                x_match = re.search(tag_x, read_file)
                y_match = re.search(tag_y, read_file)
                z_match = re.search(tag_z, read_file)
            fitness_file.close()

            if x_match and y_match and z_match:
                result_x = float(x_match.group(1))
                result_y = float(y_match.group(1))
                result_z = float(z_match.group(1))
                break
            time.sleep(5)
            toc = time.time() - tic
            if toc > 120:
                raise Exception("Error while importing data from fitness evaluation. Fitness file name: "
                                + fitness_file_name)

        if SETTINGS["fitness_evaluation"]["take_absolutes"][0]:
            result_x = abs(result_x)
        if SETTINGS["fitness_evaluation"]["take_absolutes"][1]:
            result_y = abs(result_y)
        if SETTINGS["fitness_evaluation"]["take_absolutes"][2]:
            result_z = abs(result_z)

        m = SETTINGS["fitness_evaluation"]["M"]
        m_x = SETTINGS["fitness_evaluation"]["Mx"]
        m_y = SETTINGS["fitness_evaluation"]["My"]
        m_z = SETTINGS["fitness_evaluation"]["Mz"]
        n = SETTINGS["fitness_evaluation"]["N"]
        n_x = SETTINGS["fitness_evaluation"]["Nx"]
        n_y = SETTINGS["fitness_evaluation"]["Ny"]
        n_z = SETTINGS["fitness_evaluation"]["Nz"]

        # Calculate fitness, punish for locomotion that is not in a straight line
        fitness_eval = m * (m_x * (result_x**n_x) +
                            m_y * (result_y**n_y) +
                            m_z * (result_z**n_z))**n

        fitness_evaluations.append(fitness_eval)

        # os.remove(pfp)
        # os.remove(kefp)
        # os.remove(sfp)
        # os.remove(ffp)

    print(fitness_evaluations)


if __name__ == "__main__":

    vxa_files = ["_creature0_gen0_ep14.vxa"]

    for vxa_file in vxa_files:
        evaluate_specific_vxa(vxa_file, number_of_evaluations=1)
