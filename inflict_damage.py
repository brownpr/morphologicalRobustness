import numpy as np


def remove_eighths(layers_array, xyz_voxels):

    # get voxel dimensions
    x_voxels = xyz_voxels[0]
    y_voxels = xyz_voxels[1]
    z_voxels = xyz_voxels[2]

    # set empty arrays for the four quarters of the damage
    damage_1 = np.ones([x_voxels, y_voxels], dtype=int)
    damage_2 = np.ones([x_voxels, y_voxels], dtype=int)
    damage_3 = np.ones([x_voxels, y_voxels], dtype=int)
    damage_4 = np.ones([x_voxels, y_voxels], dtype=int)

    # set empty arrays for damage of the eighths
    damage_layer_1 = []
    damage_layer_2 = []
    damage_layer_3 = []
    damage_layer_4 = []
    damage_layer_5 = []
    damage_layer_6 = []
    damage_layer_7 = []
    damage_layer_8 = []

    # setting damaged areas
    for ii in range(y_voxels):
        if ii < y_voxels/2:
            for jj in range(x_voxels):
                if jj < x_voxels/2:
                    damage_1[ii, jj] = 0
                else:
                    damage_2[ii, jj] = 0
        else:
            for jj in range(x_voxels):
                if jj < x_voxels/2:
                    damage_3[ii, jj] = 0
                else:
                    damage_4[ii, jj] = 0

    if x_voxels%2 == 0 and y_voxels%2 == 0 and z_voxels%2 == 0:
        # Cube region with even lengths x_voxels, y_voxels and z_voxels
        for k in range(z_voxels):
            row = layers_array[k]

            if k < z_voxels/2:
                # convert to array
                row_array = np.array(np.array_split(row, x_voxels))

                # Apply damage to each row
                row_damage_1 = np.multiply(row_array, damage_1)
                row_damage_2 = np.multiply(row_array, damage_2)
                row_damage_3 = np.multiply(row_array, damage_3)
                row_damage_4 = np.multiply(row_array, damage_4)

                # Convert to list
                row_damg_1 = [elem for lst in row_damage_1.tolist() for elem in lst]
                row_damg_2 = [elem for lst in row_damage_2.tolist() for elem in lst]
                row_damg_3 = [elem for lst in row_damage_3.tolist() for elem in lst]
                row_damg_4 = [elem for lst in row_damage_4.tolist() for elem in lst]

                # Apply damage to layers
                damage_layer_1.append(row_damg_1)
                damage_layer_2.append(row_damg_2)
                damage_layer_3.append(row_damg_3)
                damage_layer_4.append(row_damg_4)
                # Append unaffected layers
                damage_layer_5.append(row.tolist())
                damage_layer_6.append(row.tolist())
                damage_layer_7.append(row.tolist())
                damage_layer_8.append(row.tolist())


            else:
                # convert to array
                row_array = np.array(np.array_split(row, x_voxels))

                # Apply damage to each row
                row_damage_5 = np.multiply(row_array, damage_1)
                row_damage_6 = np.multiply(row_array, damage_2)
                row_damage_7 = np.multiply(row_array, damage_3)
                row_damage_8 = np.multiply(row_array, damage_4)

                # Convert to list
                row_damg_5 = [elem for lst in row_damage_5.tolist() for elem in lst]
                row_damg_6 = [elem for lst in row_damage_6.tolist() for elem in lst]
                row_damg_7 = [elem for lst in row_damage_7.tolist() for elem in lst]
                row_damg_8 = [elem for lst in row_damage_8.tolist() for elem in lst]

                # Append unaffected layers
                damage_layer_1.append(row.tolist())
                damage_layer_2.append(row.tolist())
                damage_layer_3.append(row.tolist())
                damage_layer_4.append(row.tolist())
                # Apply damage to layers
                damage_layer_5.append(row_damg_5)
                damage_layer_6.append(row_damg_6)
                damage_layer_7.append(row_damg_7)
                damage_layer_8.append(row_damg_8)

    eighths_layers_dic = {"eighths_creature_1": damage_layer_1,
                  "eighths_creature_2": damage_layer_2,
                  "eighths_creature_3": damage_layer_3,
                  "eighths_creature_4": damage_layer_4,
                  "eighths_creature_5": damage_layer_5,
                  "eighths_creature_6": damage_layer_6,
                  "eighths_creature_7": damage_layer_7,
                  "eighths_creature_8": damage_layer_8
                          }

    return eighths_layers_dic


def remove_halfs(layers_array, xyz_voxels):

    # get voxel dimensions
    x_voxels = xyz_voxels[0]
    y_voxels = xyz_voxels[1]
    z_voxels = xyz_voxels[2]

    # set empty arrays for the four quarters of the damage
    damage_1 = np.ones([x_voxels, y_voxels], dtype=int)
    damage_2 = np.ones([x_voxels, y_voxels], dtype=int)
    damage_3 = np.ones([x_voxels, y_voxels], dtype=int)
    damage_4 = np.ones([x_voxels, y_voxels], dtype=int)
    damage_5 = np.zeros([x_voxels, y_voxels], dtype=int)

    # set empty arrays for damage of the halfs
    damage_layer_1 = []
    damage_layer_2 = []
    damage_layer_3 = []
    damage_layer_4 = []
    damage_layer_5 = []
    damage_layer_6 = []

    for ii in range(y_voxels):
        if ii < y_voxels/2:
            for jj in range(x_voxels):
                if jj < x_voxels/2:
                    damage_1[ii, jj] = 0
                else:
                    damage_2[ii, jj] = 0
        else:
            for jj in range(x_voxels):
                if jj < x_voxels/2:
                    damage_3[ii, jj] = 0
                else:
                    damage_4[ii, jj] = 0

    if x_voxels%2 == 0 and y_voxels%2 == 0 and z_voxels%2 == 0:
        # Cube region with even lengths x_voxels, y_voxels and z_voxels
        for k in range(z_voxels):
            row = layers_array[k]
            if k < z_voxels/2:
                row_array = np.array(np.array_split(row, x_voxels))

                # Apply damage to each row
                row_damage_1 = np.multiply(row_array, damage_5)
                row_damage_2 = np.multiply(np.multiply(row_array, damage_1), damage_2)
                row_damage_3 = np.multiply(np.multiply(row_array, damage_4), damage_3)
                row_damage_4 = np.multiply(np.multiply(row_array, damage_1), damage_3)
                row_damage_5 = np.multiply(np.multiply(row_array, damage_2), damage_4)

                # Convert to list
                row_damg_1 = [elem for lst in row_damage_1.tolist() for elem in lst]
                row_damg_2 = [elem for lst in row_damage_2.tolist() for elem in lst]
                row_damg_3 = [elem for lst in row_damage_3.tolist() for elem in lst]
                row_damg_4 = [elem for lst in row_damage_4.tolist() for elem in lst]
                row_damg_5 = [elem for lst in row_damage_5.tolist() for elem in lst]

                # Apply damage to layers
                damage_layer_1.append(row_damg_1)
                damage_layer_2.append(row_damg_2)
                damage_layer_3.append(row_damg_3)
                damage_layer_4.append(row_damg_4)
                damage_layer_5.append(row_damg_5)
                # Append unaffected layers
                damage_layer_6.append(row.tolist())

            else:
                # convert to array
                row_array = np.array(np.array_split(row, x_voxels))

                # Apply damage to each row
                row_damage_2 = np.multiply(np.multiply(row_array, damage_1), damage_2)
                row_damage_3 = np.multiply(np.multiply(row_array, damage_4), damage_3)
                row_damage_4 = np.multiply(np.multiply(row_array, damage_1), damage_3)
                row_damage_5 = np.multiply(np.multiply(row_array, damage_2), damage_4)
                row_damage_6 = np.multiply(row_array, damage_5)

                # Convert to list
                row_damg_2 = [elem for lst in row_damage_2.tolist() for elem in lst]
                row_damg_3 = [elem for lst in row_damage_3.tolist() for elem in lst]
                row_damg_4 = [elem for lst in row_damage_4.tolist() for elem in lst]
                row_damg_5 = [elem for lst in row_damage_5.tolist() for elem in lst]
                row_damg_6 = [elem for lst in row_damage_6.tolist() for elem in lst]

                # Append unaffected layers
                damage_layer_1.append(row.tolist())
                # Apply damage to layers
                damage_layer_2.append(row_damg_2)
                damage_layer_3.append(row_damg_3)
                damage_layer_4.append(row_damg_4)
                damage_layer_5.append(row_damg_5)
                damage_layer_6.append(row_damg_6)

            half_layers_dic = {"half_creature_1": damage_layer_1,
                               "half_creature_2": damage_layer_2,
                               "half_creature_3": damage_layer_3,
                               "half_creature_4": damage_layer_4,
                               "half_creature_5": damage_layer_5,
                               "half_creature_6": damage_layer_6,
                               }

    return half_layers_dic

def remove_quarters(layers_array, xyz_voxels):

    # get voxel dimensions
    x_voxels = xyz_voxels[0]
    y_voxels = xyz_voxels[1]
    z_voxels = xyz_voxels[2]

    # set empty arrays for the four quarters of the damage
    damage_1 = np.ones([x_voxels, y_voxels], dtype=int)
    damage_2 = np.ones([x_voxels, y_voxels], dtype=int)
    damage_3 = np.ones([x_voxels, y_voxels], dtype=int)
    damage_4 = np.ones([x_voxels, y_voxels], dtype=int)

    # set empty arrays for damage of the quarters
    damage_layer_1 = []
    damage_layer_2 = []
    damage_layer_3 = []
    damage_layer_4 = []
    damage_layer_5 = []
    damage_layer_6 = []
    damage_layer_7 = []
    damage_layer_8 = []
    damage_layer_9 = []
    damage_layer_10 = []
    damage_layer_11 = []
    damage_layer_12 = []

    # setting damaged areas
    for ii in range(y_voxels):
        if ii < y_voxels/2:
            for jj in range(x_voxels):
                if jj < x_voxels/2:
                    damage_1[ii, jj] = 0
                else:
                    damage_2[ii, jj] = 0
        else:
            for jj in range(x_voxels):
                if jj < x_voxels/2:
                    damage_3[ii, jj] = 0
                else:
                    damage_4[ii, jj] = 0

    if x_voxels%2 == 0 and y_voxels%2 == 0 and z_voxels%2 == 0:
        # Cube region with even lengths x_voxels, y_voxels and z_voxels
        for k in range(z_voxels):
            row = layers_array[k]

            if k < z_voxels/2:
                # convert to array
                row_array = np.array(np.array_split(row, x_voxels))

                # Apply damage to each row
                row_damage_1 = np.multiply(row_array, damage_1)
                row_damage_2 = np.multiply(row_array, damage_2)
                row_damage_3 = np.multiply(row_array, damage_3)
                row_damage_4 = np.multiply(row_array, damage_4)
                row_damage_5 = np.multiply(np.multiply(row_array, damage_1), damage_3)
                row_damage_6 = np.multiply(np.multiply(row_array, damage_2), damage_4)
                row_damage_9 = np.multiply(np.multiply(row_array, damage_1), damage_2)
                row_damage_10 = np.multiply(np.multiply(row_array, damage_3), damage_4)

                # Convert to list
                row_damg_1 = [elem for lst in row_damage_1.tolist() for elem in lst]
                row_damg_2 = [elem for lst in row_damage_2.tolist() for elem in lst]
                row_damg_3 = [elem for lst in row_damage_3.tolist() for elem in lst]
                row_damg_4 = [elem for lst in row_damage_4.tolist() for elem in lst]
                row_damg_5 = [elem for lst in row_damage_5.tolist() for elem in lst]
                row_damg_6 = [elem for lst in row_damage_6.tolist() for elem in lst]
                row_damg_9 = [elem for lst in row_damage_9.tolist() for elem in lst]
                row_damg_10 = [elem for lst in row_damage_10.tolist() for elem in lst]

                # Apply damage to layers
                damage_layer_1.append(row_damg_1)
                damage_layer_2.append(row_damg_2)
                damage_layer_3.append(row_damg_3)
                damage_layer_4.append(row_damg_4)
                damage_layer_5.append(row_damg_5)
                damage_layer_6.append(row_damg_6)
                damage_layer_9.append(row_damg_9)
                damage_layer_10.append(row_damg_10)
                # Append unaffected layers
                damage_layer_7.append(row.tolist())
                damage_layer_8.append(row.tolist())
                damage_layer_11.append(row.tolist())
                damage_layer_12.append(row.tolist())

            else:
                # convert to array
                row_array = np.array(np.array_split(row, x_voxels))

                # Apply damage to each row
                row_damage_1 = np.multiply(row_array, damage_1)
                row_damage_2 = np.multiply(row_array, damage_2)
                row_damage_3 = np.multiply(row_array, damage_3)
                row_damage_4 = np.multiply(row_array, damage_4)
                row_damage_7 = np.multiply(np.multiply(row_array, damage_1), damage_3)
                row_damage_8 = np.multiply(np.multiply(row_array, damage_2), damage_4)
                row_damage_11 = np.multiply(np.multiply(row_array, damage_1), damage_2)
                row_damage_12 = np.multiply(np.multiply(row_array, damage_3), damage_4)

                # Convert to list
                row_damg_1 = [elem for lst in row_damage_1.tolist() for elem in lst]
                row_damg_2 = [elem for lst in row_damage_2.tolist() for elem in lst]
                row_damg_3 = [elem for lst in row_damage_3.tolist() for elem in lst]
                row_damg_4 = [elem for lst in row_damage_4.tolist() for elem in lst]
                row_damg_7 = [elem for lst in row_damage_7.tolist() for elem in lst]
                row_damg_8 = [elem for lst in row_damage_8.tolist() for elem in lst]
                row_damg_11 = [elem for lst in row_damage_11.tolist() for elem in lst]
                row_damg_12 = [elem for lst in row_damage_12.tolist() for elem in lst]

                # Append unaffected layers
                damage_layer_5.append(row.tolist())
                damage_layer_6.append(row.tolist())
                damage_layer_9.append(row.tolist())
                damage_layer_10.append(row.tolist())
                # Apply damage to layers
                damage_layer_1.append(row_damg_1)
                damage_layer_2.append(row_damg_2)
                damage_layer_3.append(row_damg_3)
                damage_layer_4.append(row_damg_4)
                damage_layer_7.append(row_damg_7)
                damage_layer_8.append(row_damg_8)
                damage_layer_11.append(row_damg_11)
                damage_layer_12.append(row_damg_12)

    quarters_layers_dic = {"quarters_creature_1": damage_layer_1,
                  "quarters_creature_2": damage_layer_2,
                  "quarters_creature_3": damage_layer_3,
                  "quarters_creature_4": damage_layer_4,
                  "quarters_creature_5": damage_layer_5,
                  "quarters_creature_6": damage_layer_6,
                  "quarters_creature_7": damage_layer_7,
                  "quarters_creature_8": damage_layer_8,
                  "quarters_creature_9": damage_layer_9,
                  "quarters_creature_10": damage_layer_10,
                  "quarters_creature_11": damage_layer_11,
                  "quarters_creature_12": damage_layer_12
                           }

    return quarters_layers_dic
