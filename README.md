# Adapting morphology to cope with damage in soft robots.
Genetic Algorithm to evaluate the fitness of a creature walking on a flat plane. After initial eveluation, creatures are inflicted damaged where eights, quarters and halfs are removed from the creature. These damaged morphologies are then evaluated aswell.

This simulations are run using evosoro. Evosoro is a Python soft robot simulation library based on the Voxelyze physics engine. It provides a high-level interface for the dynamic simulation and automated design of soft multimaterial robots. Evosoro was designed and developed by the Morphology, Evolution & Cognition Laboratory, University of Vermont. The library is built on top of the open source VoxCAD and the underlying voxel physics engine (Voxelyze) which were both developed by the Creative Machines Lab, Columbia University.


# How to run
Download and install Evosoro.
`git clone https://github.com/skriegman/evosoro.git`
Please see https://github.com/skriegman/evosoro for further details on how to install.

IMPORTANT: Within the `settings.json` file, ensure the `"evosoro_path"` root is correct.

While in the working directory run `python initialize.py`

# Retrieving Results

A folder called `generated_files` was created during the simulation. Within these you will see all the saved data for this simulation. 

# Settings.json
Sttings file, most variables have been placed into this settings folder.

Within the Creature class you will see the initial creatures morphology, this array is of shape (z, x*y) where (x, y, z) are the creatures structure. If you change the length of the creatures base morphology you MUST update the `"creature_structure"` parameter within the settings file. 

With the settings file open you can edit any of the parameters you wish. If you have a parameter or material property that changes throughout your evolution, you can update it within the code itself.

| Evosoro Path  |Type     |Description                                                                                                    |
| ------------- |---------|---------------------------------------------------------------------------------------------------------------|
| evosoro_path  | String  | Path to the `voxelyze` file. Used to run the vxa file of the created creature. Depending on inputs into the vxa file, creates files within the cwd                                which can ge used to evaluate the creatures fitness. E.g. `path/to/voxelize -f Example_1.vxa -p` |

***

| Parameters    |Type     |Description                                                                                                    |
| ------------- |---------|---------------------------------------------------------------------------------------------------------------|
| pop_size      | Int     | Population size. During each episode `pop_size` number of creatures are simulated and evaluated.              |
| ep_size       | Int     | Number of episodes. During each episode creature is evaluated and their morphology is changed using their ANN.|
| top           | Int     | Number of top performing creatures to keep after each generation.                                             |
| evolve        | Int     | Number of creatures to evolve their ANN.                                                                      |
| gen_size      | Int     | Number of generations. After each generation the `top` performing creatures from the evaluation are kept (ANN unchanged) and the following `evolve`                               number of performing creatures have their ANN randomly changed. The following `pop_size - top - evolve` number of creatures are created to always add                             different creatures into the system.|

***

| ANN Paramameters     |Type     |Description                                                                                                    |
| -------------------- |---------|---------------------------------------------------------------------------------------------------------------|
| num_inputs           | Int     | Number of nodes in input layer of the ANN.                                                                    |
| num_outputs          | Int     | Number of nodes in output layer of the ANN.                                                                   |
| num_hidden_layers    | Int     | Number of nodes in the hidden layer of the ANN. Currently ANN only has one hidden layer.                      |
| activation_function  | String  | Activation function to be used within hidden layers of the ANN. `"tanh"` for tanh AF and `"sigmoid"` for sigmoid. Output layer of ANN uses by default a sigmoid AF. |
| bounds               | List    | Upper and lower bounds for weights and biases, limitting them to this range.                                  |
| noise                | Float   | Percentage of desired noise when 'evolving' ANN.                                                              |

***

| Structure          |Type     |Description                                                                                                    |
| -------------------|---------|---------------------------------------------------------------------------------------------------------------|
| creature_structure | List    | Shape of creature (x, y, z). I.e. (6, 6, 6) would be a 6 by 6 by 6 creature.                                  |
| base_stiffness     | Int     | Baseline stiffness, used to initialize the creatures morphology.                                              |
| max_stiffness      | Int     | Maximun stiffness that the creature can evolve to, stiffness values that surpass this value are reset to this                                         value.                                                                                                     |
| morph_max          | Int     | Voxel material number to be asigned to voxels that equal or surpass max_stiffness.                            |
| min_stiffness      | Int     | Minimum stiffness that the creature can evolve to, stiffness values bellow this value are reset to this value.|
| morph_min          | Int     | Voxel material number to be asigned to voxels that equal or decrease bellow min_stiffness.                    |
| morph_between      | Int     | Voxel material number to be asigned to voxels that are between min_stiffness and max_stiffness.               |
| actuator_stiffness | Int     | Stiffness to be assigned to actuators.                                                                        |
| actuator_morph     | Int     | Voxel material number to be assigned to actuator voxel.                                                       |
| unchangeable_morphs| List    | List of material numbers whos voxel properties should not change throughout the evaluation.                   |

***

| mat_defaults       |Type     |Description                                                                                                    |
| -------------------|---------|---------------------------------------------------------------------------------------------------------------|
| number_of_materials| Int     | Number of different voxel materials, used top create each material.                                           |
| mat_colour         | List    | Shape (1, number_of_materials) creature.vxa material voxel colours. Must be same length as desired number of materials.  |
| integration        | List    | Shape (1, 2) creature.vxa integrator values. integaration[0] = Integrator and integration[1] = DtFrac.        |
| damping            | List    | Shape (1, 3) creature.vxa damping values. damping[0] = BondDampingZ, damping[1] = ColDampingZ and damping[2] = SlowDampingZ |
| collision          | List    | Shape (1, 3) creature.vxa collision values. collision[0] = SelfColEnabled. collision[1] = ColSystem and collision[2] = CollisionHorizon |
| features           | List    | Shape (1,3) creature.vxa features values. features[0] = FluidDampEnabled, features[1] = PoissonKickBackEnabled and features[2] = EnforceLatticeEnabled.|
| stopConditions     | List    | Shape (1, 3) creature.vxa stopConditions values. stopConditions[0] = stopConditionType, stopConditions[1] = stopConditionValue and stopConditions[2] = InitCmTime.|
| drawSmooth         | Int-bool| drawSmooth, 0 or 1.                                                                                           |
| wrtie_fitness      | Int-bool| writeFitness, 0 or 1.                                                                                         |      
| QhullTmpFile       | String  | Temp file name E.g. Qhull_temp0                                                                               |
| CurvaturesTmpFile  | String  | Temp file name E.g. curve_temp0                                                                               |
| numFixed           | Int     | numFixed                                                                                                      |
| numForced          | Int     | numForced                                                                                                     |
| gravity            | List    | Shape (1, 6) creature.vxa gravity values. gravity[0] = gravEnabled, gravity[1] = gravAcc, gravity[2] = floorEnabled, gravity[3] = sloped_floor, gravity[4] = floorEnabled and gravity[5] = bump_sep |
| thermal            | List    | Shape (1, 5) creature.vxa thermal values. thermal[0] = tempEnabled, thermal[1] = tempAmp, thermal[2] = tempBase, thermal[3] = varyTempEnabled and thermal[4] = tempPeriod |
| version            | Float   | version number                                                                                                |
| lattice            | List    | Shape (1, 8) creature.vxa lattice values. lattice[0] = lattice_dim, lattice[1] = x_dim_adj, lattice[2] = y_dim_adj, lattice[3] = z_dim_adj, lattice[4] = x_line_offset, lattice[5] = y_line_offset, lattice[6] = x_layer_offset and lattice[7] = y_layer_offset |
| voxel              | List     | Shape (1, 3) creature.vxa voxel values. voxel[0] = vox_name, voxel[1] = x_squeeze, voxel[2] = y_squeeze and voxel[3] = z_squeeze|
| mat_type           | Int      | Material type                                                                                                |  
| mechanical_properties  | List    | Shape (1, 13) creature.vxa mechanical_properties values. mechanical_properties[0] = mat_model, mechanical_properties[1] = elastic_mod, mechanical_properties[2] = plastic_mod, mechanical_properties[3] = yield_stress, mechanical_properties[4] = fail_model, mechanical_properties[5] = fail_stress, mechanical_properties[6] = fail_strain, mechanical_properties[7] = density, mechanical_properties[8] = poissons_ration, mechanical_properties[9] = CTE, mechanical_properties[10] = uStatic, mechanical_properties[11] = uDynamic and mechanical_properties[12] = isConductive |
| compression_type   | String  | Compression type, e.g. ASCII_READABLE                                                                         |
| phase_offset       | Float   | Phase offset magnitude, used to create phase offset array.                                                    |

| phase_offset       | Float   | Phase offset magnitude, used to create phase offset array.                                                    |





                                                  

# TODO

- Update Readme (functions)
- Recover parameter def from evosoro

