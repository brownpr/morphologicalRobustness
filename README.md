# Adapting morphology to cope with damage in soft robots.
Genetic Algorithm to evaluate the fitness of a creature walking on a flat plane. After initial eveluation, creatures are inflicted damaged where eights, quarters and halfs are removed from the creature. These damaged morphologies are then evaluated aswell.

This simulations are run using evosoro. Evosoro is a Python soft robot simulation library based on the Voxelyze physics engine. It provides a high-level interface for the dynamic simulation and automated design of soft multimaterial robots. Evosoro was designed and developed by the Morphology, Evolution & Cognition Laboratory, University of Vermont. The library is built on top of the open source VoxCAD and the underlying voxel physics engine (Voxelyze) which were both developed by the Creative Machines Lab, Columbia University.

# Requirements
- python2.7
- numpy
- matplotlib
- evosoro

# How to run
Download and install Evosoro.
`git clone https://github.com/skriegman/evosoro.git`
Please see https://github.com/skriegman/evosoro for further details on how to install.

IMPORTANT: Within the `settings.json` file, ensure the `"evosoro_path"` root is correct.

While in the working directory run `python initialize.py`.

# Retrieving Results

A folder called `generated_files` was created during the simulation. Within these you will see all the saved data for this simulation. 

# Settings.json
Settings file, most variables have been placed into this settings folder for ease of access and adaptability.

Within the Creature class you will see the initial creatures morphology, this array is of shape (z, x*y) where (x, y, z) are the creatures structure. If you change the length of the creatures base morphology you MUST update the `"creature_structure"` parameter within the settings file. 

With the settings file open you can edit any of the parameters you wish. If you have a parameter or material property that changes throughout your evolution, you can update it within the code itself.

| Evosoro Path  |Type     |Description                                                                                                    |
| ------------- |---------|---------------------------------------------------------------------------------------------------------------|
| evosoro_path  | String  | Path to the `voxelyze` file. Used to run the vxa file of the created creature. Depending on inputs into the vxa file, creates files within the cwd which can ge used to evaluate the creatures fitness. E.g. `path/to/voxelize -f Example_1.vxa -p` |

***

| Parameters    |Type     |Description                                                                                                    |
| ------------- |---------|---------------------------------------------------------------------------------------------------------------|
| pop_size      | Int     | Population size. During each episode `pop_size` number of creatures are simulated and evaluated.              |
| ep_size       | Int     | Number of episodes. During each episode creature is evaluated and their morphology is changed using their ANN.|
| top           | Int     | Number of top performing creatures to keep after each generation.                                             |
| evolve        | Int     | Number of creatures to evolve their ANN.                                                                      |
| gen_size      | Int     | Number of generations. After each generation the `top` performing creatures from the evaluation are kept (ANN unchanged) and the following `evolve` number of performing creatures have their ANN randomly changed. The following `pop_size - top - evolve` number of creatures are created to always add different creatures into the system.|
| stiff_delta_mult | Float   | Multiplier to stiffness delta, used to fine-tune stiffness change throughout episodes. |
| keep_files | bool   | Keep larger files? Files are: Pressure, Kinetic Energy and Strain CSV files. |

***

| ANN Paramameters     |Type     |Description                                                                                                    |
| -------------------- |---------|---------------------------------------------------------------------------------------------------------------|
| num_inputs           | Int     | Number of nodes in input layer of the ANN.                                                                    |
| num_outputs          | Int     | Number of nodes in output layer of the ANN.                                                                   |
| num_hidden_layers    | Int     | Number of nodes in the hidden layer of the ANN. Currently ANN only has one hidden layer.                      |
| activation_function  | String  | Activation function to be used within hidden layers of the ANN. `"tanh"` for tanh AF and `"sigmoid"` for sigmoid. Output layer of ANN uses by default a sigmoid AF. |
| bounds               | List    | Upper and lower bounds for weights and biases, limitting them to this range.                                  |
| parameter_change | Float   | Percentage of desired ANN parameter (weights, biases) change desired if creature is chosen to evolve. |
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
| integration        | List    | Shape (1, 2) creature.vxa integrator values. `Integrator` and `DtFrac`.        |
| damping            | List    | Shape (1, 3) creature.vxa damping values. `BondDampingZ`, `ColDampingZ` and `SlowDampingZ` |
| collision          | List    | Shape (1, 3) creature.vxa collision values. `SelfColEnabled`, `ColSystem` and `CollisionHorizon` |
| features           | List    | Shape (1,3) creature.vxa features values. `FluidDampEnabled`, `PoissonKickBackEnabled` and `EnforceLatticeEnabled`.|
| stopConditions     | List    | Shape (1, 3) creature.vxa `stopConditions` values. `stopConditionType`, `stopConditionValue` and `InitCmTime`.|
| drawSmooth         | Int-bool| `drawSmooth`, 0 or 1.                                                                                           |
| wrtie_fitness      | Int-bool| `writeFitness`, 0 or 1.                                                                                         |      
| QhullTmpFile       | String  | Temp file name E.g. `Qhull_temp0`                                                                               |
| CurvaturesTmpFile  | String  | Temp file name E.g. `curve_temp0`                                                                               |
| numFixed           | Int     | `numFixed`                                                                                                      |
| numForced          | Int     | `numForced`                                                                                                     |
| gravity            | List    | Shape (1, 6) creature.vxa gravity values. `gravEnabled`, `gravAcc`, `floorEnabled`, `sloped_floor`, `floorEnabled` and `bump_sep` |
| thermal            | List    | Shape (1, 5) creature.vxa thermal values. `tempEnabled`, `tempAmp`, `tempBase`, `varyTempEnabled` and `tempPeriod` |
| version            | Float   | version number                                                                                                |
| lattice            | List    | Shape (1, 8) creature.vxa lattice values. `lattice_dim`, `x_dim_adj`, `y_dim_adj`, `z_dim_adj`, `x_line_offset`, `y_line_offset`, `x_layer_offset` and `y_layer_offset` |
| voxel              | List     | Shape (1, 3) creature.vxa voxel values. `vox_name`, `x_squeeze`, `y_squeeze` and `z_squeeze`|
| mat_type           | Int      | Material type                                                                                                |  
| mechanical_properties  | List    | Shape (1, 13) creature.vxa mechanical_properties values. `mat_model`, `elastic_mod`, `plastic_mod`, `yield_stress`, `fail_model`, `fail_stress`, `fail_strain`, `density`, `poissons_ration`, `CTE`, `uStatic`, `uDynamic` and `isConductive` |
| compression_type   | String  | Compression type, e.g. `"ASCII_READABLE"`                                                                         |
| phase_offset       | Float   | Phase offset magnitude, used to create phase offset array.                                                    |

| phase_offset       | Float   | Phase offset magnitude, used to create phase offset array.                                                    |





                                                  

# TODO

- Update Readme (functions)
- Update code comments 
- Recover parameter def from evosoro

