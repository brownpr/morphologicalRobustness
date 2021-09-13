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
A folder called `generated_files` was created during the simulation. Within these you will see all the saved data for this simulation. In the settings.json file, detailed bellow, you can enable or disable file saving. 

If you use the population.py class function pop_name.save_population() a pickle file will be saved as an instance of your population class. You can load this population at any time with loaded_pop = Population(..., load_pop=True). 

