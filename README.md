# Adapting morphology to cope with damage in soft robots.
Genetic Algorithm to evaluate the fitness of a creature walking on a flat plane. After initial eveluation, creatures are inflicted damaged where eights, quarters and halfs are removed from the creature. These damaged morphologies are then evaluated and ran again. 

# How to run
Download and install Evosoro.
`git clone https://github.com/skriegman/evosoro.git`
Please see https://github.com/skriegman/evosoro for further details on how to install.

Evosoro is a Python soft robot simulation library based on the Voxelyze physics engine. It provides a high-level interface for the dynamic simulation and automated design of soft multimaterial robots.

Evosoro was designed and developed by the Morphology, Evolution & Cognition Laboratory, University of Vermont. The library is built on top of the open source VoxCAD and the underlying voxel physics engine (Voxelyze) which were both developed by the Creative Machines Lab, Columbia University.

Open the `settings.json` file and ensure the `"evosoro_path"` parameter is correct.

While you have the settings file open you can edit any of the parameters you wish. These are set to defaults, if you have a parameter or material property that changes throughout your evolution, you can update it within the code itself.

Within the Creature class you will see the initial creature morphology, if you wish to you can change so now. If you change the creatures base morphology you MUST update the `"creature_structure"` parameter within the settings file. 

If you are happy with the code and settings, simply run the code.

`python initialize.py`

# How to retrieve results

A file called `generated_files` was created while running simulation. Within these you will see all the saved data for this simulation. 
