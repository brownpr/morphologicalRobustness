import json


class Voxel:
    def __init__(self, coordinates, material_number):
        # import settings
        settings_file = open("settings.json")
        self.settings = json.load(settings_file)
        settings_file.close()

        if material_number in self.settings["structure"]["unchangeable_morphs"]:
            self.can_be_changed = False
        else:
            self.can_be_changed = True

        # Location parameters
        self.coordinates = coordinates                                          # list, z, y, x coordinates (yes, z,y,x)
        self.neighbours = []                                                    # list, of neighbours
        self.section = None                                                      # int, region in which the cube belongs

        # Material properties
        self.material_number = material_number                                  # int, material number of voxel
        self.stiffness = self.settings["structure"]["base_stiffness"]           # int, default stiffness

    def update_with_stiffness(self, new_stiffness):
        if self.can_be_changed:
            if new_stiffness > self.settings["structure"]["max_stiffness"]:
                self.stiffness = self.settings["structure"]["max_stiffness"]
                self.material_number = self.settings["structure"]["morph_max"]
            elif new_stiffness < self.settings["structure"]["min_stiffness"]:
                self.stiffness = self.settings["structure"]["min_stiffness"]
                self.material_number = self.settings["structure"]["morph_min"]
            else:
                self.stiffness = new_stiffness
                self.material_number = self.settings["structure"]["morph_between"]

    def remove(self):
        if self.can_be_changed:
            self.material_number = 0
            self.stiffness = self.settings["structure"]["min_stiffness"]
