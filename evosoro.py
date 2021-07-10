import json

import numpy as np


class Evosoro:
    def __init__(self):
        # Import material defaults form settings file
        settings_file = open("settings.json")
        self.settings = json.load(settings_file)
        settings_file.close()

        # Creature Structure
        self.structure = self.settings["structure"]["creature_structure"]

        # Initial morphology
        self.base_morphology = np.array([
            [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
            [3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3],
            [3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3],
            [3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3],
            [3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3],
            [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
        ])

        # As just initialized, base morphology is the creatures morphology
        self.morphology = self.base_morphology.reshape((self.settings["structure"]["creature_structure"][2],
                                                       self.settings["structure"]["creature_structure"][1],
                                                       self.settings["structure"]["creature_structure"][0]))

        # Creature material properties (Initially set to defaults), please refer to settings.json to see definitions
        self.vxa_file = None
        self.number_of_materials = self.settings["mat_defaults"]["number_of_materials"]
        self.integration = self.settings["mat_defaults"]["integration"]
        self.damping = self.settings["mat_defaults"]["damping"]
        self.collision = self.settings["mat_defaults"]["collision"]
        self.features = self.settings["mat_defaults"]["features"]
        self.stopConditions = self.settings["mat_defaults"]["stopConditions"]
        self.drawSmooth = self.settings["mat_defaults"]["drawSmooth"]
        self.write_fitness = self.settings["mat_defaults"]["write_fitness"]
        self.QhullTmpFile = self.settings["mat_defaults"]["QhullTmpFile"]
        self.CurvaturesTmpFile = self.settings["mat_defaults"]["CurvaturesTmpFile"]
        self.numFixed = self.settings["mat_defaults"]["numFixed"]
        self.numForced = self.settings["mat_defaults"]["numForced"]
        self.gravity = self.settings["mat_defaults"]["gravity"]
        self.thermal = self.settings["mat_defaults"]["thermal"]
        self.version = self.settings["mat_defaults"]["version"]
        self.lattice = self.settings["mat_defaults"]["lattice"]
        self.voxel = self.settings["mat_defaults"]["voxel"]
        self.mat_type = self.settings["mat_defaults"]["mat_type"]
        self.mat_colour = self.settings["mat_defaults"]["mat_colour"]
        self.mechanical_properties = self.settings["mat_defaults"]["mechanical_properties"]
        self.compression_type = self.settings["mat_defaults"]["compression_type"]
        self.phase_offset = self.settings["mat_defaults"]["phase_offset"]
        self.stiffness_array = self.settings["mat_defaults"]["stiffness_array"]
        self.gen_algorithm = None

    def update_vxa_file(self, creature, number_of_materials=None, integration=None, damping=None, collision=None,
                        features=None, stopConditions=None, drawSmooth=None, write_fitness=None, QhullTmpFile=None,
                        CurvaturesTmpFile=None, numFixed=None, numForced=None, gravity=None, thermal=None, version=None,
                        lattice=None, voxel=None, mat_type=None, mat_colour=None, mechanical_properties=None,
                        compression_type=None, phase_offset=None, stiffness_array=None):
        # Using creature material properties, creates readable VXA file for execution on Voxelyze. See settings.json
        # for more explanation on parameters.
        #
        # ARGUMENTS
        # - generation              int, new creature generation number
        # - episode                 int, new creature episode number
        # - number_of_materials     int, used to create a list of creature materials
        # - integration             (1, 2) list, used to set integration parameters in vxa file
        # - damping                 (1, 3) list, used to set damping parameters in vxa file
        # - collision               (1, 3) list, used to set collision parameters in vxa file
        # - features                (1, 3) list, used to set features parameters in vxa file
        # - stopConditions          (1, 3) list, used to set stopConditions parameters in vxa file
        # - drawSmooth              int, used to set drawSmooth parameter in vxa file
        # - write_fitness           int, used to set write_fitness parameter in vxa file
        # - QhullTmpFile            string, used to set QhullTmpFile parameter in vxa file
        # - CurvaturesTmpFile       string, used to set CurvaturesTmpFile parameter in vxa file
        # - numFixed                int, used to set numFixed parameter in vxa file
        # - numForced               int, used to set numForced parameter in vxa file
        # - gravity                 (1, 6) list, used to set gravity parameters in vxa file
        # - thermal                 (1, 5) list, used to set thermal parameters in vxa file
        # - version                 string, used to set version parameter in vxa file
        # - lattice                 (1, 8) list, used to set lattice parameters in vxa file
        # - voxel                   (1, 4) list, used to set voxel parameters in vxa file
        # - mat_type                int, used to set mat_type parameter in vxa file
        # - mat_colour              (1, number_of_materials) list, used to set mat_colour parameters in vxa file
        # - mechanical_properties   (1, 13) list, used to set mechanical_properties parameters in vxa file
        # - compression_type        string, used to set compression_type parameter in vxa file
        # - phase_offset            int, used to set phase_offset parameter in vxa file
        # - stiffness_array         np.ndarray, used to set stiffness_array in vxa file
        #
        # RETURNS
        # - vxa_file                string,stiffness_array containing vxa file

        # If property given change self.PROstiffness_arrayPERTY_NAME
        if number_of_materials is not None:
            self.number_of_materials = number_of_materials
        if integration is not None:
            self.integration = integration
        if damping is not None:
            self.damping = damping
        if collision is not None:
            self.collision = collision
        if features is not None:
            self.features = features
        if stopConditions is not None:
            self.stopConditions = stopConditions
        if drawSmooth is not None:
            self.drawSmooth = drawSmooth
        if write_fitness is not None:
            self.write_fitness = write_fitness
        if QhullTmpFile is not None:
            self.QhullTmpFile = QhullTmpFile
        if CurvaturesTmpFile is not None:
            self.CurvaturesTmpFile = CurvaturesTmpFile
        if numFixed is not None:
            self.numFixed = numFixed
        if numForced is not None:
            self.numForced = numForced
        if gravity is not None:
            self.gravity = thermal
        if version is not None:
            self.version = version
        if lattice is not None:
            self.lattice = lattice
        if voxel is not None:
            self.voxel = voxel
        if mat_type is not None:
            self.mat_type = mat_type
        if mat_colour is not None:
            self.mat_colour = mat_colour
        if mechanical_properties is not None:
            self.mechanical_properties = mechanical_properties
        if compression_type is not None:
            self.compression_type = compression_type
        if phase_offset is not None:
            self.phase_offset = phase_offset
        if stiffness_array is not None:
            self.stiffness_array = np.reshape(stiffness_array, (self.settings["structure"]["creature_structure"][2],
                                                                self.settings["structure"]["creature_structure"][1] *
                                                                self.settings["structure"]["creature_structure"][0]))

        # Set Genetic Algorithm variable
        self.gen_algorithm = [self.write_fitness, creature.fitness_file_name, self.QhullTmpFile, self.CurvaturesTmpFile]

        # set variables
        [integrator, dtFrac] = self.integration
        [bondDampingZ, colDampingZ, slowDampingZ] = self.damping
        [selfColEnabled, colSystem, collisionHorizon] = self.collision
        [fluidDampEnabled, poissonKickBackEnabled, enforceLatticeEnabled] = self.features
        [stopConditionType, stopConditionValue, InitCmTime] = self.stopConditions
        [WriteFitnessFile, FitnessFileName, QhullTmpFile, CurvaturesTmpFile] = self.gen_algorithm
        [gravEnabled, gravAcc, floorEnabled, sloped_floor, bump_size, bump_sep] = self.gravity
        [tempEnabled, tempAmp, tempBase, varyTempEnabled, tempPeriod] = self.thermal
        [lattice_dim, x_dim_adj, y_dim_adj, z_dim_adj, x_line_offset,
         y_line_offset, x_layer_offset, y_layer_offset] = self.lattice
        [vox_name, x_squeeze, y_squeeze, z_squeeze] = self.voxel
        [x_voxels, y_voxels, z_voxels] = self.structure

        init_text = '''<?xml version="1.0" encoding="ISO-8859-1"?>
        <VXA Version="1.0">
        <Simulator>
        <Integration>
        <Integrator>''' + str(integrator) + '''</Integrator>
        <DtFrac>''' + str(dtFrac) + '''</DtFrac>
        </Integration>
        <Damping>
        <BondDampingZ>''' + str(bondDampingZ) + '''</BondDampingZ>
        <ColDampingZ>''' + str(colDampingZ) + '''</ColDampingZ>
        <SlowDampingZ>''' + str(slowDampingZ) + '''</SlowDampingZ>
        </Damping>
        <Collisions>
        <SelfColEnabled>''' + str(selfColEnabled) + '''</SelfColEnabled>
        <ColSystem>''' + str(colSystem) + '''</ColSystem>
        <CollisionHorizon>''' + str(collisionHorizon) + '''</CollisionHorizon>
        </Collisions>
        <Features>
        <FluidDampEnabled>''' + str(fluidDampEnabled) + '''</FluidDampEnabled>
        <PoissonKickBackEnabled>''' + str(poissonKickBackEnabled) + '''</PoissonKickBackEnabled>
        <EnforceLatticeEnabled>''' + str(enforceLatticeEnabled) + '''</EnforceLatticeEnabled>
        </Features>
        <SurfMesh>
        <CMesh>
        <DrawSmooth>''' + str(self.drawSmooth) + '''</DrawSmooth>
        <Vertices/>
        <Facets/>
        <Lines/>
        </CMesh>
        </SurfMesh>
        <StopCondition>
        <StopConditionType>''' + str(stopConditionType) + '''</StopConditionType>
        <StopConditionValue>''' + str(stopConditionValue) + '''</StopConditionValue>
        <InitCmTime>''' + str(InitCmTime) + '''</InitCmTime>
        </StopCondition>
        <GA>
        <WriteFitnessFile>''' + str(WriteFitnessFile) +'''</WriteFitnessFile>
        <FitnessFileName>''' + str(FitnessFileName) +'''</FitnessFileName>
        <QhullTmpFile>''' + str(QhullTmpFile) +'''</QhullTmpFile>
        <CurvaturesTmpFile>''' + str(CurvaturesTmpFile) +'''</CurvaturesTmpFile>
        </GA>
        </Simulator>
        <Environment>
        <Fixed_Regions>
        <NumFixed>''' + str(self.numFixed) + '''</NumFixed>
        </Fixed_Regions>
        <Forced_Regions>
        <NumForced>''' + str(self.numForced) + '''</NumForced>
        </Forced_Regions>
        <Gravity>
        <GravEnabled>''' + str(gravEnabled) + '''</GravEnabled>
        <GravAcc>''' + str(gravAcc) + '''</GravAcc>
        <FloorEnabled>''' + str(floorEnabled) + '''</FloorEnabled>
        <FloorSlope>''' + str(sloped_floor) + '''</FloorSlope>
        <bump_size>''' + str(bump_size) + '''</bump_size>
        <bump_seperation>''' + str(bump_sep) + '''</bump_seperation>
        </Gravity>
        <Thermal>
        <TempEnabled>''' + str(tempEnabled) + '''</TempEnabled>
        <TempAmp>''' + str(tempAmp) + '''</TempAmp>
        <TempBase>''' + str(tempBase) + '''</TempBase>
        <VaryTempEnabled>''' + str(varyTempEnabled) + '''</VaryTempEnabled>
        <TempPeriod>''' + str(tempPeriod) + '''</TempPeriod>
        </Thermal>
        </Environment>
        <VXC Version="''' + str(self.version) + '''">
        <Lattice>
        <Lattice_Dim>''' + str(lattice_dim) + '''</Lattice_Dim>
        <X_Dim_Adj>''' + str(x_dim_adj) + '''</X_Dim_Adj>
        <Y_Dim_Adj>''' + str(y_dim_adj) + '''</Y_Dim_Adj>
        <Z_Dim_Adj>''' + str(z_dim_adj) + '''</Z_Dim_Adj>
        <X_Line_Offset>''' + str(x_line_offset) + '''</X_Line_Offset>
        <Y_Line_Offset>''' + str(y_line_offset) + '''</Y_Line_Offset>
        <X_Layer_Offset>''' + str(x_layer_offset) + '''</X_Layer_Offset>
        <Y_Layer_Offset>''' + str(y_layer_offset) + '''</Y_Layer_Offset>
        </Lattice>
        <Voxel>
        <Vox_Name>''' + str(vox_name) + '''</Vox_Name>
        <X_Squeeze>''' + str(x_squeeze) + '''</X_Squeeze>
        <Y_Squeeze>''' + str(y_squeeze) + '''</Y_Squeeze>
        <Z_Squeeze>''' + str(z_squeeze) + '''</Z_Squeeze>
        </Voxel>'''

        # Currently static!!!!!!
        voxel_text = '''
        <Palette>
        <Material ID="1">
        <MatType>0</MatType>
        <Name>Passive_Soft</Name>
        <Display>
        <Red>0</Red>
        <Green>1</Green>
        <Blue>1</Blue>
        <Alpha>1</Alpha>
        </Display>
        <Mechanical>
        <MatModel>0</MatModel>
        <Elastic_Mod>1000</Elastic_Mod>
        <Plastic_Mod>0</Plastic_Mod>
        <Yield_Stress>0</Yield_Stress>
        <FailModel>0</FailModel>
        <Fail_Stress>0</Fail_Stress>
        <Fail_Strain>0</Fail_Strain>
        <Density>1200.0</Density>
        <Poissons_Ratio>0.4</Poissons_Ratio>
        <CTE>0</CTE>
        <uStatic>1</uStatic>
        <uDynamic>0.5</uDynamic>
        </Mechanical>
        </Material>
        <Material ID="2">
        <MatType>0</MatType>
        <Name>Passive_Hard</Name>
        <Display>
        <Red>0</Red>
        <Green>0</Green>
        <Blue>1</Blue>
        <Alpha>1</Alpha>
        </Display>
        <Mechanical>
        <MatModel>0</MatModel>
        <Elastic_Mod>10000000</Elastic_Mod>
        <Plastic_Mod>0</Plastic_Mod>
        <Yield_Stress>0</Yield_Stress>
        <FailModel>0</FailModel>
        <Fail_Stress>0</Fail_Stress>
        <Fail_Strain>0</Fail_Strain>
        <Density>2200.0</Density>
        <Poissons_Ratio>0.4</Poissons_Ratio>
        <CTE>0</CTE>
        <uStatic>1</uStatic>
        <uDynamic>0.5</uDynamic>
        </Mechanical>
        </Material>
        <Material ID="3">
        <MatType>0</MatType>
        <Name>Active_+</Name>
        <Display>
        <Red>1</Red>
        <Green>0</Green>
        <Blue>0</Blue>
        <Alpha>1</Alpha>
        </Display>
        <Mechanical>
        <MatModel>0</MatModel>
        <Elastic_Mod>1.0e+006</Elastic_Mod>
        <Plastic_Mod>0</Plastic_Mod>
        <Yield_Stress>0</Yield_Stress>
        <FailModel>0</FailModel>
        <Fail_Stress>10</Fail_Stress>
        <Fail_Strain>0</Fail_Strain>
        <Density>1200.0</Density>
        <Poissons_Ratio>0.4</Poissons_Ratio>
        <CTE>0</CTE>
        <uStatic>1</uStatic>
        <uDynamic>0.5</uDynamic>
        </Mechanical>
        </Material>
        <Material ID="4">
        <MatType>0</MatType>
        <Name>Active_-</Name>
        <Display>
        <Red>0</Red>
        <Green>1</Green>
        <Blue>0</Blue>
        <Alpha>1</Alpha>
        </Display>
        <Mechanical>
        <MatModel>0</MatModel>
        <Elastic_Mod>1.0e+006</Elastic_Mod>
        <Plastic_Mod>0</Plastic_Mod>
        <Yield_Stress>0</Yield_Stress>
        <FailModel>0</FailModel>
        <Fail_Stress>0</Fail_Stress>
        <Fail_Strain>0</Fail_Strain>
        <Density>1200.0</Density>
        <Poissons_Ratio>0.4</Poissons_Ratio>
        <CTE>0.02</CTE>
        <uStatic>1</uStatic>
        <uDynamic>0.5</uDynamic>
        </Mechanical>
        </Material>
        <Material ID="5">
        <MatType>0</MatType>
        <Name>Aperture</Name>
        <Display>
        <Red>1</Red>
        <Green>0.784</Green>
        <Blue>0</Blue>
        <Alpha>1</Alpha>
        </Display>
        <Mechanical>
        <MatModel>0</MatModel>
        <Elastic_Mod>5e+007</Elastic_Mod>
        <Plastic_Mod>0</Plastic_Mod>
        <Yield_Stress>0</Yield_Stress>
        <FailModel>0</FailModel>
        <Fail_Stress>0</Fail_Stress>
        <Fail_Strain>0</Fail_Strain>
        <Density>1200.0</Density>
        <Poissons_Ratio>0.4</Poissons_Ratio>
        <CTE>-0.04</CTE>
        <uStatic>1</uStatic>
        <uDynamic>0.5</uDynamic>
        </Mechanical>
        </Material>'''

        morph_array = []
        for row in self.morphology.reshape((self.settings["structure"]["creature_structure"][2],
                                           self.settings["structure"]["creature_structure"][1] *
                                           self.settings["structure"]["creature_structure"][0])):
            temp_text = "".join([str(int(elem)) for elem in row])
            morph_array.append(temp_text)

        morph_text = "\n".join(['''        <Layer><![CDATA[''' + row + ''']]></Layer>''' for row in morph_array])

        structure_text = '''
        </Palette>
        <Structure Compression="''' + str(self.compression_type) + '''">
        <X_Voxels>''' + str(x_voxels) + '''</X_Voxels>
        <Y_Voxels>''' + str(y_voxels) + '''</Y_Voxels>
        <Z_Voxels>''' + str(z_voxels) + '''</Z_Voxels>
        <Data>\n'''
        end_structure = '''
        </Data>\n'''
        structure_text = structure_text + morph_text + end_structure

        offset_array = []
        offset_base = np.round([np.multiply(1, (-1) * yi * self.phase_offset) for yi in range(self.structure[0])],
                               decimals=1)
        offset_vec = offset_base
        for i in range(self.structure[1] - 1):
            offset_vec = np.concatenate((offset_vec, offset_base))

        offset_map = offset_vec
        for j in range(self.structure[2] - 1):
            offset_map = np.vstack((offset_map, offset_vec))

        for row in offset_map:
            temp_text = ",".join([str(elem) for elem in row])
            offset_array.append(temp_text)

        pre_text = "        <PhaseOffset>"
        offset_text = "\n".join(['''        <Layer><![CDATA[''' + row + ''']]></Layer>''' for row in offset_array])
        post_text = "        </PhaseOffset>\n"

        phase_offset_text = pre_text + "\n" + offset_text + "\n" + post_text

        stiff_array = []
        for row in self.stiffness_array:
            temp_text = ",".join([str(elem) for elem in row])
            stiff_array.append(temp_text)

        pre_text = '''        <Stiffness>
        <MinElasticMod>10000.0</MinElasticMod>
        <MaxElasticMod>1000000</MaxElasticMod>'''

        offset_text = "\n".join(['''        <Layer><![CDATA[''' + row + ''']]></Layer>''' for row in stiff_array])
        post_text = '''        </Stiffness>'''
        stiffness_text = pre_text + "\n" + offset_text + "\n" + post_text

        end_text = '''
        </Structure>
        </VXC>
        </VXA>'''

        self.vxa_file = init_text + voxel_text + structure_text + phase_offset_text + stiffness_text + end_text