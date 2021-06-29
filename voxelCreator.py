import sys
import numpy as np

from decimal import Decimal


# Set variables for simulator conditions
def init_creator(integration, damping, collision, features, stopConditions, drawSmooth, ga):

    # Set default Draw Smooth
    if drawSmooth == "":
        drawSmooth = 1

    # Error check for integration input
    integration_defaults = [0, 0.9]  # integration defaults
    if integration == "":
        integration_list = integration_defaults
    elif isinstance(integration, str):
        integration.replace(" ", "")  # remove whitespace
        integration_list = integration.split(',')
    else:
        integration_list = integration
    # Ensure correct length for parameters
    if len(integration_list) != 2:
        print("ERROR: Only expecting two variables for integration parameter and got "
              + str(len(integration_list)) + ".")
        sys.exit()
    # Set empty variables to defaults and error check parameter lists
    for ii in range(len(integration_list)):
        if integration_list[ii] == "":
            integration_list[ii] = integration_defaults[ii]
        # Ensure only floats are used
        try:
            float(integration_list[ii])
        except ValueError:
            print("ERROR: Value introduced in integration parameter " + str(ii) + "is incorrect, only use floats.")
            sys.exit()

    # Error check for damping input
    damping_defaults = [1, 0.8, 0.01]
    if damping == "":
        damping_list = damping_defaults
    elif isinstance(damping, str):
        damping.replace(" ", "")  # remove whitespace
        damping_list = damping.split(',')
    else:
        damping_list = damping
    # Ensure correct length for parameters
    if len(damping_list) != 3:
        print("ERROR: Only expecting three variables for damping parameter and got "
              + str(len(damping_list)) + ".")
        sys.exit()
    # Set empty variables to defaults and error check parameter lists
    for ii in range(len(damping_list)):
        if damping_list[ii] == "":
            damping_list[ii] = damping_defaults[ii]
        # Ensure only floats are used
        try:
            float(damping_list[ii])
        except ValueError:
            print("ERROR: Value introduced in damping parameter " + str(ii) + "is incorrect, only use floats.")
            sys.exit()

    # Error check for collision input
    collision_defaults = [1, 3, 2]
    if collision == "":
        collision_list = collision_defaults
    elif isinstance(collision, str):
        collision.replace(" ", "")  # remove whitespace
        collision_list = collision.split(',')
    else:
        collision_list = collision
    # Ensure correct length for parameters
    if len(collision_list) != 3:
        print("ERROR: Only expecting three variables for collision parameter and got "
              + str(len(collision_list)) + ".")
        sys.exit()
    # Set empty variables to defaults and error check parameter lists
    for ii in range(len(collision_list)):
        if collision_list[ii] == "":
            collision_list[ii] = collision_defaults[ii]
        # Ensure only floats are used
        try:
            float(collision_list[ii])
        except ValueError:
            print("ERROR: Value introduced in collision parameter " + str(ii) + " is incorrect, only use floats.")
            sys.exit()

    # Error check for features input
    features_defaults = [0, 0, 0]
    if features == "":
        features_list = features_defaults
    elif isinstance(features, str):
        features.replace(" ", "")  # remove whitespace
        features_list = features.split(',')
    else:
        features_list = features
    # Ensure correct length for parameters
    if len(features_list) != 3:
        print("ERROR: Only expecting three variables for features parameter and got "
              + str(len(features_list)) + ".")
        sys.exit()
    # Set empty variables to defaults and error check parameter lists
    for ii in range(len(features_list)):
        if features_list[ii] == "":
            features_list[ii] = features_defaults[ii]
        # Ensure only floats are used
        try:
            float(features_list[ii])
        except ValueError:
            print("ERROR: Value introduced in features parameter " + str(ii) + " is incorrect, only use floats.")
            sys.exit()

    # Error check for stopConditions input
    stopConditions_defaults = [2, 8, 1.0]
    if stopConditions == "":
        stopConditions_list = stopConditions_defaults
    elif isinstance(stopConditions, str):
        stopConditions.replace(" ", "")  # remove whitespace
        stopConditions_list = stopConditions.split(',')
    else:
        stopConditions_list = stopConditions
    # Ensure correct length for parameters
    if len(stopConditions_list) != 3:
        print("ERROR: Only expecting two variables for stopConditions parameter and got "
              + str(len(collision_list)) + ".")
        sys.exit()
    # Set empty variables to defaults and error check parameter lists
    for ii in range(len(stopConditions_list)):
        if stopConditions_list[ii] == "":
            stopConditions_list[ii] = stopConditions_defaults[ii]
        # Ensure only floats are used
        try:
            float(stopConditions_list[ii])
        except ValueError:
            print("ERROR: Value introduced in stopConditions parameter " + str(
                ii) + " is incorrect, only use floats.")
            sys.exit()

    # # Error check for simStopCon input
    # simStopCon_defaults = [1.0, 1000000000]
    # if simStopCon == "":
    #     simStopCon_list = simStopCon_defaults
    # elif isinstance(simStopCon, str):
    #     simStopCon.replace(" ", "")  # remove whitespace
    #     simStopCon_list = simStopCon.split(',')
    # else:
    #     simStopCon_list = simStopCon
    # # Ensure correct length for parameters
    # if len(simStopCon_list) != 2:
    #     print("ERROR: Only expecting two variables for simStopCon parameter and got "
    #           + str(len(simStopCon_list)) + ".")
    #     sys.exit()
    # # Set empty variables to defaults and error check parameter lists
    # for ii in range(len(simStopCon_list)):
    #     if simStopCon_list[ii] == "":
    #         simStopCon_list[ii] = simStopCon_defaults[ii]
    #     # Ensure only floats are used
    #     try:
    #         float(simStopCon_list[ii])
    #     except ValueError:
    #         print("ERROR: Value introduced in simStopCon parameter " + str(ii) + " is incorrect, only use floats.")
    #         sys.exit()

    ga_defaults = [1, "temp_name", "Qhull_temp0", "curve_temp0"]
    if ga == "":
        ga = ga_defaults
    else:
        if ga[0] == "":
            ga[0] = ga_defaults[0]
        if ga[1] == "":
            print("ERROR: Fitness file name cannot be empty string.")
            sys.exit()
        if ga[2] == "":
            ga[2] = ga_defaults[2]
        if ga[3] == "":
            ga[3] = ga_defaults[3]

    ga_list = ga

    # set variables
    [integrator, dtFrac] = integration_list
    [bondDampingZ, colDampingZ, slowDampingZ] = damping_list
    [selfColEnabled, colSystem, collisionHorizon] = collision_list
    [fluidDampEnabled, poissonKickBackEnabled, enforceLatticeEnabled] = features_list
    [stopConditionType, stopConditionValue, InitCmTime] = stopConditions_list
    [WriteFitnessFile, FitnessFileName, QhullTmpFile, CurvaturesTmpFile] = ga_list


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
    <DrawSmooth>''' + str(drawSmooth) + '''</DrawSmooth>
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
    </Simulator>'''

    return init_text


def environment_creator(numFixed, numForced, gravity, thermal):
    # error check numFixed and set default
    if numFixed == "":
        numFixed = 0
    elif int(numFixed) < 0 or int(numFixed) >= 1:
        print("ERROR: numFixed must be a value between 0 and 1")
        sys.exit()

    # Set default numForced
    if numForced == "":
        numForced = 0

    # Error checking gravity inputs
    gravity_defaults = [1, -9.81, 1, 0, "bump1", "bump2"]
    if gravity == "":
        gravity_list = gravity_defaults
    elif isinstance(gravity, str):
        gravity.replace(" ", "")  # remove whitespace
        gravity_list = gravity.split(',')
    else:
        gravity_list = gravity
    # Ensure correct parameter length
    if len(gravity_list) != 6:
        print("ERROR: Only expecting six variables for gravity parameter and got " + str(len(gravity_list)) + ".")
        sys.exit()

    # Error checking thermal inputs
    thermal_defaults = [1, 39, 25, 1, 0.25]
    if thermal == "":
        thermal_list = thermal_defaults
    elif isinstance(thermal, str):
        thermal.replace(" ", "")  # remove whitespace
        thermal_list = thermal.split(',')
    else:
        thermal_list = thermal
    # Ensure correct parameter length
    if len(thermal_list) != 5:
        print("ERROR: Only expecting five variables for thermal parameter and got " + str(len(thermal_list)) + ".")
        sys.exit()
    # Set empty variables to defaults and error check parameter lists
    for ii in range(len(thermal_list)):
        if thermal_list[ii] == "":
            thermal_list[ii] = thermal_defaults[ii]
        # Ensure only floats are used
        try:
            float(thermal_list[ii])
        except ValueError:
            print("ERROR: Value introduced in thermal parameter " + str(ii) + "is incorrect, only use floats.")
            sys.exit()

    # Set variables
    [gravEnabled, gravAcc, floorEnabled, sloped_floor, bump_size, bump_sep] = gravity_list
    [tempEnabled, tempAmp, tempBase, varyTempEnabled,tempPeriod] = thermal_list

    environment_text = '''
    <Environment>
    <Fixed_Regions>
    <NumFixed>''' + str(numFixed) + '''</NumFixed>
    </Fixed_Regions>
    <Forced_Regions>
    <NumForced>''' + str(numForced) + '''</NumForced>
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
    </Environment>'''

    return environment_text


def vxc_creator(version, lattice, voxel):
    # set default version number
    if version == "":
        version = "0.93"

    # Error checking lattice inputs
    lattice_defaults = [0.05, 1, 1, 1, 0, 0, 0, 0]
    if lattice == "":
        lattice_list = lattice_defaults
    elif isinstance(lattice, str):
        lattice.replace(" ", "")  # remove whitespace
        lattice_list = lattice.split(',')
    else:
        lattice_list = lattice
    # Ensure correct parameter length
    if int(len(lattice_list)) != 8:
        print("ERROR: Only expecting eight variables for lattice parameter and got " + str(len(lattice_list)) + ".")
        sys.exit()
    # Set empty variables to defaults and error check parameter lists
    for ii in range(len(lattice_list)):
        if lattice_list[ii] == "":
            lattice_list[ii] = lattice_defaults[ii]
        # Ensure only floats are used
        try:
            float(lattice_list[ii])
        except ValueError:
            print("ERROR: Value introduced in lattice parameter " + str(ii) + "is incorrect, only use floats.")
            sys.exit()

    # Error checking voxel inputs
    voxel_defaults = ["BOX", 1, 1, 1]
    if voxel == "":
        voxel_list = voxel_defaults
    elif isinstance(voxel, str):
        voxel.replace(" ", "")  # remove whitespace
        voxel_list = voxel.split(',')
    else:
        voxel_list = voxel
    # Ensure correct parameter length
    if int(len(voxel_list)) != 4:
        print("ERROR: Only expecting four variables for voxel parameter and got " + str(len(voxel_list)) + ".")
        sys.exit()
    # Set empty variables to defaults and error check parameter lists
    for ii in range(len(voxel_list)):
        if voxel_list[ii] == "":
            voxel_list[ii] = voxel_defaults[ii]
        # Ensure only floats are used
        if ii > 0:
            try:
                float(voxel_list[ii])
            except ValueError:
                print("ERROR: Value introduced in voxel parameter " + str(ii) + " is incorrect, only use floats.")
                sys.exit()

    # set variables
    [lattice_dim, x_dim_adj, y_dim_adj, z_dim_adj, x_line_offset,
     y_line_offset, x_layer_offset, y_layer_offset] = lattice_list
    [vox_name, x_squeeze, y_squeeze, z_squeeze] = voxel_list

    vcx_init_text = '''
    <VXC Version="''' + str(version) + '''">
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

    return vcx_init_text


def voxel_creator(mat_ID, mat_type, mat_name, mat_colour, mechanical_properties):
    # mat_ID must have input and cannot equal 0
    if mat_ID == "":
        print("ERROR: Must input mat_ID.")
        sys.exit()
    elif float(mat_ID) == 0:
        print("ERROR: mat_ID cannot equal 0 as this is the default for an empty voxel.")
        sys.exit()

    # Default mat_type
    if mat_type == "":
        mat_type = 0

    # Default naming based on mat_ID
    if mat_name == "":
        mat_name = "default_" + str(mat_ID)  # default material naming

    # default colour list: blue,green,red,yellow,cian,white,black default colours
    mat_colour_list = ["0,0,1,1", "0,1,0,1", "1,0,0,1", "1,1,0,1", "0,1,1,1", "1,1,1,1", "0,0,0,1"]
    # if mat_colour == "" and int(mat_ID) >= 8:
    #     # print error if material ID exceeds 8
    #     print("WARNING: No material colour chosen and material ID number "
    #           "exceeds available default colours. A random colour will be set with alpha as 1.")
    #     mat_colour_list = [np.rand([0, 1]), np.rand([0, 1]), np.rand([0, 1]), 1]
    # elif mat_colour == "":
    #     mat_colour_list = default_colours[(int(mat_ID) - 1)]  # select default colour from mat_ID number
    #     mat_colour_list = mat_colour_list.split(',')
    # elif mat_colour == ("rand" or "Rand" or "RAND"):
    #     # random colour chosen
    #     mat_colour_list = [np.rand([0, 1]), np.rand([0, 1]), np.rand([0, 1]), np.rand([0, 1])]
    # elif isinstance(mat_colour, str):
    #     mat_colour.replace(" ", "")  # remove whitespace
    #     mat_colour_list = mat_colour.split(',')
    # else:
    #     mat_colour_list = mat_colour
    # # Set empty variables and error check parameter lists
    # for ii in range(len(mat_colour_list)):
    #     if mat_colour_list[ii] == "":
    #         # set default red
    #         print("WARNING: Colour parameter value at " + str(ii) + " is empty, this will be set as a randomly.")
    #         mat_colour_list[ii] = np.rand([0, 1])
    #     # Ensure only floats are used
    #     try:
    #         float(mat_colour_list[ii])
    #     except ValueError:
    #         print("ERROR: Value introduced in colour parameter " + str(ii) + " is incorrect, only use floats.")
    #         sys.exit()
    #
    #     # material colours should be between 0.00 and 1.00
    #     if int(mat_colour_list[ii]) < 0.01 or int(mat_colour_list[ii]) > 1.00:
    #         if int(mat_colour_list[ii]) != 0:
    #             print("ERROR: you need to choose a material colour value between 0.00 and 1.00.")
    #             sys.exit()

    # Error check material properties and set defaults
    mechanical_properties_defaults = [0, 1e+007, 0, 0, 0, 0, 0, 1e+006, 0.35, 0.00, 1, 0.5, 1]
    if mechanical_properties == "":
        mechanical_properties_list = mechanical_properties_defaults
    elif isinstance(mechanical_properties, str):
        mechanical_properties.replace(" ", "")  # remove whitespace
        mechanical_properties_list = mechanical_properties.split(',')
    else:
        mechanical_properties_list = mechanical_properties
    # Ensure correct length of parameters
    if len(mechanical_properties_list) != 13:
        print("ERROR: Only expecting thirteen variables for mechanical_properties parameter and got "
              + str(len(mechanical_properties_list)) + ".")
        sys.exit()
    # if len(mat_colour_list) != 4:
    #     print("ERROR: Only expecting four variables for mat_colours and got "
    #           + str(len(mechanical_properties_list)) + ".")
    #     sys.exit()
    # Set empty variables to defaults and error check parameter lists
    for ii in range(len(mechanical_properties_list)):        
        if mechanical_properties_list[ii] == "":
            mechanical_properties_list[ii] = mechanical_properties_defaults[ii]
        # Ensure only floats are used
        try:
            float(mechanical_properties_list[ii])
        except ValueError:
            print("ERROR: Value introduced in mechanical_properties parameter "
                  + str(ii) + "is incorrect, only use floats.")
            sys.exit()

    # set variables
    [mat_model, elastic_mod, plastic_mod, yield_stress, fail_model, fail_stress, fail_strain, density,
     poissons_ration, CTE, uStatic, uDynamic, isConductive] = mechanical_properties_list
    # [red, green, blue, alpha] = mat_colour_list

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
    # voxel_text = '''
    # <Palette>
    # <Material ID="''' + str(mat_ID) + '''">
    # <MatType>''' + str(mat_type) + '''</MatType>
    # <Name>''' + mat_name + '''</Name>
    # <Display>
    # <Red>''' + str(red) + '''</Red>
    # <Green>''' + str(green) + '''</Green>
    # <Blue>''' + str(blue) + '''</Blue>
    # <Alpha>''' + str(alpha) + '''</Alpha>
    # </Display>
    # <Mechanical>
    # <MatModel>''' + str(mat_model) + '''</MatModel>
    # <Elastic_Mod>''' + '%.2e' % Decimal(str(elastic_mod)) + '''</Elastic_Mod>
    # <Plastic_Mod>''' + str(plastic_mod) + '''</Plastic_Mod>
    # <Yield_Stress>''' + str(yield_stress) + '''</Yield_Stress>
    # <FailModel>''' + str(fail_model) + '''</FailModel>
    # <Fail_Stress>''' + str(fail_stress) + '''</Fail_Stress>
    # <Fail_Strain>''' + str(fail_strain) + '''</Fail_Strain>
    # <Density>''' + '%.2e' % Decimal(str(density)) + '''</Density>
    # <Poissons_Ratio>''' + str(poissons_ration) + '''</Poissons_Ratio>
    # <CTE>''' + str(CTE) + '''</CTE>
    # <uStatic>''' + str(uStatic) + '''</uStatic>
    # <uDynamic>''' + str(uDynamic) + '''</uDynamic>
    # <IsConductive>''' + str(isConductive) + '''</IsConductive>
    # </Mechanical>
    # </Material>'''


    return voxel_text


def layers_creator(layers_array, xyz_voxels):
    # cant have empty array
    if len(layers_array) == 0 or layers_array is None:
        print("ERROR layer creator array cannot be empty.")
        sys.exit()

    # Layers cannot be empty string
    if not isinstance(layers_array, np.ndarray):
        print("ERROR: Layers must be an array.")
        sys.exit()

    text_array = []

    for row in layers_array:
        temp_text = "".join([str(int(elem)) for elem in row])
        text_array.append(temp_text)

    layers_text = "\n".join(['''    <Layer><![CDATA[''' + row + ''']]></Layer>''' for row in text_array])
    return layers_text


def structure_creator(compression_type, xyz_voxels, layers_array):
    # Set default compression_type
    if compression_type == "":
        compression_type = "ASCII_READABLE"

    # Error checking vxy_voxels
    if isinstance(xyz_voxels, str):
        xyz_voxels.replace(" ", "")  # remove whitespace
        xyz_voxels_list = xyz_voxels.split(',')
    else:
        xyz_voxels_list = xyz_voxels
    # Ensure correct length of parameters
    if len(xyz_voxels_list) != 3:
        print("ERROR: Only expecting three variables for xyz_voxel parameter and got "
              + str(len(xyz_voxels_list)) + ".")
        sys.exit()
    # Set empty variables to defaults and error check parameter lists
    for ii in range(len(xyz_voxels_list)):
        if xyz_voxels_list[ii] == "":
            xyz_voxels_list[ii] = xyz_voxels_defaults[ii]
        # Ensure only floats are used
        try:
            float(xyz_voxels_list[ii])
        except ValueError:
            print("ERROR: Value introduced in xyz_voxels parameter "
                  + str(ii) + "is incorrect, only use floats.")
            sys.exit()

    # Convert layers_array into layers text
    layers = layers_creator(layers_array, xyz_voxels)

    # set variables
    [x_voxels, y_voxels, z_voxels] = xyz_voxels_list

    structure_text = '''
    </Palette>
    <Structure Compression="''' + str(compression_type) + '''">
    <X_Voxels>''' + str(x_voxels) + '''</X_Voxels>
    <Y_Voxels>''' + str(y_voxels) + '''</Y_Voxels>
    <Z_Voxels>''' + str(z_voxels) + '''</Z_Voxels>
    <Data>\n'''

    new_line = '''
    </Data>\n'''
    structure_text = structure_text + layers + new_line

    return structure_text


def offset_creator(xyz_voxels, phase_offset_mag):
    # Error checking vxy_voxels
    if isinstance(xyz_voxels, str):
        xyz_voxels.replace(" ", "")  # remove whitespace
        xyz_voxels_list = xyz_voxels.split(',')
    else:
        xyz_voxels_list = xyz_voxels

    if phase_offset_mag is None:
        phase_str = ""
        return phase_str
    elif isinstance(phase_offset_mag, str):
        phase_offset_mag = float(phase_offset_mag)

    text_array = []

    offset_base = np.round([np.multiply(1, (-1)*yi*phase_offset_mag) for yi in range(xyz_voxels_list[0])], decimals=1)
    offset_vec = offset_base
    for i in range(xyz_voxels_list[1]-1):
        offset_vec = np.concatenate((offset_vec, offset_base))

    offset_map = offset_vec
    for j in range(xyz_voxels_list[2] - 1):
        offset_map = np.vstack((offset_map, offset_vec))

    # for i in range(0, xyz_voxels_list[2]):
    #     offset_vec = []
    #     offset_vec = np.round(np.concatenate(np.vstack([np.multiply(np.ones((1, xyz_voxels_list[0])), (-1)*yi*phase_offset_mag)
    #                                                     for yi in range(xyz_voxels_list[1])])), decimals=1)
    #     offset_map.append(offset_vec)

    for row in offset_map:
        temp_text = ",".join([str(elem) for elem in row])
        text_array.append(temp_text)

    pre_text = "    <PhaseOffset>"
    offset_text = "\n".join(['''    <Layer><![CDATA[''' + row + ''']]></Layer>''' for row in text_array])
    post_text = "    </PhaseOffset>\n"

    phase_offset_text = pre_text + "\n" + offset_text + "\n" + post_text
    return phase_offset_text


def stiffness_creator(stiffness_array):
    # stiffness array is an array
    if stiffness_array is None:
        stiff_string = ""
        return stiff_string
    elif not isinstance(stiffness_array, np.ndarray):
        print("ERROR Stiffness must be an np array.")
        sys.exit()

    text_array = []
    for row in stiffness_array:
        temp_text = ",".join([str(elem) for elem in row])
        text_array.append(temp_text)

    pre_text = '''    <Stiffness>
    <MinElasticMod>1000.0</MinElasticMod>
    <MaxElasticMod>1000000</MaxElasticMod>'''

    offset_text = "\n".join(['''    <Layer><![CDATA[''' + row + ''']]></Layer>''' for row in text_array])
    post_text ='''    </Stiffness>'''

    phase_offset_text = pre_text + "\n" + offset_text + "\n" + post_text
    return phase_offset_text


def end_creator():
    end_text = '''
    </Structure>
    </VXC>
    </VXA>'''

    return end_text


if __name__ == '__main__':
    pass

    # a = init_creator("name", "", "", "", "", "", "", "")
    # b = environment_creator("", "", "", "")
    # c = vxc_creator("", "", "")
    # d = voxel_creator(1, "", "", "", "")
    # e = structure_creator("", [3, 1, 3], np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]))
    # f = offset_creator([10, 10, 10], 0.1)
    # g = stiffness_creator(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]))
    # h = end_creator()
    # all_text = a + b + c + d + e + f + g + h
    # gh = 1
