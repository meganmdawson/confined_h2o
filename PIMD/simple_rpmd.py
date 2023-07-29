from __future__ import print_function
from openmm_comparison.Snapshot import Snapshot
from openmm import app
import openmm as mm
from openmm import unit
import h5py
import numpy as np
import os

""" This version of rigid.py has been modified to calculate the rotational
correlation function along the x, y and z axis """

"""
# Outputs:0
    CtauP,T contains four columns; the first contains imaginary time, the second
    z-correlation, then y-correlation and finally z-correlation
    
    xC{P}_{T}, yC{P}_{T} and zC{P}_{T} each contain six columns; the first 
    indicates the bead number (recall that indexing begins at zero), and the 
    second is the imaginary time of observation. The following three columns 
    denote the (non-normalized) coordinates of the body-fixed axes (x, y, or z 
    depending on the xC, yC or zC file), while the fourth column contains the 
    magnitude of the vector prior to normalization
    These files exist to describe the motion of beads in the simulation. This
    enables calculation of various values of interest, such as the rotational 
    correlation function or standard error.
    
    If the option to calculate various values within the pimd function is
    turned off or reanalysis of a simulation is desired, the xC{P}_{T} files 
    can be input to the functions in stat_fns, yielding values like deviation. 

    Plots of output CtauP,T can be generated using anim_rot.py
"""
def pimd(period, skipSteps, P, Temperature, dt, folder, gamma_denom = 1.,
         constraintTol = None):

    steps = np.floor((period * 1000) / (dt/unit.femtosecond))
    steps = int(steps)

    print("steps = ", steps)

    print("starting: T{}, P{}, L{}, dt{}, 1/g{}".format(Temperature, P, period,
                                                        dt, gamma_denom))

    defaultPath = os.path.realpath(folder)
    
    #check if folder exists
    if os.path.exists(os.path.realpath(folder)) == False:
        os.mkdir(defaultPath)
           
    Nwater=1
    nAtoms=3
    
    system = mm.System()
    pdb = app.PDBFile("openmm_comparison/monomer.pdb")
    forcefield = app.ForceField("tip4pfb.xml")
    # forcefield = app.ForceField("tip3p.xml")
    
    nonbonded = app.NoCutoff

    system = forcefield.createSystem(pdb.topology, nonbondedMethod=nonbonded, 
                                     nonbondedCutoff=1e3*unit.nanometer,
                                     constraints=None,rigidWater=False)
    
    # friction = 1.0/unit.picoseconds
    gamma_zero = 1.0/(gamma_denom* unit.picosecond)
    
    integrator = mm.RPMDIntegrator(P, Temperature*unit.kelvin,
                                   gamma_zero, dt)
    
    if constraintTol != None:
        integrator.setConstraintTolerance(constraintTol)

    platform = mm.Platform.getPlatformByName('CUDA')
    properties = {'CudaPrecision': 'mixed'}
    simulation = app.Simulation(pdb.topology, system, integrator, platform,
                                properties)
    simulation.context.setPositions(pdb.positions)
    simulation.context.computeVirtualSites()
    	
    simulation.context.setVelocitiesToTemperature(Temperature*unit.kelvin)
    
    state = simulation.context.getState(getForces=True, getEnergy=True, getPositions=True)

    beadTraj = Snapshot('bead_traj.xyz',simulation)

    # h5 to store information about the positions of each individual atom 
    out_h5 = h5py.File(os.path.join(defaultPath, 
                       "dimer_{P}_{T}.h5".format(P=P,T=Temperature)), 'w')
    h5coords=out_h5.create_dataset("coordinates",(int(steps/skipSteps)+1,P,
                                    nAtoms,3), dtype = "f")
    h5kEnergy=out_h5.create_dataset("kineticEnergy",(int(steps/skipSteps)+1), 
                                    dtype = "f")
    h5pEnergy=out_h5.create_dataset("potentialEnergy",(int(steps/skipSteps)+1), 
                                    dtype = "f")
    h5coords.attrs['units'] = "nanometers"
    h5coords.attrs['frame'] = "lab-fixed"
    h5coords.attrs['temperature'] = Temperature
    h5coords.attrs['beads'] = P
    h5coords.attrs['steps'] = steps
    h5coords.attrs['skipSteps'] = skipSteps
    h5coords.attrs['dt'] = dt/unit.femtosecond
    h5coords.attrs['gamma_denom'] = gamma_denom
    if constraintTol != None:
        h5coords.attrs['constraint_tols'] = constraintTol
        
    print("skipping equilibration period")

    simulation.step(1000)

    print("skipped equilibration period")

    nSteps = int(steps/skipSteps)

    for step in range(nSteps):
        
        print("stepping {}/{}".format(step, nSteps))

        simulation.step(skipSteps)
        print("stepped {}/{}".format(step, nSteps))

        state = simulation.context.getState(getEnergy=True)

        print("acquired state {}/{}".format(step, nSteps))
        h5kEnergy[step] = state.getKineticEnergy()/unit.kilojoules_per_mole
        h5pEnergy[step] = state.getPotentialEnergy()/unit.kilojoules_per_mole

        for beadi in range(P):
            current_state = simulation.integrator.getState(beadi,
                                                           getPositions=True)
            posi = current_state.getPositions(asNumpy=True)

            # record the position of beads in the h5 file
            h5coords[step, beadi, 0, :] = posi[0]/unit.nanometer
            h5coords[step, beadi, 1, :] = posi[1]/unit.nanometer
            h5coords[step, beadi, 2, :] = posi[2]/unit.nanometer

    #simulation.reporters[2].close()
    beadTraj.close()

    print("complete: T{}, P{}, N{}, dt{}, 1/g{}".format(Temperature,P,steps,dt,gamma_denom))

#function to call pimd and generate imaginary time correlation function plot
def rotCorFun(temperatures, Ps, period, dt, gamma_denom = 1, skipSteps = 1,
              constraints = True, constraintTol = None, c60 = False, 
              version = None):
    #define all standard variables for pimd
    #standard duration and time: 10 ps, 0.1 fs
    #iterate through all of the temperatures we are assessing the function at

    for i in Ps:
            for ind, j in enumerate(temperatures):
                #c = colour[temperatures.index(j)]
                #label for each line
                #text = "T = {0} K"

                if constraints and constraintTol is None:
                    folder = "P{beads}_{temp}K_{step}fs_{period}ps_g{gamma}_{ss}skip_results"
                elif constraints and constraintTol is not None:
                    folder = "P{beads}_{temp}K_{step}fs_{period}ps_g{gamma}_{ss}skip_tol%s_results" % constraintTol
                else:
                    folder = "P{beads}_{temp}K_{step}fs_{period}ps_g{gamma}_{ss}skip_flex_results"
                folder = folder.format(beads = i, temp = j, 
                                       step = dt[ind], period = period[ind],
                                       gamma = gamma_denom, ss = skipSteps)
                
                if version:
                    folder = folder + "_" + version

                pimd(period[ind], skipSteps, i, j, dt[ind]*unit.femtoseconds,
                     folder, gamma_denom, c60 = c60, constraints = constraints)

# trial with no constraints
Ps = 8*np.ones(1, dtype = int)
Ts = [50, 150, 300]
Ls = [1, 1, 1]
stepSizes = [0.1] #fs

"""rotCorFun(temp, ps, steps, dt (fs), gamma, skipsteps, constraints, constraintTol, c60)"""
rotCorFun(Ts, Ps, Ls, [0.1, 0.1, 0.1], gamma_denom = 100, constraints = False, 
          c60 = False, version = "V76_simple")
