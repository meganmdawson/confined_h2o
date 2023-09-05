import os
import h5py
import sys
from threading import Timer

def save(f, gamma0, L, dt, data = {}, overwrite = False):
    """ Creates a h5df containing trajectory information in a new group. Ensures 
        that data in old groups for the same simulation type is not overwritten 
        unless specified.

        Input:
        f: str
            Path for the file that the data will be saved to
        gamma0: float
            Friction value utilized in integrator
        L: float
            Length of simulation (in picoseconds)
        dt: float
            Length of timesteps (in femtoseconds)
        overwrite: bool
            Instruction on whether to erase data already existing under a 
            specified path to replace with new info.
        data: dict
            A dictionary containing data and the labels that are to be used in
            the affiliated h5f
            
        Functions:
        whitePILE (in constrained.py)
    """
    # check if the file already exists
    exists = os.path.exists(f)

    if overwrite or not exists: # create new file
        print("Creating a new file...")
        h5f = h5py.File(f, 'w')
    else: # append to existing file
        print("Appending data to an existing file...")
        h5f = h5py.File(f, 'a')

    grpName = genGroupName(dt, gamma0, L)
    grp = h5f.create_group(grpName)
    print("The simulation will save under the following group name: ", grpName)

    # save all information provided for group in separate datasets
    for key in data:
        dset = grp.create_dataset(key, data = data[key], dtype = 'f')


def genGroupName(dt, gamma0, L):
    """ Generate names for groups within hdf5 files
        Inputs:
            dt: float
                Size of time step (femtoseconds)
            gamma0: float
                Friction utilized for integrator
            L: float
                Total length of simulation (picoseconds)
            
        Functions:
        save
    """
    template = "{}fs_{}ps_{}g"
    f = template.format(dt, L, gamma0)

    return f

def genFileName(P, T, folder = 'data', integrator = 'wPILE',):
    """ Generate file names given input parameters
        Inputs:
        P: int
            Number of beads/duplicates
        T: int
            Temperature of simulation
        folder: str
                The name of the folder that the data will be written to. This is
                typically 'data'
        integrator: str
            Name of integrator utilized. This determines the subfolder within
            the data folder that information is stored to. Often this is 'wPILE'
        
        Functions:
        whitePILE
    """
    template = "P{}_{}K_results.hdf5"
    f = template.format(P, T)

    root = os.getcwd()

    # check if path to folder exists
    f_folder = os.path.join(root, folder, integrator)

    if not os.path.exists(f_folder): # create folder if it does not exist
        os.mkdir(f_folder)

    f_full = os.path.join(f_folder, f)

    return f_full


def checkSimExists(f, gamma0, L, dt, timeout = 60):
    """ Checks if the given simulation has already been run given the file and
        group name affiliated with the input parameters 

        Input:
        f: str
            Path for the file that the data will be saved to
        gamma0: float
            Friction value utilized in integrator
        L: float
            Length of simulation (in picoseconds)
        dt: float
            Length of timesteps (in femtoseconds)
        timeout: int
            Number of seconds that the program will wait for user input before 
            removing a pre-existing group to allow for rewriting of the group 
            with a new simulation
            
        Functions:
        whitePILE (in constrained.py)"""
    
    # check if the file exists
    if os.path.exists(f): # open the file if it exists
        h5 = h5py.File(f, 'r+')
    else: # continue program if the file does not exist
        return
    
    grpName = genGroupName(dt, gamma0, L)

    if grpName in h5: # check if the group (sim) already exists
        # give the user an opportunity to abort the re-writing of the simulation
        notice = "The desired trajectory already exists. \n" + \
                 "To prevent rewriting, enter 'e' within the next minute. " + \
                 "To rewrite immediately, enter 'r'. "
        redo = "\nYou have selected to replace the following simulation:\n" + \
               "{}: gamma = {}, L = {} ps, dt = {} fs".format(f, gamma0, L, dt) 
        skip = "\nYou have selected to keep the saved simulation:\n" + \
               "{}: gamma = {}, L = {} ps, dt = {} fs".format(f, gamma0, L, dt)

        t = Timer(timeout, print, [redo])
        
        t.start()
        answer = input(notice)
        t.cancel()
    
        if answer == "e":
            h5.close()
            sys.exit(skip)

        elif answer == "r":
            print(redo)
            # delete the existing entry to permit overwriting
            del h5[grpName]
            h5.close()
            return
        
        elif answer != "":
            h5.close()
            sys.exit("Invalid input. Simulation skipped.")

        return


    