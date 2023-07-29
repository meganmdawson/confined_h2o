# -*- coding: utf-8 -*-
"""
stat_fns.py

Contains functions to calculate statistics of a given system. A system output
from rigid.py is accepted by the various functions. The file from rigid.py 
typically has a name following the format  {r}C{P}_{T}.dat, where
    r = direction (x, y, or z)
    P = number of beads
    T = temperature of the simulation

@author: megan
"""
from ctypes.wintypes import HACCEL
import os
from ssl import HAS_TLSv1_2
import numpy as np
import matplotlib.pyplot as plt
import h5py
import file_manager

# object containing the data from the input system file ({r}C{P}_{T}.dat)
class trajectory:
    def __init__(self, beads, temperature, stepsize, period = 0, gamma = 1, 
                 skipSteps = 1, filename_format = None, constraints = True,
                 constraintTol = None, c60 = False, pill = False,
                 waterType = "tip4p", version = None):
        """
        Parameters
        ----------
        beads : int
            Number of beads present in your simulation.
        temperature : int
            Temperature that the simulation was run at.
        stepsize : float
            Size of steps that the simulation was performed with (fs).
        period : int
            The length of the simulation in picoseconds.
        REMOVED: steps : int
            Expected final number of steps. Used as an identifier in folder
            names.
        gamma : float
            Denominator of gamma used for simulation. This is an identifier
            in titles of newer folders.
        skipSteps: int
            indicates the number of steps that can be skipped.
        filename_format : string, optional
            Format of the name of the file containing information about the 
            bead trajectories. To format for temperature, use {T}, while {P} 
            is the number of beads, {dt} is the stepsize and {dim} is the
            dimension that the file is affiliated with (either x, y, or z).
            The default input is None; if no input is provided the standard 
            format from rigid.py is assumed.
        constraints: boolean, optional
            Indicates whether the simulation is performed with constraints.
            If False, then the additional term 'flex' is added to the 
            folder name.                       
        constraintTol: float, optional
            Indicates whether a user-input constraint tolerance was specified 
            for the integrator when the simulation was conducted. If not 
            specified, None is passed so the standard naming convention is used.
        waterType: string, optional
            Type of water model that is used in the simulation. Typically this 
            is tip4p.   
        c60: boolean, optional
            Informs the program of whether a c60 was added to the system that is
            being measured. This is solely important for naming convention      
        version: string, optional
            Informs the program of the openmm version utilized to generate a sim
            to better keep track of errors that may arise due to differences
            between updates. 
            Formatted as V{}, where the number -- without decimals -- is placed
            in the parentheses       

        Returns
        -------
        None.

        """
        # initialize constants
        self.k_B = 1.380649e-23 # J/K

        # extract information about the simulation
        self.beads = beads
        self.temperature = temperature
        self.stepsize = stepsize
        self.tau = 1/(temperature * beads) # NOTE: I need to review why we don't mult by k_B
        self.gamma = gamma
        self.period = period
        self.tot_steps = int(np.floor(self.period*1000/self.stepsize))
        self.constraints = constraints
        self.skipSteps = skipSteps
        self.constraintTol = constraintTol

        self.extractedMFF = False
        self.waterType = waterType
        self.steps = self.tot_steps/self.skipSteps
        self.c60 = c60
        self.pill = pill

        self.version = version

    def root_path(self, steps = None):
        """
        Returns the default root path that relevant files for the simulation
        would be stored in.
        
        If a full simulation is being investigated, do not input any number of 
        steps. Else, input the full expected number of steps.

        Returns
        -------
        folder_name: the standard folder name that should be used to access
                     previously generated data.

        """
        
        period = self.period       
        T = self.temperature
        beads = self.beads
        stepsize = self.stepsize
        gamma = self.gamma
        skip = self.skipSteps

        folder_name = file_manager.genFileName(beads, T)
        print("folder name is:", folder_name)
        
        return folder_name
        
    def extractMFF(self):
        """
        Opens all position files relevant to the simulation and stores their
        data in dictionary objects. The x, y, and z dictionaries each correspond 
        to a molecule-fixed-frame; each contains a set of lab-fixed coordinates 
        describing the molecule-fixed-frame at a given time.

        Eg.: 
            x (dict):

            Key:          Description:
            "taus"        Array of time steps in simulation (shape = (L, P),
                          L = number of time steps)
            "xs"          Array of coordinates along the x-axis of the 
                          lab-fixed-frame to describe the x axis of the 
                          molecule-fixed-frame; unnormalized, (L, P)
            "ys"          Array of coordinates along the y-axis of the  
                          lab-fixed-frame to describe the x axis of the 
                          molecule-fixed-frame; unnormalized, (L, P) 
            "zs"          Array of coordinates along the z-axis of the 
                          lab-fixed-frame  to describe the x axis of the 
                          molecule-fixed-frame; unnormalized, (L, P) 
            "mags"        Array of magnitudes of the molecule-fixed x-axis at 
                          each time step according to the lab-fixed frame:
                          sqrt(xs[t]^2 + ys[t]^2 + zs[t]^2); 
                          unnormalized, (L, P)

        Does not return any objects; assigns x, y, and z dictionary to the
        simulation object.
        """

        # generate lists to store information from the system file 
        # (converted to array later)
        self.x = {}
        self.y = {}
        self.z = {}
        
        # unpack coordinate data in affiliated hdf5
        try:
            getattr(self, "coordinates_h5")
        except AttributeError:
            self.unpackH5(attributes = ["coordinates"])

        self.steps = self.coordinates_h5.shape[0]

        halfH = 0.5 *(self.coordinates_h5[:, :, 1]+self.coordinates_h5[:, :, 2]) 

        z_axis = self.coordinates_h5[:, :, 0, :] - halfH
        x_axis = self.coordinates_h5[:, :, 2, :] - halfH

        y_axis = np.cross(z_axis, x_axis)

        x_axis = np.cross(y_axis, z_axis)

        taus = np.tile(np.arange(self.steps) * self.stepsize, self.beads) 
        taus = taus.reshape(self.steps, self.beads) * self.tau

        MFFcoords = [x_axis, y_axis, z_axis]

        print()

        # iterate through the various dimensions and read data for each 
        for i, dim in enumerate([self.x, self.y, self.z]):

            dim["taus"] = taus
            dim['xs'] = MFFcoords[i][:, :, 0] # still need to reshape -- check shape of dim["xs"]
            dim['ys'] = MFFcoords[i][:, :, 1]
            dim['zs'] = MFFcoords[i][:, :, 2]
            dim['mags'] = np.linalg.norm(MFFcoords[i], axis = 2)

            print("min magnitude = ", np.min(dim['mags']))
            print("max magnitude = ", np.max(dim['mags']))

            self.extractedMFF = True

    def unpackH5(self, attributes = ["coordinates"], default_folder = True):
        """open the relevant .h5 file to extract various variables. All listed 
           attributes are added to the simulation object.
        
        Parameters
        ----------
        attributes : list, optional 
            List of strings, each referring to a field you would like 
            to retrieve from the .h5 file. If nothing is provided, the default
            of just positions is selected. 

        Attributes included in output from rigid.py are:
            Name in h5:         Name of object:         Units:
            "cell_angles"       "cell_angles_h5"        degrees
            "cell_lengths"      "cell_lengths_h5"       nanometers
            "coordinates"       "coordinates_h5"        nanometers
            "kineticEnergy"     "kineticEnergy_h5"      kilojoules_per_mole
            "potentialEnergy"   "potentialEnergy_h5"    kilojoules_per_mole
            "temperature"       "temperature_h5"        kelvin
            "time"              "time_h5"               picoseconds
            "topology"          "topology_h5"           flavour: python
        The keyword used to extract an item is the first column. This is also 
        what should be input to this function. 
        The second column is the name of the variable in the simulation object.
        When accessing the object after it is saved in this function, it should
        be called using the name with the "_h5" prefix.     

        """
        groupName = file_manager.genGroupName(dt = self.stepsize, 
                                              gamma0 = self.gamma,
                                              L = self.period)
        
        if default_folder == True:
            out_path = self.root_path()
            # out_path = os.path.realpath(out_path) # removed for updated 
            #                                         comparison scripts
            
        else:
            h5Name = "P{P}_{T}K_results.hdf5".format(P = self.beads, 
                                                     T = self.temperature)
            out_path = os.path.realpath(h5Name)

        # open h5 file
        with h5py.File(out_path, 'r') as f:
            # iterate through the list of attributes that should be saved
            for attr in attributes:
                setattr(self, attr+"_h5", f[groupName][attr][:])

    def rotational_correlation(self, altSkipSteps = None, default_folder = True):
        """calculate the rotational correlation function based on internal vals.
           This method does not calculate error and will consider variables as 
           they are already defined in the simulation object. 
           
           Returns
           -------
           None.
           
           """
        
        # ensure that relevant files have been read and stored as dicts
        if not self.extractedMFF:
            self.extractMFF()

        # store correlation function in out_path
        out_file = "Ct{P}_{T}.dat".format(P = self.beads, 
                                          T = self.temperature)
        
        if default_folder == True:
            out_path = os.path.join(self.root_path(), out_file)
            out_path = os.path.realpath(out_path)
            
        else:
            out_path = os.path.realpath(out_file)
       
        ctau_dat = open(out_path, 'w')
        
        ctau = np.zeros((self.beads, 3), dtype = "float32")
        
        if altSkipSteps == None:
            optimSkipSteps = self.skipSteps
        else:
            optimSkipSteps = altSkipSteps
        
        totSteps = round(self.steps/optimSkipSteps)
        nData = self.beads * totSteps
        
        # create np arrays to hold the body-frame axis info in one object
        # ensure that all values are normalized
        """ shape of dim_xyz is the following:
                0   1   2   3  ...  P
            x:  x1  x2  x3  x4      xP
            y:  y1  y2  y3  y4      yP
            z:  z1  z2  z3  z4      zP
        """
        x_xyz = np.array([self.x["xs"], self.x["ys"], 
                          self.x["zs"]])/self.x["mags"]
        y_xyz = np.array([self.y["xs"], self.y["ys"], 
                          self.y["zs"]])/self.y["mags"]
        z_xyz = np.array([self.z["xs"], self.z["ys"], 
                          self.z["zs"]])/self.z["mags"]

        # iterate through all dimensions (x, y, z)
        for ind, dim in enumerate([x_xyz, y_xyz, z_xyz]):
            x = dim[0]
            y = dim[1]
            z = dim[2]
            
            # iterate through all of the steps in the simulation
            for step in range(totSteps):
                # iterate through pairs of beads
                for beadi in range(self.beads):
                    for beadj in range(self.beads):
                        #selects a bead that is j beads away from i
                        beadij=(beadj+beadi)%(self.beads)
                        
                        xyz_ij = [x[step, beadij], y[step, beadij], 
                               z[step, beadij]]
                        xyz_i = [x[step, beadi], y[step, beadi], 
                               z[step, beadi]]
                        
                        ctau[beadj, ind]+=np.dot(xyz_ij, xyz_i)
        
        # find the average value between two beads across the midpoint, equally
        # spaced by i beads from each end (midpoint = P/2)
        for beadi in range(self.beads):
            if beadi>0 and beadi<self.beads/2:
                for a in range(3):
                    ctau[beadi, a]=.5*(ctau[beadi, a]+ctau[self.beads-beadi, a])

            if beadi>self.beads/2:
                for a in range(3):
                    ctau[beadi, a]=ctau[self.beads-beadi, a]                     
                        
            # record the final ctau values
            ctau_dat.write(str(beadi*self.tau)+ ' ' + 
                           str(ctau[beadi, 0]/nData) + " " + 
                           str(ctau[beadi, 1]/nData) + " " + 
                           str(ctau[beadi, 2]/nData) + '\n')
            
        # must record final step that returns the function to unity
        ctau_dat.write(str(self.beads*self.tau) + ' ' + 
                       str(ctau[0, 0]/nData) + ' ' + 
                       str(ctau[0, 1]/nData) + ' ' + 
                       str(ctau[0, 2]/nData) + '\n')
        
        # create/update ctau value belonging to object
        self.ctau = ctau
        
    def rot_correlation_se(self, default_folder = True, 
                           altSkipSteps = None):
        """calculate the standard error associated with the 
           rotational correlation function. This method considers all variables
           that were used in the original simulation. 
           
           Parameters
           ----------
           default_folder: boolean, optional
                determines whether to use the standard data path given the 
                simulation variables or not.
           altSkipSteps: int, optional
                If nothing is entered, the default value of skipsteps as defined
                in the simulation object will be used. If there is an 
                alternative number of skipsteps which is indicated by the time 
                correlation function to be optimal, that value can be entered
                into this function. 
        """
        
        # ensure that relevant files have been read and stored as dicts
        if not self.extractedMFF:
            self.extractMFF()

        # define number of steps to iterate over
        if altSkipSteps == None:
            optimSkipSteps = self.skipSteps
        else:
            optimSkipSteps = altSkipSteps
        
        totSteps = round(self.steps/optimSkipSteps)
        
        # store correlation function in out_path
        out_file = "stdError{P}_{T}.dat".format(P = self.beads, 
                                                T = self.temperature)
        
        if default_folder == True:
            out_path = os.path.join(self.root_path(), out_file)
            out_path = os.path.realpath(out_path)
            
        else:
            out_path = os.path.realpath(out_file)
        
        error_dat = open(out_path, 'w')
        
        error = np.zeros((self.beads, 3), dtype = "float32")
        ctau = np.zeros((self.beads, 3), dtype = "float32")
        
        nData = self.beads * totSteps
        
        # create np arrays to hold the body-frame axis info in one object
        """ shape of dim_xyz is the following:
                      0   1   2   3  ...  P
            x:  T1    x1  x2  x3  x4      xP
                T2    x1  x2  x3  x4      xP
                ...    
                Tn    x1  x2  x3  x4      xP
            
            y:  T1    y1  y2  y3  y4      yP
                T2    y1  y2  y3  y4      yP
                ...    
                Tn    y1  y2  y3  y4      yP
            
            z:  T1    z1  z2  z3  z4      zP
                T2    z1  z2  z3  z4      zP
                ...
                Tn    z1  z2  z3  z4      zP
        """
        
        print("delete this later: ", np.shape(self.x["xs"]), np.shape(self.x["mags"]))
        print(type(self.x["xs"]), type(self.x["mags"]))

        x_xyz = np.array([self.x["xs"], self.x["ys"], 
                          self.x["zs"]])/self.x["mags"]
        
        y_xyz = np.array([self.y["xs"], self.y["ys"], 
                          self.y["zs"]])/self.y["mags"]
        
        z_xyz = np.array([self.z["xs"], self.z["ys"],  
                          self.z["zs"]])/self.z["mags"]
        
        # iterate through all dimensions (x, y, z)
        for ind, dim in enumerate([x_xyz, y_xyz, z_xyz]):
            x = dim[0]
            y = dim[1]
            z = dim[2]
            
            # iterate through all of the steps in the simulation
            for step in range(totSteps):
                # iterate through pairs of beads
                for beadi in range(self.beads):
                    for beadj in range(self.beads):
                        #selects a bead that is j beads away from i
                        beadij=(beadj+beadi)%(self.beads)
                        
                        xyz_ij = [x[step, beadij], y[step, beadij], 
                               z[step, beadij]]
                        xyz_i = [x[step, beadi], y[step, beadi], 
                               z[step, beadi]]
                         
                        ctau[beadj, ind]+=np.dot(xyz_ij, xyz_i)/nData
                        error[beadj, ind]+=(np.dot(xyz_ij, xyz_i)**2)/nData
        
        # find the average value between two beads across the midpoint, equally
        # spaced by i beads from each end (midpoint = P/2)
        for beadi in range(self.beads):
            if beadi>0 and beadi<self.beads/2:
                for a in range(3):
                    ctau[beadi, a]=.5*(ctau[beadi, a]+ctau[self.beads-beadi, a])
                    error[beadi, a]=.5*(error[beadi, a]+
                                        error[self.beads-beadi, a])
            if beadi>self.beads/2:
                for a in range(3):
                    ctau[beadi, a]=ctau[self.beads-beadi, a]
                    error[beadi, a]=error[self.beads-beadi, a]
                    
        # ctau = average; error = sum of squared terms in ctau/ndata
        # find the std error from the sum of squares
        error = np.sqrt(np.abs((error - ctau**2)/nData))  
                        
        # record the final error values
        for beadi in range(self.beads):
            error_dat.write(str(beadi*self.tau)+ ' ' + 
                           str(error[beadi, 0]) + " " + 
                           str(error[beadi, 1]) + " " + 
                           str(error[beadi, 2]) + '\n')
            
        # calculate the final error value for the step returning the function
        # to unity
        error_dat.write(str(self.beads*self.tau) + ' ' + 
                       str(error[0, 0]) + ' ' + 
                       str(error[0, 1]) + ' ' + 
                       str(error[0, 2]) + '\n')
        
        # create/update ctau se value belonging to object
        self.ctau_se = error
        
    def projections(self, bins = 15, default_folder = True):
        """find the projection of each of the BFF axes on the stationary LFF 
           axes, and then bin the results and generate a histogram of the 
           results.

           Uses the values stored in xs, ys and zs as they are already the 
           projection of MFF axes onto their corresponding LFFs
           
           Parameters
           ----------
           bins: int, optional
                Number of bins to sort data into
           default_folder: boolean, optional
                determines whether to use the standard data path given the 
                simulation variables or not.
           altSkipSteps: int, optional
                If nothing is entered, the default value of skipsteps as defined
                in the simulation object will be used. If there is an 
                alternative number of skipsteps which is indicated by the time 
                correlation function to be optimal, that value can be entered
                into this function.
           """
        
        # ensure that relevant files have been read and stored as dicts
        if not self.extractedMFF:
            self.extractMFF()

        # store correlation function in out_path
        out_file = "projections{P}_{T}.jpg".format(P = self.beads, 
                                                   T = self.temperature)
        
        if default_folder == True:
            out_path = os.path.join(self.root_path(), out_file)
            out_path = os.path.realpath(out_path)
            
        else:
            out_path = os.path.realpath(out_file)
    
        normXx = np.divide(self.x["xs"], self.x["mags"]).flatten()
        normXy = np.divide(self.x["ys"], self.y["mags"]).flatten()

        theta = np.arctan2(normXy, normXx)
        
        normZ = np.divide(self.z["zs"], self.z["mags"])
        normZ = normZ.flatten()

        zCoords = np.array([self.z["xs"], self.z["ys"], self.z["zs"]])/self.z["mags"]

        # take dot product with e_z [0 0 1] (== cos(azimuthal) )
        zDot = zCoords[2]
        # take magnitude of cross with e_z [0 0 1] (== sin(azimuthal), jacobian)
        zCross = np.cross(zCoords, np.array([0,0,1]), axisa = 0).transpose()
        zCross = np.linalg.norm(zCross, axis = 0).transpose()

        # prob density for z norm
        zNormDensity = np.mean(np.multiply(zDot, zCross), axis = 1)

        # azimuthal angle
        azi = np.arccos(normZ)

        normZDensity = 2*np.multiply(normZ, np.sin(azi))

        # f, axs = plt.subplots(2, 2)
        # axs[0,0].hist(azi, bins = bins, density = True)
        # axs[0,1].hist(normZDensity, bins = bins, density = True)
        # axs[1,0].hist(normZ, bins = bins, density = True)
        # axs[1,1].hist(np.sin(azi), bins = bins)

        # axs[0,0].set_xlabel(r'$\phi$')
        # axs[0,0].set_ylabel(r'$\rho$')
        # axs[0,0].title.set_text("Azimuthal probability density")

        # axs[0,1].set_xlabel(r'$z_{z}$')
        # axs[0,1].set_ylabel(r'$\rho$')
        # axs[1,0].title.set_text("Z-projection probability density")

        f, axs = plt.subplots(2, 1)
        axs[0].hist(azi, bins = bins, density = True)
        axs[1].hist(normZDensity, bins = bins, density = True)

        axs[0].set_xlabel(r'$\phi$')
        axs[0].set_ylabel(r'$\rho$')
        axs[0].title.set_text(r'Orientation of $e_{z}$')

        axs[1].set_xlabel(r'$z_{z}$')
        axs[1].set_ylabel(r'$\rho$')
        
        plt.tight_layout()
        plt.show()
        plt.savefig(out_path)

        plt.close()

    def rot_corr_and_error(self, default_folder = True, altSkipSteps = None):
        """calculate the rotational correlation function and the standard error  
           associated with the rotational correlation function. This function 
           typically uses the standard variables defined in the simulation 
           object. 

           Parameters
           ----------
           default_folder: boolean, optional
                determines whether to use the standard data path given the 
                simulation variables or not.
           altSkipSteps: int, optional
                If nothing is entered, the default value of skipsteps as defined
                in the simulation object will be used. If there is an 
                alternative number of skipsteps which is indicated by the time 
                correlation function to be optimal, that value can be entered
                into this function. """

        # ensure that relevant files have been read and stored as dicts
        if not self.extractedMFF:
            self.extractMFF()

        # define number of steps to iterate over
        if altSkipSteps == None:
            optimSkipSteps = self.skipSteps
        else:
            optimSkipSteps = altSkipSteps
        
        totSteps = round(self.steps/optimSkipSteps)

        # store correlation function in out_path_rcf and error in out_path_se
        out_file_se = "stdError{P}_{T}.dat".format(P = self.beads, 
                                                   T = self.temperature)
        out_file_rcf = "Ct{P}_{T}.dat".format(P = self.beads, 
                                              T = self.temperature)
        
        if default_folder == True:
            out_path_se = os.path.join(self.root_path(), out_file_se)
            out_path_se = os.path.realpath(out_path_se)
            
            out_path_rcf = os.path.join(self.root_path(), out_file_rcf)
            out_path_rcf = os.path.realpath(out_path_rcf)
            
        else:
            out_path_se = os.path.realpath(out_file_se)
            out_path_rcf = os.path.realpath(out_file_rcf)
        
        ctau_dat = open(out_path_rcf, 'w')
        error_dat = open(out_path_se, 'w')
        
        error = np.zeros((self.beads, 3), dtype = "float32")
        ctau = np.zeros((self.beads, 3), dtype = "float32")
        
        nData = self.beads * totSteps
        
        # create np arrays to hold the body-frame axis info in one object
        """ shape of dim_xyz is the following:
                      0   1   2   3  ...  P
            x:  T1    x1  x2  x3  x4      xP
                T2    x1  x2  x3  x4      xP
                ...    
                Tn    x1  x2  x3  x4      xP
            
            y:  T1    y1  y2  y3  y4      yP
                T2    y1  y2  y3  y4      yP
                ...    
                Tn    y1  y2  y3  y4      yP
            
            z:  T1    z1  z2  z3  z4      zP
                T2    z1  z2  z3  z4      zP
                ...
                Tn    z1  z2  z3  z4      zP
        """
        
        print("delete this later: ",np.shape(self.x["xs"]),np.shape(self.x["mags"]))
        print(type(self.x["xs"]), type(self.x["mags"]))

        x_xyz = np.array([self.x["xs"], self.x["ys"], 
                          self.x["zs"]])/self.x["mags"]
        
        y_xyz = np.array([self.y["xs"], self.y["ys"], 
                          self.y["zs"]])/self.y["mags"]
        
        z_xyz = np.array([self.z["xs"], self.z["ys"],  
                          self.z["zs"]])/self.z["mags"]
        
        # iterate through all dimensions (x, y, z)
        for ind, dim in enumerate([x_xyz, y_xyz, z_xyz]):
            x = dim[0]
            y = dim[1]
            z = dim[2]
            
            # iterate through all of the steps in the simulation
            for step in range(totSteps):
                # iterate through pairs of beads
                for beadi in range(self.beads):
                    for beadj in range(self.beads):
                        #selects a bead that is j beads away from i
                        beadij=(beadj+beadi)%(self.beads)
                        
                        xyz_ij = [x[step, beadij], y[step, beadij], 
                               z[step, beadij]]
                        xyz_i = [x[step, beadi], y[step, beadi], 
                               z[step, beadi]]
                         
                        ctau[beadj, ind]+=np.dot(xyz_ij, xyz_i)/nData
                        error[beadj, ind]+=(np.dot(xyz_ij, xyz_i)**2)/nData
                        
        # find the average value between two beads across the midpoint, equally
        # spaced by i beads from each end (midpoint = P/2)
        for beadi in range(self.beads):
            if beadi>0 and beadi<self.beads/2:
                for a in range(3):
                    ctau[beadi, a]=.5*(ctau[beadi, a]+ctau[self.beads-beadi, a])
                    error[beadi, a]=.5*(error[beadi, a]+
                                        error[self.beads-beadi, a])
            if beadi>self.beads/2:
                for a in range(3):
                    ctau[beadi, a]=ctau[self.beads-beadi, a]
                    error[beadi, a]=error[self.beads-beadi, a]
                    
        # ctau = average; error = sum of squared terms in ctau/ndata
        # find the std error from the sum of squares
        error = np.sqrt(np.abs((error - ctau**2)/nData)) 
        
        # record the final correlation and error values
        for beadi in range(self.beads):              
            error_dat.write(str(beadi*self.tau)+ ' ' + 
                           str(error[beadi, 0]) + " " + 
                           str(error[beadi, 1]) + " " + 
                           str(error[beadi, 2]) + '\n')
            
            ctau_dat.write(str(beadi*self.tau)+ ' ' + 
                           str(ctau[beadi, 0]) + " " + 
                           str(ctau[beadi, 1]) + " " + 
                           str(ctau[beadi, 2]) + '\n')
            
        # calculate the final error value for the step returning the function
        # to unity
        error_dat.write(str(self.beads*self.tau) + ' ' + 
                       str(error[0, 0]) + ' ' + 
                       str(error[0, 1]) + ' ' + 
                       str(error[0, 2]) + '\n')
        
        ctau_dat.write(str(self.beads*self.tau) + ' ' + 
                       str(ctau[0, 0]) + ' ' + 
                       str(ctau[0, 1]) + ' ' + 
                       str(ctau[0, 2]) + '\n')
        
        # create/update ctau and error value belonging to object
        self.ctau = ctau
        self.ctau_se = error

    def autocorrelation(self, default_folder = True):
        """
        Generates the time correlation function for the z axis only. This
        is used to determine the value of skipsteps.
        Results are output in a file of the format 
        cdt_{nSteps}_{stepsize}fs.dat

        Updated to reflect std method utilized in NE452
        """
        # ensure that relevant files have been read and stored as dicts
        if not self.extractedMFF:
            self.extractMFF()

        # store correlation function in out_path
        out_file = "cdt_{n}_{size}.dat".format(n = self.beads,
                                               size = self.period)
        
        if default_folder == True:
            out_path = os.path.join(self.root_path(), out_file)
            out_path = os.path.realpath(out_path)
            
        else:
            out_path = os.path.realpath(out_file)
        
        ct_out = open(out_path, "w")
        ctime = np.zeros((self.steps, 1), dtype = "float32")
        
        # normalize all of the ez components
        z_xyz = np.array([self.z["xs"], self.z["ys"],  
                          self.z["zs"]])/self.z["mags"]
        
        # determine the average and variance in the z-axis
        mean = np.average(z_xyz, axis = 1)
        var = np.var(z_xyz, axis = 1)

        print("z_xyz.shape", z_xyz.shape)
        print("mean.shape", mean.shape)
        print("var.shape", var.shape)

        # iterate over each bead
        # only for ez, although could be modified with another loop to 
        # also work with other dims (ctime[:, 0] instead of ctime)
        nSteps = round(self.steps/self.skipSteps)
        for stepi in range(nSteps):
            t_i = nSteps - stepi
            coeff = 1/(t_i * self.beads)

            # iterate over each time step 
            for bead in range(self.beads):
                ez_ik = z_xyz[:, stepi, bead]
                
                # iterate over steps after time origin
                for stepj in range(stepi, self.steps):
                    ez_jk = z_xyz[:, stepj, bead]
                    
                    d_ik = ez_ik - mean[:, bead]
                    d_jk = ez_jk - mean[:, bead]
                    
                    dt = stepj - stepi

                    ctimeij = np.average(np.dot(d_ik, d_jk)/var[:, bead])
                    ctime[dt] += coeff * ctimeij
                    
        # write results to output file
        for stepi in range(round(self.steps/self.skipSteps)):
            ct_out.write(str(self.stepsize * stepi) + " " + 
                         str(ctime[stepi][0]) + " \n")
            
        ct_out.close()
            
        self.ctime = ctime
    
    def time_correlation(self, default_folder = True):
        """
        Generates the time correlation function for the z axis only. This
        is used to determine the value of skipsteps.
        Results are output in a file of the format 
        cdt_{nSteps}_{stepsize}fs.dat
        """
        # ensure that relevant files have been read and stored as dicts
        if not self.extractedMFF:
            self.extractMFF()

        # store correlation function in out_path
        out_file = "cdt_{n}_{size}.dat".format(n = self.beads,
                                               size = self.period)
        
        if default_folder == True:
            out_path = os.path.join(self.root_path(), out_file)
            out_path = os.path.realpath(out_path)
            
        else:
            out_path = os.path.realpath(out_file)
        
        ct_out = open(out_path, "w")
        ctime = np.zeros((self.steps, 1), dtype = "float32")
        
        # normalize all of the ez components
        z_xyz = np.array([self.z["xs"], self.z["ys"],  
                          self.z["zs"]])/self.z["mags"]
        
        # iterate over each bead
        # only for ez, although could be modified with another loop to 
        # also work with other dims (ctime[:, 0] instead of ctime)
        nSteps = round(self.steps/self.skipSteps)
        for stepi in range(nSteps):
            t_origin = stepi * self.stepsize * self.skipSteps
            
            # iterate over each time step 
            for bead in range(self.beads):
                ez_ik = z_xyz[:, stepi, bead]
                
                # iterate over steps after time origin
                for stepj in range(stepi, self.steps):
                    ez_jk = z_xyz[:, stepj, bead]

                    ctimeij = np.dot(ez_ik, ez_jk)/(self.beads*self.steps)
                    ctime[stepj - stepi] += ctimeij
                    
        # write results to output file
        for stepi in range(round(self.steps/self.skipSteps)):
            ct_out.write(str(self.stepsize * stepi) + " " + 
                         str(ctime[stepi][0]) + " \n")
            
        ct_out.close()
            
        self.ctime = ctime
                
    def plot_time_corr(self, default_folder = True):
        """
        Function to plot the time correlation function. This function will 
        search for information within the trajectory object. If there is no
        existing ctime object within the trajectory, it will search for an 
        appropriately titled data file. If this also does not exist,
        it will call the function required to generate the time correlation
        plot. Plots will be saved as cdtz_plot.png

        NOT FUNCTIONAL YET
        """
        
        plot_file = "cdtz_plot.png"
        
        # check if the time correlation function has already been made
        out_file = "cdt_{n}_{size}.dat".format(n = self.beads, 
                                               T = self.temperature)
        
        if default_folder == True:
            out_path = os.path.join(self.root_path(), out_file)
            out_path = os.path.realpath(out_path)
            
            plot_path = os.path.join(self.root_path(), plot_file)
            plot_path = os.path.realpath(plot_path)
            
        else:
            out_path = os.path.realpath(out_file)
        
        path_exists = os.path.exists(out_path)
        
        try:
            obj_exists = type(self.ctime) == np.ndarray
        except NameError:
            obj_exists = False
            
        # use the object if it exists
        if obj_exists:
            plt.figure()
            plt.title(("Time correlation function at {steps} steps \n" +
                      "dt = {dt} fs").format(steps = self.steps, 
                                             dt = self.stepsize))
            plt.plot(np.arange(self.steps/self.skipSteps)*self.stepsize* \
                     self.skipSteps, self.ctime)
            plt.xlabel("Time (fs)")
            plt.ylabel("Time correlation")
            
            # plot and label points where the correlation function crosses
            # the x-axis
            zero_pts = self.ctime[self.ctime == 0]                                 
            zero_pts_t = np.arange(self.steps)*self.stepsize[self.ctime==0]
            plt.scatter(zero_pts_t, zero_pts, c = "red", marker = "*")
            # consider adding labels for the scatter points that indicate the 
            # timestep where the function crosses the x axis
            
            # save the plot
            plt.savefig(plot_path, format = "png")
            plt.close()
                
        # use previous file with correct naming convention instead
        elif path_exists:
            ctime_list = []
            time_list = []
            
            # read each line in the file
            with open(out_path, "r") as f:
                ctime = f.readlines()
            
            for line in ctime:
                columns = line.split()
                time_list.append(columns[0])
                ctime_list.append(columns[1])
                
            ctime = np.array(ctime_list)
            time = np.array(time_list)
                
            # plot the function
            plt.figure()
            plt.title(("Time correlation function at {steps} steps \n" +
                      "dt = {dt} fs").format(steps = self.steps, 
                                             dt = self.stepsize))
            plt.plot(time, ctime)
            plt.xlabel("Time (fs)")
            plt.ylabel("Time correlation")
            
            # plot and label points where the correlation function crosses
            # the x-axis
            zero_pts = ctime[ctime == 0]                                 
            zero_pts_t = np.arange(self.steps)*self.stepsize[ctime==0]
            plt.scatter(zero_pts_t, zero_pts, c = "red", marker = "*")
            # consider adding labels for the scatter points that indicate the 
            # timestep where the function crosses the x axis
            
            # save the plot
            plt.savefig(plot_path, format = "png")
            plt.close()
            
        # calculate the time correlation function since it is not saved 
        # elsewhere
        else:
            
            self.time_correlation()
            
            plt.figure()
            plt.title(("Time correlation function at {steps} steps \n" +
                      "dt = {dt} fs").format(steps = self.steps, 
                                             dt = self.stepsize))
            plt.plot(np.arange(self.steps)*self.stepsize, self.ctime)
            plt.xlabel("Time (fs)")
            plt.ylabel("Time correlation")
            
            # plot and label points where the correlation function crosses
            # the x-axis
            zero_pts = self.ctime[self.ctime == 0]                                 
            zero_pts_t = np.arange(self.steps)*self.stepsize[self.ctime==0]
            plt.scatter(zero_pts_t, zero_pts, c = "red", marker = "*")
            # consider adding labels for the scatter points that indicate the 
            # timestep where the function crosses the x axis
            
            # save the plot
            plt.savefig(plot_path, format = "png")
            plt.close()
    
    def rotate(self, omega):
        """ Creates a rotation matrix according to the set of input Euler
            angles, omega.
        """
        # rotmat in the order x, y, z

        a, b, c = omega[0], omega[1], omega[2]
        cos = np.cos
        sin = np.sin

        rotmat = np.zeros((3,3))
        
        rotmat[0,0] = cos(a)*cos(b)*cos(c) - sin(a)*sin(c)
        rotmat[0,1] = -cos(a)*cos(b)*sin(c) - sin(a)*cos(c)
        rotmat[0,2] = cos(a)*sin(b)

        rotmat[1,0] = sin(a)*cos(b)*cos(c) + cos(a)*sin(c)
        rotmat[1,1] = -sin(a)*cos(b)*sin(c) + cos(a)*cos(c)
        rotmat[1,2] = sin(a)*sin(b)

        rotmat[2,0] = -sin(b)*cos(c) 
        rotmat[2,1] = sin(b)*sin(c)
        rotmat[2,2] = cos(b)

        return rotmat

    def energyEstimator2(self, default_folder = True, compareOpenMM = False): 
        # coordinates_h5 should be of shape (nSteps + 1, P, 3, 3)
        # first index is step number, second is bead number, third is molecule 
        # (O, H1, H2), and fourth is coordinate along a particular axis (x,y,z)

        # open a file to store energy values in
        fname = "exact_energy.dat"
        if default_folder == True:
            out_path = os.path.join(self.root_path(), fname)
            out_path = os.path.realpath(out_path)
            
        else:
            out_path = os.path.realpath(fname)
        
        energy_out = open(out_path, "w")

        # make a file for euler angles to be stored in
        eul_path = out_path.replace(fname, "euler_angles.dat")
        eul_out = open(eul_path, "w")

        # make a file for inertia terms to be stored in
        I_terms_path = out_path.replace(fname, "Inn_terms.dat")
        I_out = open(I_terms_path, "w")
        I_out.write("P step Ixx Iyy Izz \n")

        # make a file for bond lengths (ensure that constraints are working)
        bond_len_p = out_path.replace(fname, "bond_len.dat")
        bond_len = open(bond_len_p, "w")
        bond_len.write("P step ||O-H1|| ||O-H2|| ||H2-H1||\n")

        # check that coordinates from the corresponding .h5 file have been read
        try:
            obj_exists = type(self.coordinates_h5) == h5py._hl.dataset.Dataset
        except NameError:
            obj_exists = False
        
        if not obj_exists:
            self.unpackH5(attributes = ["coordinates"])

        hbar = 1.054571817e-34 # Js
        DaToKg = 1.660539066e-27 # kg/Da
        nmToM = 1e-9 # m/nm
        N_A = 6.0221e23 # /mol
        kB = 1.380649e-23 # J/K

        # set up dictionary of atoms in water and their corresponding masses
        masses = {'M':18.016, 'O':16.0, 'H1':1.008, 'H2':1.008}

        # find the centre of mass and corresponding moments of inertia
        # centre should be an (nSteps+1, P, 1, 3) array [x,y,z]
        centre = (masses['O']/masses['M'])*self.coordinates_h5[:, :, 0, :] + \
                 (masses['H1']/masses['M'])*self.coordinates_h5[:, :, 1, :] + \
                 (masses['H2']/masses['M'])*self.coordinates_h5[:, :, 2, :]

        # # convert the coordinates to the LFF based on the centre of mass
        # centreShape = centre.shape
        # centre = np.tile(centre, (1, 1, 3)).reshape(centreShape[0], 
        #                                             centreShape[1], 
        #                                             len(masses.keys())-1, 
        #                                             centreShape[2])

        coordsMFF = self.coordinates_h5 #- centre

        # indices of levi civita tensor that equal positive one
        posLeviCivita = (np.array([0,1,2]), np.array([1,2,0]), 
                         np.array([2,0,1]))

        # set up arrays to store the energy found at each step and each bead
        totSteps = centre.shape[0]
        energy = np.zeros(totSteps)

        # at each step find the related inertia tensor
        # note: for some reason the last step has only zeros; check rigid.py later
        for i in range(totSteps - 1):
            # create array for each term of energy estimator at this step&bead
            E = np.empty(3)

            for p in range(self.beads):
                p2 = (p+1)%self.beads            
                # find first terms of each component of the inertia tensor based
                # on the position of the centre of mass
                # eg: Ixx = -M*(Y^2 + Z^2) + ..., but in the MFF X, Y and Z = 0
                Ixx = 0 # x *should* start at CoM and point towards H2
                Iyy = 0 # y is orthogonal to the xz plane; vector starts at CoM 
                Izz = 0 # z is defined from CoM to midpoint between Hs

                # find the normalized axes of the MFF water @ the current bead
                x1 = coordsMFF[i, p, 2, :] - centre[i, p, :]
                x1 /= np.linalg.norm(x1)
                z1 = (coordsMFF[i, p, 2, :] + coordsMFF[i, p, 1, :])/2 - \
                     centre[i, p, :]
                z1 = z1/np.linalg.norm(z1)
                y1 = np.cross(x1, z1)
                y1 = y1/np.linalg.norm(y1)
                x1 = np.cross(y1, z1)
                x1 = x1/np.linalg.norm(x1)

                # find the normalized axes of MFF water @ the next bead
                x2 = coordsMFF[i, p2, 2, :] - centre[i, p2, :]
                x2 /= np.linalg.norm(x2)
                z2 = (coordsMFF[i, p2, 2, :] + coordsMFF[i, p2, 1, :])/2 - \
                     centre[i, p2, :]
                z2 = z2/np.linalg.norm(z2)
                y2 = np.cross(x2, z2)
                y2 = y2/np.linalg.norm(y2)
                x2 = np.cross(y2, z2)
                x2 = x2/np.linalg.norm(x2)
                
                # find the euler angles between the current bead & the next
                omegaPrime = [np.arctan2(z2[0], -z2[1]),
                              np.arccos(z2[2]),
                              np.arctan2(x2[2], y2[2])]

                # arctan2 input: y-coord, x-coord
                omega = [np.arctan2(z1[0], -z1[1]),
                         np.arccos(z1[2]),
                         np.arctan2(x1[2], y1[2])]
                
                eul_out.write(str(omega[0]) + ' ' + str(omega[1]) + ' ' + \
                              str(omega[2]) + '\n')

                # calculate corresponding rotation matrix
                A_omega = self.rotate(omega)
                A_omega_prime = self.rotate(omegaPrime)

                A_omega_tilda = np.matmul(np.transpose(A_omega), A_omega_prime)

                # calculate the position of each molecule relative to the axis 
                # defined by the previous bead
                relCoords = np.zeros((3,3))
                relCoords[0, :] = coordsMFF[i, p2, 0, :] - centre[i, p2, :]
                relCoords[1, :] = coordsMFF[i, p2, 1, :] - centre[i, p2, :]
                relCoords[2, :] = coordsMFF[i, p2, 2, :] - centre[i, p2, :]
                
                # if using the MFF of the same bead, all inertia terms should be the same??
                # iterate through all of the atoms in the water molecule (inertia)
                for j, atom in enumerate(["O", "H1", "H2"]):
                    # convert coords to MFF of current step
                    # atomMFFCoords = np.matmul(A_omega_prime, 
                    #                           relCoords[j, :])
                    atomMFFCoords = np.matmul(A_omega_prime.transpose(), 
                                              relCoords[j, :])
                    Ixx+= masses[atom]*(atomMFFCoords[1]**2+atomMFFCoords[2]**2)
                    Iyy+= masses[atom]*(atomMFFCoords[0]**2+atomMFFCoords[2]**2)
                    Izz+= masses[atom]*(atomMFFCoords[0]**2+atomMFFCoords[1]**2)
                    
                # write inertia terms
                I_out.write("{} {} {} {} {} \n".format(p2, i, Ixx, Iyy, Izz))

                # write out bond lengths
                bond_len.write("{} {} {} {} {} \n".format(p2, i, 
                np.linalg.norm(coordsMFF[i, p, 0, :] - coordsMFF[i, p, 1, :]),
                np.linalg.norm(coordsMFF[i, p, 0, :] - coordsMFF[i, p, 2, :]),
                np.linalg.norm(coordsMFF[i, p, 2, :] - coordsMFF[i, p, 1, :])))

                # (kg*m**2/s)(kg*m**2/s) / (g/mol * nm**2)(m**2/nm**2)(mol)(kg/g)
                # (kg*m**2/s)(kg*m**2/s) / (kg * m**2)
                # kg
                # convert inertia from Da*nm^2 to kg*m^2
                # B_x, B_y, and B_z, should all be in K -- CURRENTLY IN kJ
                Be_n = [hbar**2/(2*Ixx * nmToM**2 / (N_A * 1000)) / (1.e3),
                        hbar**2/(2*Iyy * nmToM**2 / (N_A * 1000)) / (1.e3),
                        hbar**2/(2*Izz * nmToM**2 / (N_A * 1000)) / (1.e3)]

                # iterate through permitted idxs to calculate the first energy 
                # contribution terms at this bead
                for j in posLeviCivita:
                    E1 = (1/Be_n[j[0]] - 1/Be_n[j[1]] - 1/Be_n[j[2]]) * \
                         (1-A_omega_tilda[j[0], j[0]]) / (4*self.tau**2)
                    
                    # convert from K^2 to kJ/mol using kB/mol
                    E1 *= (kB**2)/(1000**2)
                    E[0] += E1

                # add two remaining (non-iterating) energy terms
                E[1] += np.sum(Be_n)/(4) 
                E[2] += kB*3/(2*self.tau * 1.e3)
            
            # print("energy terms: ", E)
            # sum and convert to molar
            energy[i] = np.sum(E) * N_A
            # write the energy at this step to the file
            energy_out.write(str(i*self.stepsize*self.skipSteps) + " " + 
                             str(energy[i]) + "\n")

        energy_out.close()
        eul_out.close()
        I_out.close()
        bond_len.close()

        # if comparing to openMM, check that appropriate h5 objects have been 
        # unpacked
        if compareOpenMM:
            try:
                exists=(type(self.kineticEnergy_h5)==h5py._hl.dataset.Dataset and
                        type(self.potentialEnergy_h5)==h5py._hl.dataset.Dataset)
            except AttributeError:
                exists = False
            
            if not exists:
                self.unpackH5(attributes = ["kineticEnergy", "potentialEnergy"])

            # open a new data file to store values in
            fname = "openMM_energy.dat"
            if default_folder == True:
                out_path = os.path.join(self.root_path(), fname)
                out_path = os.path.realpath(out_path)
                
            else:
                out_path = os.path.realpath(fname)
            
            openmmEnergy = open(out_path, "w")

            for i in range(totSteps):
                openmmEnergy.write(str(i) + " ")
                openmmEnergy.write(str(i*self.stepsize*self.skipSteps) + " " + 
                         str(self.kineticEnergy_h5[i]) + " " +
                         str(self.potentialEnergy_h5[i]) + " " +
                         str(self.kineticEnergy_h5[i]+self.kineticEnergy_h5[i]) 
                         + "\n")
            openmmEnergy.close()

    def plotPotentialEnergy(self, add_to_sheet = False, default_folder = True):
        # check that coordinates from the corresponding .h5 file have been read
        print("Finding potential energy")
        try:
            obj_exists = type(self.potentialEnergy_h5) == h5py._hl.dataset.Dataset
        except AttributeError:
            obj_exists = False
        
        if not obj_exists:
            self.unpackH5(attributes = ["potentialEnergy"])

        stepSize = self.stepsize * self.skipSteps
        tRange = np.arange(0, self.steps*stepSize, stepSize)

        # must convert from kJ/mol to cm^-1
        # conversion factor found on this site: 
        # http://wild.life.nctu.edu.tw/class/common/energy-unit-conv-table.html
        conversion = 0.0119627

        print("Energy:\n", self.potentialEnergy_h5[:-1])

        invcmEnergy = self.potentialEnergy_h5[:-1]#/conversion

        plt.plot(tRange, invcmEnergy)
        plt.title("Potential energy")
        plt.ylabel("V (cm-1)")
        plt.xlabel("Time (fs)")

        out_file = "UEnergy.png"

        if default_folder == True:
            out_path = os.path.join(self.root_path(), out_file)
            out_path = os.path.realpath(out_path)
            
        else:
            out_path = os.path.realpath(out_file)

        plt.savefig(out_path)
        plt.close()

        if add_to_sheet:
            averageV = np.average(invcmEnergy)

            with open("v_avg_approx.txt", 'a') as f:
                f.write("{} {}\n".format(self.temperature, averageV))

    def comparePotentials():
        """ This function will allow for comparison between exact results and
            approx results of potential energy. Recall that the approx results 
            utilize the LJ coefficients"""
        exactFile = ""

    def positionHistogram(self):
        """ This function plots the positions of the different beads and atoms.
            The intent for this function is to illustrate that the various
            particles behave as free particles when forces are not applied. 
            
            Note: each particle and bead should act as an independent particle. 
            Therefore, the beads of each atom will be aggregated and 
            the norm of each coordinate will be plotted instead. The result 
            in each """
        self.unpackH5(attributes = ["coordinates"])

        fig, axes = plt.subplots(3, 1) # create a subplot per atom

        # nsteps, P, N, 3 -> flatten over the final index (atom)
        countsOx, binsOx = np.histogram(self.coordinates_h5[:, :, 1, 0], bins = 31, density = True)
        countsOy, binsOy = np.histogram(self.coordinates_h5[:, :, 1, 1], bins = 31, density = True)
        countsOz, binsOz = np.histogram(self.coordinates_h5[:, :, 1, 2], bins = 31, density = True)
        
        axes[0].stairs(countsOx, binsOx)
        axes[0].set_title("$O_{x}$")

        axes[1].stairs(countsOy, binsOy)
        axes[1].set_title("$O_{y}$")

        axes[2].stairs(countsOz, binsOz)
        axes[2].set_title("$O_{z}$")

        plt.show()

        plt.close()

    def velocityHistogram(self):
        """ This function plots the positions of the different beads and atoms.
            The intent for this function is to illustrate that the various
            particles behave as free particles when forces are not applied. 
            
            Note: each particle and bead should act as an independent particle. 
            Therefore, the beads of each atom will be aggregated and 
            the norm of each coordinate will be plotted instead. The result 
            in each """
        self.unpackH5(attributes = ["velocities"])

        fig, axes = plt.subplots(3, 1) # create a subplot per atom

        # nsteps, P, N, 3 -> flatten over the final index (atom)
        countsOx, binsOx = np.histogram(self.velocities_h5[:, :, 1, 0], bins = 31, density = True)
        countsOy, binsOy = np.histogram(self.velocities_h5[:, :, 1, 1], bins = 31, density = True)
        countsOz, binsOz = np.histogram(self.velocities_h5[:, :, 1, 2], bins = 31, density = True)
        
        axes[0].stairs(countsOx, binsOx)
        axes[0].set_title("$O_{x}$")

        axes[1].stairs(countsOy, binsOy)
        axes[1].set_title("$O_{y}$")

        axes[2].stairs(countsOz, binsOz)
        axes[2].set_title("$O_{z}$")

        plt.show()

        plt.close()

# CURRENT:
# beads, temperature, stepsize, period = 0, gamma = 1, 
# skipSteps = 1, filename_format = None, constraints = True,
# constraintTol = None, c60 = False, pill = False,
# waterType = "tip4p"
# print("creating trajectory")
# sim1 = trajectory(8, 150, 0.1, 1, 100, 1, constraints = False, version = "V76")
# print("finding autocorrelation")
# sim1.autocorrelation()
# sim1.plotPotentialEnergy(add_to_sheet=True)

print("creating trajectory:")
sim1 = trajectory(beads = 8, temperature = 25, stepsize = 1e-15, period = 1e-13,
                  gamma = 1/25)

# beads, temperature, stepsize, period = 0, gamma = 1, 
#                  skipSteps = 1, filename_format = None, constraints = True,
#                  constraintTol = None, c60 = False, pill = False,
#                  waterType = "tip4p", version = None
sim1.positionHistogram()
sim1.velocityHistogram()