import numpy as np
from matplotlib import pyplot as plt
from file_manager import save, genFileName, checkSimExists

class WaterSystem():
    """ Parent class to describe the system parameters that are non-specific to
        the integrator used """
    def __init__(self, T, dt):
        """ initialize initial variables and update parameters: """
        # universal constants
        kB = 3.166811563e-6 # J/K
        DaToMe = 1822.89 # Da to electron mass (me/Da)
        toHartree = 1.5936254980079682e-3 # convert kcal/mol to hartree
        toBohr = 1.8903591682419658 # convert to bohr (bohr/Å)
        toTime = 41.35649296939619 # fs to A.U. time (A.U.t/fs)

        self.h_bar = 1 # atomic units

        # set up forces and potentials
        # NOTE: assumed that there is no 1/2 coefficient on the potentials
        self.QBF = lambda q, q0, a, D: (lambda d = a*(q-q0): \
                                        -D*(2*a*d -3*a*d**2 +7/3*a*d**3)) # quartic bond force
        self.HAF = lambda a, a0, k: -k*(a - a0) # harmonic angle force
        self.LJF = lambda r, s, e: (lambda b = (s/r)**6: -24*e*b*(2*b-1)/r) # lennard jones F

        self.QBP = lambda q, q0, a, D: (lambda d = a*(q-q0): D*(d**2-d**3+7/12*d**4)) # quartic bond potential
        self.HAP = lambda a, a0, k: 0.5 * k*(a - a0)**2 # harmonic angle potential
        self.LJP = lambda r, s, e: (lambda b = (s/r)**6: 4*e*b*(b-1)) # lennard jones pot.

        # load information from tip3p (https://docs.lammps.org/Howto_tip3p.html)
        mO = 15.99943 # mass, amu
        mH = 1.007947
        mC = 12.011

        cO = -0.834 # charges, e
        cH = 0.417
        cC = 0

        self.lOH = 0.9419 * toBohr # quartic bond length for OH bonds, Å -> bohr
        self.alpha_r = 2.287 / toBohr # quartic bond distance for OH bonds, 
                                      # 1/Å -> 1/bohr
        self.D_r = 485.72056 * toHartree # quartic bond force for OH bonds, 
                                              # kJ/mol -> hartree

        self.kOH_aharm = 367.5644 * toHartree # harmonic bending force for
                                                    # HOH bond, kcal/(mol*rad^2) 
                                                    # -> hartree/(rad^2)
        self.angOH_aharm = 1.8744836166419099 # harmonic bending angle for HOH bond (rad)

        # NOTE: double check the units for all params and make sure they are consistent
        # C-OH interaction for H2O@C60 -- Double check force params! 
        self.ljCH_sigma = 2.64 * toBohr # Å -> bohr
        self.ljCH_eps = 0.0256 * toHartree # 0.0256 kJ/mol -> Hartree

        self.ljCO_sigma = 3.372 * toBohr # Å -> bohr
        self.ljCO_eps = 0.1039 * toHartree # 0.1039 kcal/mol -> hartree

        # create lists containing the force parameters to make indexing easier
        self.ms = (np.array([mO, mH, mH]) * DaToMe).reshape(1, 3, 1) # convert masses to m_e from Da 
        self.cs = np.array([cO, cH, cH]) # NOTE: this should be converted to electronic charge?

        # load positions of atoms in water (Å)
        # shape: (3, 3); (O: x, y, z; H1: x, y, z; H2: x, y, z)
        waterfile = "H2O.xyz"
        self.H2Oq = np.loadtxt(fname = waterfile, skiprows = 1, dtype = np.float64) * toBohr
        # check that the loaded file takes the correct shape
        if self.H2Oq.shape == (3, 3):
            print("The initial loaded positions have the correct shape")
        else:
            errorMsg="The initial loaded positions should have shape (3,3) but have: {}"
            exit(errorMsg.format(self.H2Oq.shape))

        # load positions of atoms in C60
        c60file = "C60.xyz"
        C60q = np.loadtxt(fname = c60file, dtype = np.float64) * toBohr

        """ initialize system variables """
        self.T = T # K
        self.dt_fs = dt 
        self.dt = dt * toTime # fs -> A.U.
        print("timestep in ps:", self.dt_fs)
        print("timestep in a.u.:", self.dt)
        
        self.beta = 1/(T * kB)

    def waterGeo(self, qs, plot = False):
        """ determines the body-fixed frame for an input water molecule along with 
            distances and relevant angles 
            
            Input:
            qs: np.array
                Lab-fixed frame positions of the water molecule"""
        
        P = qs.shape[0]

        # define molecule-fixed positions and intramolecular distances
        qsBFF = qs - np.tile(qs[:, 0, :], 3).reshape(P,3,3)
        
        rH1 = np.linalg.norm(qsBFF[:, 1, :], axis = 1).reshape(P, 1)
        rH2 = np.linalg.norm(qsBFF[:, 2, :], axis = 1).reshape(P, 1)

        # calculate intramolecular angles
        dot = np.zeros(P).reshape(P, 1)
        for p in range(P): # calc dot product for each bead
            dot[p] = np.dot(qsBFF[p, 1, :], qsBFF[p, 2, :])
        dot = dot/(rH1 * rH2)
        aHOH = np.arccos(dot)

        # define plane of water molecule
        molPlane = np.cross(qsBFF[:, 1, :], qsBFF[:, 2, :])
        molPlane = molPlane/(np.linalg.norm(molPlane, axis = 1).reshape(P, 1))

        # define normalized vectors pointing from oxygen to hydrogen
        vOH1 = qsBFF[:, 1, :]/rH1
        vOH2 = qsBFF[:, 2, :]/rH2

        # define normalized vectors perpendicular to the oxygen-hydrogen connection 
        # that lie within the molecular plane
        vOH1p = np.cross(vOH1, molPlane)
        vOH2p = np.cross(molPlane, vOH2)

        # plot the direction vectors that diff forces will point in according to 
        # geometry defined in this function
        if plot:
            # plot forces etc
            fig = plt.figure()
            ax = plt.axes(projection = '3d')
            # plot the first H2O
            plotH2O = np.zeros((4, 3))
            plotH2O[0:3, :] = qs[0, :, :]
            plotH2O[3, :] = qs[0, 0, :]
            ax.plot3D(plotH2O[:, 0], plotH2O[:, 1], plotH2O[:, 2], c = "k")
            ax.scatter(plotH2O[0, 0],plotH2O[0, 1], plotH2O[0, 2], c = 'b')
            ax.scatter(plotH2O[1:3, 0],plotH2O[1:3, 1], plotH2O[1:3, 2], c = 'r')

            # plot the bond forces 
            h1hbf_plt = np.vstack([qs[0, 1, :], qs[0, 1, :] + vOH1[0] ])
            h2hbf_plt = np.vstack([qs[0, 2, :], qs[0, 2, :] + vOH2[0] ])
        
            # plot angle forces
            h1haf_plt = np.vstack([qs[0, 1, :], qs[0, 1, :] + vOH1p[0] ])
            h2haf_plt = np.vstack([qs[0, 2, :], qs[0, 2, :] + vOH2p[0] ])
            ohaf_plt = np.vstack([qs[0, 0, :], qs[0, 0, :] - vOH2p[0] - vOH1p[0]])

            # normal plane
            centre = np.mean(qs[0], axis = 0)
            normPlane = np.vstack([centre, centre + molPlane[0]])

            # plot reference directions
            ax.plot3D(h1haf_plt[:, 0], h1haf_plt[:, 1], h1haf_plt[:, 2], c = 'b', 
                    label = "HAF_ref")
            ax.plot3D(h2haf_plt[:, 0], h2haf_plt[:, 1], h2haf_plt[:, 2], c = 'b')
            ax.plot3D(ohaf_plt[:, 0], ohaf_plt[:, 1], ohaf_plt[:, 2], c = 'b')
            
            ax.plot3D(h1hbf_plt[:, 0], h1hbf_plt[:, 1], h1hbf_plt[:, 2], c = 'r', 
                    label = "HBF_ref")
            ax.plot3D(h2hbf_plt[:, 0], h2hbf_plt[:, 1], h2hbf_plt[:, 2], c = 'r')
            
            ax.plot3D(normPlane[:, 0], normPlane[:, 1], normPlane[:, 2], c = 'y', 
                    label = "norm")
            bondLen = 'OH1 = {H1}\nOH2 = {H2}\ntheta = {HOH}'
            outText = bondLen.format(H1 = rH1[0], H2 = rH2[0], HOH = aHOH[0])
            print("outText:", outText)
            ax.text(x = 0, y = -1, z = 0, s = outText, fontsize=15)

            ax.legend()
            
            plt.savefig("forces")
            plt.show()
            plt.close()

        return aHOH, rH1, rH2, vOH1, vOH2, vOH1p, vOH2p

    def nullForces(self, qs):
        relForces = np.zeros(qs.shape, dtype = np.float32)

        return relForces

    # create a function to add all relevant forces
    def forces(self, qs, c60 = False, verbose = False, plot = False):
        """ Defines forces between a given atom and others in the system. In this 
            model, harmonic bond forces and LJ forces are considered between the 
            two Hs and O. 
            Outputs a row vector of shape (3, 1), relForces, containing the force 
            for each atom. The first index corresponds to oxygen and the following
            two correspond to the two hydrogen atoms.

            This is the standard set of forces for the TIP3P model.

            Input:
            qs: np.array
                Array with shape (P,3,3) containing the positions of each atom. The 
                first row contains the oxygen position while the second and third
                contain the hydrogen positions."""
        
        # NOTE: will have to retrofit for carbon at some point
        P = qs.shape[0]
        invP = 1/P
        relForces = np.zeros((P, 3, 3))  

        aHOH, rH1, rH2, vOH1, vOH2, vOH1p, vOH2p = self.waterGeo(qs)

        """ harmonic angle force """
        # determine the angular harmonic oscillator force magnitude
        # see https://mines-paristech.hal.science/hal-00924263/document for partial
        # derivative explanation
        angleF = self.HAF(aHOH, self.angOH_aharm, self.kOH_aharm)

        # determine the directions of the forces
        h1HAF = angleF.reshape(P, 1) * vOH1p / rH1
        h2HAF = angleF.reshape(P, 1) * vOH2p / rH2

        """ harmonic bond forces """
        # determine the harmonic bond forces
        HBF1_mag = (self.QBF(rH1, self.lOH, self.alpha_r, self.D_r)()).reshape(P, 1)
        HBF2_mag = (self.QBF(rH2, self.lOH, self.alpha_r, self.D_r)()).reshape(P, 1)

        HBF1 =  HBF1_mag * vOH1
        HBF2 =  HBF2_mag * vOH2

        # if verbose:
        #     print("aHOH.shape", aHOH.shape)
        #     print("The bond angle is ", aHOH)
        #     print("The harmonic angle is ", angOH_aharm)
        #     print("angleF: ", angleF)
        #     print("HBF1: ", HBF1)
        #     print("HBF2: ", HBF2)

        """ LJ interactions with C60"""
        # iterate over carbon in C60
        if c60:
            for c in range(60):
                # find distances between the replicas of each atom in H2O
                rHOH_C = qs - self.C60q[c].reshape(1, 1, 3) 
                magHOH_C = np.linalg.norm(rHOH_C, axis = 2) 
                dirHOH_C = rHOH_C/magHOH_C.reshape(P, self.ms.size, 1)*invP

                # add force component for O-C interaction
                COforce = self.LJF(magHOH_C[:,0], self.ljCO_sigma, \
                                   self.ljCO_eps)().reshape(8, 1)
                relForces[:, 0] += COforce * dirHOH_C[:, 0, :]

                # add force components for H-C interactions
                CHforce1 = self.LJF(magHOH_C[:, 1], self.ljCH_sigma, \
                                    self.ljCH_eps)().reshape(8, 1)
                relForces[:, 1] += CHforce1 * dirHOH_C[:, 1, :]

                CHforce2 = self.LJF(magHOH_C[:, 2], self.ljCH_sigma, \
                                    self.ljCH_eps)().reshape(8, 1)
                relForces[:, 2] += CHforce2 * dirHOH_C[:,0,:]
                
        if verbose:
            print("max LJ C-OH interaction:", np.max(relForces))

        # populate the force array
        relForces[:, 0] = (-HBF1 - HBF2 - h1HAF - h2HAF) * invP
        relForces[:, 1] = (h1HAF + HBF1) * invP
        relForces[:, 2] = (h2HAF + HBF2) * invP

        if verbose:
            # sanity check: ensure that the sum of all forces = 0:
            print("The sum of all forces is: ", np.sum(relForces))
            print("HBF1[0]: ", HBF1[0])
            print("HBF2[0]: ", HBF2[0])
            print("h1HAF[0]: ", h1HAF[0])
            print("h2HAF[0]: ", h2HAF[0])
            print("The sum of h2HAF and h2HAF is non-zero: ", np.average(h1HAF + h2HAF, axis = 0))
        
        # plot the direction vectors that diff forces will point in according to 
        # geometry defined in this function
        if plot:
            # plot forces etc
            fig = plt.figure()
            ax = plt.axes(projection = '3d')
            # plot the first H2O
            plotH2O = np.zeros((4, 3))
            plotH2O[0:3, :] = qs[0, :, :]
            plotH2O[3, :] = qs[0, 0, :]
            ax.plot3D(plotH2O[:, 0], plotH2O[:, 1], plotH2O[:, 2], c = "k")
            ax.scatter(plotH2O[0, 0],plotH2O[0, 1], plotH2O[0, 2], c = 'b')
            ax.scatter(plotH2O[1:3, 0],plotH2O[1:3, 1], plotH2O[1:3, 2], c = 'r')

            # plot the bond forces 
            f1 = np.sign(HBF1_mag[0])
            f2 = np.sign(HBF2_mag[0])
            h1hbf_plt = np.vstack([qs[0, 1, :], qs[0, 1, :] + f1 * vOH1[0]])
            h2hbf_plt = np.vstack([qs[0, 2, :], qs[0, 2, :] + f2 * vOH2[0]])
        
            # plot angle forces
            h1haf_plt = np.vstack([qs[0, 1, :], 
                                   qs[0, 1, :] + np.sign(angleF[0]) * vOH1p[0]])
            h2haf_plt = np.vstack([qs[0, 2, :], 
                                   qs[0, 2, :] + np.sign(angleF[0]) * vOH2p[0]])
            ohaf_plt = np.vstack([qs[0, 0, :], 
                                  qs[0, 0, :]-np.sign(angleF[0])*(vOH2p[0]+vOH1p[0])])
            print("np.sign(angleF[0]):", np.sign(angleF[0]))
            print("vOH1p[0]:", vOH1p[0])

            # plot reference directions
            ax.plot3D(h1haf_plt[:, 0], h1haf_plt[:, 1], h1haf_plt[:, 2], c = 'b', 
                    label = "HAF_ref")
            ax.plot3D(h2haf_plt[:, 0], h2haf_plt[:, 1], h2haf_plt[:, 2], c = 'b')
            ax.plot3D(ohaf_plt[:, 0], ohaf_plt[:, 1], ohaf_plt[:, 2], c = 'b')
            
            ax.plot3D(h1hbf_plt[:, 0], h1hbf_plt[:, 1], h1hbf_plt[:, 2], c = 'r', 
                    label = "HBF_ref")
            ax.plot3D(h2hbf_plt[:, 0], h2hbf_plt[:, 1], h2hbf_plt[:, 2], c = 'r')

            bondLen = 'OH1 = {H1}\nOH2 = {H2}\ntheta = {HOH}'
            outText = bondLen.format(H1 = rH1[0], H2 = rH2[0], HOH = aHOH[0])
            print("outText:", outText)
            ax.text(x = 0, y = -1, z = 0, s = outText, fontsize=15)

            ax.legend()
            
            plt.savefig("forces")
            plt.show()
            plt.close()

        return relForces

# def classicalEnergy():
#     """ Defines forces between a given atom and others in the system. In this 
#         model, harmonic bond forces and LJ forces are considered between the 
#         Outputs a row vector of shape (3, 1), relForces, containing the force 
#         for each atom. The first index corresponds to oxygen and the following
#         two correspond to the two hydrogen atoms.

#         This is the standard set of forces for the TIP3P model.

#         Input:
#         qs: np.array
#             Array with shape (3,3) containing the positions of each atom. The 
#             first row contains the oxygen position while the second and third
#             contain the hydrogen positions."""        

    def velBoltzDistrib(self, m, shape, plot = False):
        """ Return random velocities according to the Boltzmann distribution. Used 
            to generate initial velocities for the simulation. Returns an 
            array populated with xyz velocities, where n is the number of particles.
            
            Input:
            m: np.array, float
                Masses of the atoms.
            size: tuple, int
                Shape of the output array. Generally takes the form 
                (P, nParticles, 3)
    """ 
        scale = 1/(m * self.beta)**0.5 # pv = np.exp(-m*np.power(v, 2)*beta/2)
        v = np.zeros(shape)
        shapei = (1, shape[1], shape[2])
        for i in range(shape[0]): # iterate over all beads and assign each bead its
                                # own
            v[i] = np.random.normal(loc = 0, scale = scale, size = shapei)

        if plot:
            counts, bins = np.histogram(v)

            plt.stairs(counts, bins)
            plt.show()
            plt.close()

        # convert to Å/s
        v = v
        # v = np.random.normal(size = shape) * 1e12

        return v

    def energyEst(self):
        """ Energy estimator as defined by Guillon et al.
            """
        i = 0


""" define integrators and functions to run simulations: """
class WhitePILE(WaterSystem):

    def __init__(self, T, dt, gamma, P, nSteps, force = "forces", 
                 folder = "data", integrator = "wPILE"):
        super().__init__(T, dt)
        self.gamma_inv_s = gamma 
        self.gamma = gamma #* 2.418e-17 # convert gamma from s^-1 to  A.U^-1
        self.P = P
        self.folder = folder
        self.nSteps = int(nSteps)
        self.integrator = integrator

        self.force  = self.__getattribute__(force) # call force from parent obj

        self.beta_n = self.beta/P
        
        omega_n = 1/(self.beta_n * self.h_bar)
        self.omegas = np.zeros(P)
        self.omegas[0] = self.gamma * 0.5
        self.omegas[1:] = 2*omega_n * np.sin(np.arange(1, P)*np.pi/P) # eqn 20

        self.gammas = np.zeros(P) # eqn 36
        self.gammas[0] = self.gamma
        self.gammas[1:] = 2 * self.omegas[1:]

        # generate propagation matrix
        omegasDt = self.omegas * self.dt
        self.prop = np.zeros((self.P, 2, 2))
        
        for k in range(self.P): # populate the matrix
            # calculate velocity and positions (desired):
            self.prop[k, 0, 0] = np.cos(omegasDt[k])
            self.prop[k, 0, 1] = -self.omegas[k]*np.sin(omegasDt[k])
            
            self.prop[k, 1, 0] = 1/(self.omegas[k])*np.sin(omegasDt[k])
            self.prop[k, 1, 1] = np.cos(omegasDt[k])

    def generateRing(self, r = 5, plot = False):
            """ Generate a ring shape that the replicas can be positioned in. The output
                ring lies in the XY plane.
            
            Input:
            r: float
                Radius of the ring in r_bohr
            
            Output:
            ring: nd.array
                Array with shape (P, 3) which contains the positions
                of the replicas. 
            """

            angles = np.linspace(0, 2 * np.pi, self.P)
            ring = np.zeros((self.P, 1, 3), dtype = np.float64)

            ring[:, 0, 0] = np.sin(angles) * r
            ring[:, 0, 1] = np.cos(angles) * r
            ring[:, 0, 2] = np.tan(angles) * r

            if plot:
                plt.scatter(ring[:, 0], ring[:, 1])
                plt.savefig("circle.png")

            return ring

    def toNormalMode(self, bA):
        """ Function that takes the momentum or position of particle duplicates 
            (ie bead representation) and converts to normal mode representation

            The initial method as explicitly defined in the paper has been moved to
            the code graveyard. If retrieved, the function genTransform should also
            be revived.
            
            bAq.shape = (P: number of beads, N: number of particles, 3: dimensions)
            bAp.shape = (P: number of beads, N: number of particles, 3: dimensions)"""
        
        dims = bA.shape
        pHalf = int(0.5 * self.P)      

        # construct fourier transformed matrix
        fftA = np.zeros(bA.shape, dtype = complex)
        for i in range(dims[1]): # iterate over the different particles
            for j in range(dims[2]): # iterate over the dimensions
                fftA[:, i, j] = np.fft.fft(bA[:, i, j], norm = 'forward')

        # find normal modes from the fourier transform
        nmA = np.zeros(bA.shape, dtype = np.float64)
        nmA[0, :, :] = fftA[0, :, :].real
        for p in range(1, pHalf): # iterate over complex elements and decompose
            nmA[p, :, :] = fftA[p, :, :].real
            nmA[-p, :, :] = fftA[p, :, :].imag
        nmA[pHalf, :, :] = fftA[pHalf, :, :].real

        return nmA

    def toBead(self, nmAp):
        """ Function that takes the momentum or position of particles
            (in normal mode representation) and converts to bead representation"""

        dims = nmAp.shape
        pHalf = int(self.P/2)
        fftA = np.zeros(nmAp.shape, dtype = complex)

        fftA[0, :, :] = nmAp[0, :, :] + 0j

        for p in range(1, pHalf): # reconstruct the complex fft output from the nm
            fftA[p, :, :] = nmAp[p, :, :] + 1j*nmAp[-p, :, :]
            fftA[-p, :, :] = fftA[p, :, :].real - 1j*fftA[p, :, :].imag 
            # NOTE: in the original program, the fftA[-p] term was set to equal 
            #       fftA[p] (as in following commented code line). No changes were 
            #       made to the sign of the imaginary component as done in this 
            #       program.
            # fftA[-p, :, :] = fftA[p, :, :]

        fftA[pHalf] = nmAp[pHalf, :, :] + 0j

        # bAp = np.fft.ifftn(fftA)
        
        # reverse the transform on the reconstructed fft matrix
        bAp = np.zeros(nmAp.shape, dtype = np.float64)
        for i in range(dims[1]): # iterate over the different particles
            for j in range(dims[2]): # iterate over the dimensions
                bAp[:, i, j] = np.fft.ifft(fftA[:, i, j], norm = 'forward')
        return bAp

    def addNoise(self, aP):
        """ Accept a momentum array in bead representation 
            aP: (P, N, 3)"""
        
        # convert to normal mode representation
        aP_nm = self.toNormalMode(aP)

        # make shape of gammas compatible with aP
        gammas = self.gammas.reshape(aP.shape[0], 1, 1)
        
        # calculate coefficients and random number
        c1 = np.exp(-self.dt * 0.5* gammas)
        c2 = np.sqrt(1 - c1*c1)
        # reshape for appropriate multiplication with aP
        c1 = c1.reshape(self.P, 1, 1)
        c2 = c2.reshape(self.P, 1, 1)

        zeta = np.random.standard_normal(aP.shape)
        # create a 3x3 array with masses (a row per atom type)
        masses = np.zeros((3, 3))
        masses[:] = self.ms

        aP_nm = c1*aP_nm + np.sqrt(1/(self.beta_n*masses.T)) *c2*zeta

        # transform back to bead representation
        aP = self.toBead(aP_nm)

        return aP

    def wPropagate(self, v, q, verbose = False, fixCentroid = True):
        """ Propagates the normal mode form of the trajectory at a given set of 
            velocities and positions. Prop is the propagator defined in 
            WhitePILE's __init__ function (shape (P, 2, 2)). This applies the 
            thermostat in the bead representation instead of the normal mode
            representation"""

        if verbose:
            print("initial q", q)
            print("initial v", v)

        v1 = np.zeros((self.P, self.ms.size, 3), dtype = np.cdouble)
        q1 = np.zeros((self.P, self.ms.size, 3), dtype = np.cdouble)

        # add noise before propagation (1/2)
        v = self.addNoise(v) # equations 27-29 (but vel)

        if verbose:
            print("post-noise v:", v)

        F = self.force(q) # convert from kN to N
        v += F/self.ms * self.dt * 0.5 # v0.5, eqn 21
        if verbose:
            print("change in velocity 0.5dt1:", v)

        # eqn 22: convert velocity and position to normal mode
        q = self.toNormalMode(q)
        v = self.toNormalMode(v)

        if fixCentroid:
            init_k_iter = 1
        else:
            init_k_iter = 0

        # eqn 23: propagate both velocities and positions
        # not the most efficient -- but iterate over array slices for now
        for k in range(init_k_iter, self.P): # iterate over vibrational modes
            for i in range(self.ms.size): # iterate over the number of molecules
                for dim in range(3):
                    out = np.matmul(self.prop[k], 
                                    np.array([v[k, i, dim], q[k, i, dim]]))
                    v1[k, i, dim] = out[0] 
                    q1[k, i, dim] = out[1]

        # eqn 24: convert back to bead representation
        v = self.toBead(v1)
        q = self.toBead(q1)

        if verbose: 
            print("post-prop q: \n", q)

        # eqn 25: integrate trajectory
        F = self.force(q)
        v += F/self.ms * self.dt * 0.5 # v0.5, eqn 21

        if verbose: 
            print("change in velocity 0.5dt2: \n", v)
        
        # eqns 27-29, add thermal noise again
        v = self.addNoise(v)

        if verbose:
            print("post-noise v 2: \n", v)
            print("post-noise q: \n", q)

        return v, q
    
    # set up Langevin integrator
    def genTrajectory(self, verbose = False, genRing = False, testFFT = False):
        """ Function to generate a trajectory using the path integral langevin 
            equation (PILE) integrator defined by Ceriotti et al. (2010); 
            https://doi.org/10.1063/1.3489925

            This simply uses ns = 0 (white noise)

            q.shape = (P: number of beads, N: number of particles, 3: dimensions)
            v.shape = (P: number of beads, N: number of particles, 3: dimensions)
        """
        # NOTE: replace 8 in the below eqn with whatever it is that P equals

        # check that the simulation has not already been run
        trajName = genFileName(self.P, self.T, self.folder, 
                               integrator = self.integrator)
        
        simLen = self.dt_fs * self.nSteps * 1e-3 # fs -> ps

        checkSimExists(trajName, self.gamma_inv_s, simLen, self.dt_fs)
        
        # initialize variables for propagation and noise
        dims = (self.nSteps, self.P, self.ms.size, 3)

        # initialize q0 and v0
        qs = np.zeros(dims, dtype = np.float64)
        vs = np.zeros(dims, dtype = np.float64)

        qs[0] = np.broadcast_to(self.H2Oq, (self.P, self.ms.size, 3))
        vs[0] = self.velBoltzDistrib(self.ms, (self.P, self.ms.size, 3))
        
        # establish a ring shape to adjust particle positions so they do not start
        # in the same spots
        if genRing:
            ring = self.generateRing(r = 5) # radius = 5 angstroms
            qs[0] += ring

        # show initial conditions
        if verbose:
            print("initial v:", vs[0])
            print("initial q:", qs[0])
            print("ring:", ring)

        # propagate through trajectory with the wPILE integrator
        for l in range(1, self.nSteps):
            l0 = l - 1
            vs[l], qs[l] = self.wPropagate(vs[l0], qs[l0])

        # aside: check that the transformation between bead and nm rep. is 
        # functional
        if testFFT:
            from tests import nmTransform
            nmTransform(self.toNormalMode, self.toBead, vs[-1], qs[-1])
            
        # save the trajectory
        datDict = {"velocities": vs,
                   "coordinates": qs} # save output in bohr/time (a.u.) and bohr
        
        save(trajName, self.gamma_inv_s, simLen, self.dt_fs, data = datDict)


if __name__ == "__main__":
    print("Generating simulation object:")
    T = 200
    gamma = 1/(T)
    wpile = WhitePILE(T = T, dt = 1.0, gamma = gamma, P = 128, nSteps = 1e4, 
                      force="nullForces", folder = "data", integrator = "wPILE")
    
    print("Generating trajectory:")
    wpile.genTrajectory()
    print("Simulation complete!")