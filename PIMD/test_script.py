import numpy as np
from constrained import toNormalMode
from constrained import toBead
from constrained import addNoise
from matplotlib import pyplot as plt

def testFourier():
    aP = np.random.rand(6*3*3).reshape(6, 3, 3)

    nmAp = toNormalMode(aP)

    bAp = toBead(nmAp)

    if np.sum(aP - bAp) == 0.:
        print("alles gut")
    else:
        print("alles nicht so gut :(")

        print("Sum of differences = ", np.sum(aP - bAp))

        print("original matrix: \n", aP)

        print("returned matrix: \n", bAp)

    # fftAp = fft(aP)
    # ifftAp = ifft(fftAp)

    # print(fftAp)
    # print(ifftAp)

    # fftAp = fft(aP)
    # ifftAp = ifft(fftAp)

    # print(fftAp)
    # print(ifftAp)

def testNoise():
    # check that the noise occupies the expected distribution when plotted.
    aP = np.random.rand(6*3*3).reshape(6, 3, 3)
    

def testBondForceOH():
    # range of positions to test over (m)
    testx = np.arange(1, 20, 0.2) * 1e-11

    N_A = 6.02214e23 # mol^-1
    invN_A = 1/N_A

    """ quartic bond test """
    # QBF = lambda q, q0, a, D: -D*(2*a*(a*(q-q0)) - 3*a*(a*(q-q0))**2 + 7/3*a*(a*(q-q0))**3)
    QBF = lambda q, q0, a, D: (lambda d = a*(q-q0): -D*(2*a*d -3*a*d**2 +7/3*a*d**3)) # quartic bond force

    # load information from tip3p (https://docs.lammps.org/Howto_tip3p.html)
    mO = 15.99943 # mass, amu
    mH = 1.007947
    mC = 12.011

    cO = -0.834 # charges, e
    cH = 0.417
    cC = 0

    D_r = 485.72056 * invN_A # quartic bond force for OH bonds, 
                            # kJ
    alpha_r = 2.287 * 1e10 # quartic bond distance for OH bonds,
                        # 1/Å -> 1/m
    lOH_q = 0.9419 * 1e-10 # quartic bond length for OH bonds, Å -> m

    """ harmonic bond test """
    HBF = lambda q, q0, k: -2*k*(q - q0) # harmonic bond force

    kOH_harm = 1882.8 * invN_A * 1e20 # harmonic bond force for OH bonds, 
                                    # kJ/(mol*Å^2) * mol * (Å/m)^2 -> kJ
    lOH_h = 0.9572 * 1e-10 # harmonic bond length for OH bonds, Å -> m

    quarticForces = QBF(testx, lOH_q, alpha_r, D_r)()
    harmonicForces= HBF(testx, lOH_h, kOH_harm)

    plt.plot(testx*1e10, quarticForces, c = 'b', label = "QTIP4P OH stretch")
    plt.plot(testx*1e10, harmonicForces, c = 'r', label = "TIP3P OH stretch")
    plt.ylabel("Force (N)")
    plt.xlabel("OH bond length (Angstrom)")
    plt.legend()
    plt.savefig("force_comparison.jpg")

if __name__ == "__main__":
    testFourier()