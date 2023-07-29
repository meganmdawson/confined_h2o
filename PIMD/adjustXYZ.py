import numpy as np

# import c60 and h2o and compare their centroid positions. adjust the cs in c60
# positions such that they align with the CoM of h2o

waterfile = "H2O.xyz"
c60file = "C60.xyz"

H2Oq = np.loadtxt(fname = waterfile, skiprows = 1, dtype = np.float64)
C60q = np.loadtxt(fname = c60file, dtype = np.float64)

h2oCoM = np.mean(H2Oq, axis = 0)
c60CoM = np.mean(C60q, axis = 0)

print("the centre of mass for H2O is ", h2oCoM)
print("the centre of mass for the c60 is ", c60CoM)

# confirm that the c60 file is in angstrom
c60Rs = np.linalg.norm(C60q-c60CoM, axis = 1)
print("approx radius of c60", max(c60Rs))

# shift the centre of mass of the c60 to match that of the h2o
C60q = C60q - c60CoM + h2oCoM

# write over the c60 file such that it is properly centred
np.savetxt(c60file, C60q)